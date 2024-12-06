import logging
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
#from utils.training import count_parameters #, move_to
import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from src.dataloaders import SequenceDataset  # TODO make registry

#from accel_model import SequenceModule
from caduceus.modeling_esp import ESPForMaskedLM
from caduceus.configuration_esp import ESPConfig
from src.utils import registry
import src.utils as utils
from src.utils.train import get_grad_norm, get_param_norm
from src.utils.optim_groups import add_optimizer_hooks
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    #from accelerate import DistributedDataParallelKwargs
    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #True)
    #accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    if config.train.seed is not None:
        set_seed(42) #config.train.seed)
    model_config = ESPConfig(**config.model.config)
    model = ESPForMaskedLM(model_config)
    print(model)
    #print(model_config)

    # Dataset arguments
    print(SequenceDataset.registry.keys())
    self.dataset = SequenceDataset.registry[config.dataset._name_](
        **config.dataset
    )

    eval_dl = self.dataset.val_dataloader(**config.loader)
    train_dl = self.dataset.dataloader(**config.loader)

    config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

    # Initialise your wandb run, passing wandb parameters and any config information
    init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
    accelerator.init_trackers(
        project_name="ESP",
        config=dict(config),
        init_kwargs=init_kwargs
        )

    accelerator.print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
    accelerator.print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
    accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")
    num_training_steps = 1 * len(train_dl)

    # Set zero weight decay for some params
    if 'optimizer_param_grouping' in model.hparams.train:
        add_optimizer_hooks(model, **model.hparams.train.optimizer_param_grouping)
    # Normal parameters
    all_params = list(model.parameters())
    params = [p for p in all_params if not hasattr(p, "_optim")]
    optimizer = utils.instantiate(registry.optimizer, model.hparams.optimizer, params)
    del model.hparams.optimizer._name_
    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    #model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dl, eval_dl, lr_scheduler
    #     )
    model,optimizer,train_dl,eval_dl = accelerator.prepare(model, optimizer, train_dl, eval_dl)
    
    if accelerator.is_main_process:
        progress_bar = tqdm(range(num_training_steps), initial = 0 * len(train_dl))

    # Start model training and defining the training loop
    model.train()
    world_size = torch.cuda.device_count()
    #print(world_size)
    #print(train_dl.sampler)
    for epoch in range(0,1):
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            # Training
            print(f'{dist.get_rank()} - {len(batch)} * {batch[0].shape}')
            if world_size > 1:
                loss = model.module._shared_step(batch, batch_idx, prefix="train")
            else:
                loss = model._shared_step(batch, batch_idx, prefix="train")
            accelerator.backward(loss)
            rank = dist.get_rank() if dist.is_initialized() else 0
            optimizer.step()
            #lr_scheduler.step()
            if accelerator.is_main_process:
                progress_bar.update(world_size)
            accelerator.log({'loss':loss, 'grad_norm':get_grad_norm(model),'param_norm':get_param_norm(model)})
    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    accelerator.end_training()

if __name__ == "__main__":
    main()
