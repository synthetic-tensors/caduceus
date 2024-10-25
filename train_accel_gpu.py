import logging
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
#from utils.training import count_parameters #, move_to
import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed

from accel_model import SequenceModule
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
        set_seed(config.train.seed)
    model = SequenceModule(config)
    print(model)
    train_dl, eval_dl = model.train_dataloader(), model.val_dataloader()

    #config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

    # Initialise your wandb run, passing wandb parameters and any config information
    init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
    accelerator.init_trackers(
        project_name="CaduceusCP",
        config=dict(config),
        init_kwargs=init_kwargs
        )

    #accelerator.print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
    #accelerator.print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
    #accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")
    num_training_steps = 1 * len(train_dl)

    # Set zero weight decay for some params
    if 'optimizer_param_grouping' in model.hparams.train:
        add_optimizer_hooks(model, **model.hparams.train.optimizer_param_grouping)
    # Normal parameters
    all_params = list(model.parameters())
    params = [p for p in all_params if not hasattr(p, "_optim")]
    optimizer = utils.instantiate(registry.optimizer, model.hparams.optimizer, params)
    del model.hparams.optimizer._name_

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    print("Hyperparameter groups:", hps)  # TODO: log.info throws error because hps is list of dicts
    for hp in hps:
        params = [p for p in all_params if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **model.hparams.optimizer, **hp}
        )

    # Layer Decay
    if model.hparams.train.layer_decay['_name_'] is not None:
        get_num_layer = utils.instantiate(
            registry.layer_decay,
             model.hparams.train.layer_decay['_name_'],
            partial=True,
        )

        # Go through all parameters and get num layer
        layer_wise_groups = {}
        num_max_layers = 0
        for name, p in model.named_parameters():
            # Get layer id for each parameter in the model
            layer_id = get_num_layer(name)

            # Add to layer wise group
            if layer_id not in layer_wise_groups:
                layer_wise_groups[layer_id] = {
                    'params': [],
                    'lr': None,
                    'weight_decay': model.hparams.optimizer.weight_decay
                }
            layer_wise_groups[layer_id]['params'].append(p)

            if layer_id > num_max_layers:
                num_max_layers = layer_id

        # Update lr for each layer
        for layer_id, group in layer_wise_groups.items():
            group['lr'] = model.hparams.optimizer.lr * (
                    model.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

        # Reset the torch optimizers param groups
        optimizer.param_groups = []
        for layer_id, group in layer_wise_groups.items():
            optimizer.add_param_group(group)

    # Print optimizer info for debugging
    keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
    utils.train.log_optimizer(logger, optimizer, keys)

    lr_scheduler = utils.instantiate(
        registry.scheduler, model.hparams.scheduler, optimizer
    )
    scheduler = {
        "scheduler": lr_scheduler,
        "interval": model.hparams.train.interval,  # 'epoch' or 'step'
        "monitor": model.hparams.train.monitor,
        "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
    }

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
         model, optimizer, train_dl, eval_dl, lr_scheduler
         )
    if accelerator.is_main_process:
        progress_bar = tqdm(range(num_training_steps), initial = 0 * len(train_dl))

    # Start model training and defining the training loop
    model.train()
    world_size = torch.cuda.device_count()
    print(world_size)
    for epoch in range(0,1):
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            # Training
            #print(f'forward on {dist.get_rank()}')
            if world_size > 1:
                loss = model.module._shared_step(batch, batch_idx, prefix="train")
            else:
                loss = model._shared_step(batch, batch_idx, prefix="train")
            #batch = move_to(batch, device)
            #print(f'backward on {dist.get_rank()}')
            accelerator.backward(loss)
            #print(f'optimize on {dist.get_rank()}')
            optimizer.step()
            lr_scheduler.step()
            if accelerator.is_main_process:
                progress_bar.update(world_size)
            accelerator.log({'loss':loss, 'grad_norm':get_grad_norm(model),'param_norm':get_param_norm(model)})
    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    accelerator.end_training()

if __name__ == "__main__":
    main()
