import logging
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
#from utils.training import count_parameters #, move_to
import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

#from src.dataloaders import SequenceDataset  # TODO make registry
import datasets
#from accel_model import SequenceModule
from caduceus.modeling_esp import ESPForMaskedLM
from caduceus.configuration_esp import ESPConfig
from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.utils import registry
import src.utils as utils
from src.utils.train import get_grad_norm, get_param_norm
from src.utils.optim_groups import add_optimizer_hooks
from omegaconf import OmegaConf
from src.dataloaders.utils.mlm import mlm_esp_getitem, mlm_getitem
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn(batch, mlm_probability=0.15):
    seq_ids, seq_targets, expr_values, expr_targets = [], [], [], []
    for sample in batch:
        input_ids, input_vals, sgrna = sample['input_ids'], sample['input_vals'], sample['sgrna']
        seq_id, seq_target = mlm_getitem(
            input_ids,
                mlm_probability=mlm_probability,
                tokenizer=CaduceusTokenizer(model_max_length=2**23),
            )

        expr_value, expr_target = mlm_esp_getitem(
            input_vals,
            mlm_probability=mlm_probability,
        )
        seq_ids.append(seq_id)
        seq_targets.append(seq_target)
        expr_values.append(expr_value)
        expr_targets.append(expr_target)
    return {'input_ids': torch.stack(seq_ids), \
            'values': torch.cat(expr_values), \
            'labels': torch.stack(seq_targets), \
            'value_labels': torch.cat(expr_targets)}
    # corresponding model inputs are (input_ids, values, labels, value_labels)

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
    #print(SequenceDataset.registry.keys())
    #self.dataset = SequenceDataset.registry[config.dataset._name_](
    #    **config.dataset
    #)
    local_rank = dist.get_rank()
    print(f'/home/ubuntu/josiah-fs1/caduceus/dataset_bulk_exp23_8gpu/gpu_{local_rank}')
    dataset = datasets.load_from_disk(f'/home/ubuntu/josiah-fs1/caduceus/dataset_bulk_exp23_8gpu/gpu_{local_rank}').with_format('torch').train_test_split(0.1)
    train_dl = DataLoader(
                dataset['train'],
                batch_size=2, #batch_size,
                shuffle=False,
                #sampler=sampler,
                collate_fn=collate_fn,
                #**kwargs,
                )
    val_dl = DataLoader(
                dataset['test'],
                batch_size=2, #batch_size,
                shuffle=False,
                #sampler=sampler,
                collate_fn = collate_fn,
                #**kwargs,
                )
    #config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

    # Initialise your wandb run, passing wandb parameters and any config information
    init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}

    accelerator.init_trackers(
        project_name="ESP",
        config=dict(config),
        init_kwargs=init_kwargs
        )

    #accelerator.print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
    #accelerator.print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
    #accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")

    num_training_steps = len(train_dl)

    # Set zero weight decay for some params
    #if 'optimizer_param_grouping' in model.hparams.train:
    #    add_optimizer_hooks(model, **model.hparams.train.optimizer_param_grouping)
    # Normal parameters
    #all_params = list(model.parameters())
    #params = [p for p in all_params if not hasattr(p, "_optim")]
    #optimizer = utils.instantiate(registry.optimizer, model.hparams.optimizer, params)
    #del model.hparams.optimizer._name_
    optimizer = AdamW(model.parameters(), lr=0.0001)
    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    #model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dl, eval_dl, lr_scheduler
    #     )
    model, optimizer = accelerator.prepare(model, optimizer)
    if accelerator.is_main_process:
        progress_bar = tqdm(range(num_training_steps), initial = 0 * len(train_dl))

    # Start model training and defining the training loop
    model.train()
    for epoch in range(0,1):
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            # Training
            d = {k:v.shape for k,v in batch.items()}
            print(f'{dist.get_rank()} - {d}')
            if dist.get_world_size() > 1:
                #loss = model.module._shared_step(batch, batch_idx, prefix="train")
                output = model(**batch, return_dict=True)
                loss, mlm_loss, value_loss = output
            else:
                #loss = model._shared_step(batch, batch_idx, prefix="train")
                raise Exception("need distributed")
            accelerator.backward(loss)
            optimizer.step()
            #lr_scheduler.step()
            if accelerator.is_main_process:
                progress_bar.update(1)
            accelerator.log({'loss': loss, 'mlm_loss':mlm_loss, 'value_loss':value_loss, 'grad_norm':get_grad_norm(model),'param_norm':get_param_norm(model)})
    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    accelerator.end_training()

if __name__ == "__main__":
    main()
