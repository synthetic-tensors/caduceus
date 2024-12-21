import logging
from time import gmtime, strftime
from tqdm.auto import tqdm
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.regression import PearsonCorrCoef
#from utils.training import count_parameters #, move_to
import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
import os
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


def clip_min_max_norm(example, upper_lim=10, lower_lim=-5):
    data = example["input_vals"]
    data[data < lower_lim] = lower_lim
    data[data > upper_lim] = upper_lim
    example['input_vals'] = (data - lower_lim)/(upper_lim-lower_lim)
    return example

def collate_fn(batch, mlm_probability=0.15, sep_token=1):
    seq_ids, seq_targets, expr_values, expr_targets, ctrls = [], [], [], [], []
    for sample in batch:
        input_ids, input_vals, ctrl = sample['input_ids'], sample['input_vals'], sample['ctrl']
        seq_id = input_ids #torch.cat([sgrna, input_ids, sgrna.flip(0)])
        expr_value, expr_target = mlm_esp_getitem(
            input_vals,
            mlm_probability=1.0, #Makes them all mask tokens
        )
        ctrls.append(ctrl)
        seq_ids.append(seq_id)
        expr_values.append(expr_value)
        expr_targets.append(expr_target)
    return {'input_ids': torch.stack(seq_ids), \
            'values': torch.cat(expr_values), \
            'labels': None, \
            'ctrl':torch.cat(ctrls), \
            'value_labels': torch.cat(expr_targets)}
    # corresponding model inputs are (input_ids, values, labels, value_labels)

def collate_fn_sg(batch, mlm_probability=0.15, max_len=4096, sep_token=1, pad_token=4):
    seq_ids, seq_targets, expr_values, expr_targets = [], [], [], []
    #max_seq_len = max([len(x['input_ids']) for x in batch])
    ctrls=[]
    for sample in batch:
        input_ids, input_vals, ctrl = sample['input_ids'], sample['input_vals'], sample['ctrl']
        input_ids = input_ids[-max_len:]
        pad = nn.ConstantPad1d([0,max_len - len(input_ids)],pad_token)
        
        seq_id = pad(input_ids)
        #expr_value, expr_target = mlm_esp_getitem(
        #    input_vals,
        #    mlm_probability=1.0, #Makes them all mask tokens
        #)
        #if not expr_value or not 
        expr_value, expr_target = torch.Tensor([-100]), input_vals.unsqueeze(0)
        ctrl = ctrl.unsqueeze(0)
        #print(expr_value, expr_target)
        seq_ids.append(seq_id)
        ctrls.append(ctrl)
        expr_values.append(expr_value)
        expr_targets.append(expr_target)
    return {'input_ids': torch.stack(seq_ids), \
            'values': torch.cat(expr_values), \
            'labels': None, \
            'ctrl': torch.cat(ctrls), \
            'value_labels': torch.cat(expr_targets)}
    # corresponding model inputs are (input_ids, values, labels, value_labels)

@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    #from accelerate import DistributedDataParallelKwargs
    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #True)
    #accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device
    pcc = PearsonCorrCoef().to(device)
    pcc_ctrl = PearsonCorrCoef().to(device)
    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    if config.train.seed is not None:
        set_seed(42) #config.train.seed)
    model_config = ESPConfig(**config.model.config)
    model = ESPForMaskedLM(model_config)
    print("number of parameters in model:",sum(p.numel() for p in model.parameters())/10**6," Million")
    freeze_layers = config.model.config.freeze_n_layers
    if freeze_layers > 0:
        print(f"Freezing {freeze_layers} layers")
        modules_to_freeze = [model.ESP.backbone.embeddings.word_embeddings,
                             model.ESP.backbone.layers[:freeze_layers]]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    print(model)
    #print(model_config)

    # Dataset arguments
    #print(SequenceDataset.registry.keys())
    #self.dataset = SequenceDataset.registry[config.dataset._name_](
    #    **config.dataset
    #)
    local_rank = dist.get_rank()
    if config.single_gene:
        dataset = datasets.load_from_disk(config.dataset.path).with_format('torch')
    else:
        dataset = datasets.load_from_disk(os.path.join(config.dataset.path,f'gpu_{local_rank}')).with_format('torch')
    dataset = dataset.map(partial(clip_min_max_norm, lower_lim=config.dataset.lower_lim, upper_lim=config.dataset.upper_lim),num_proc=8)
    #print(dataset[0]['input_vals'])
    #dataset = dataset.train_test_split(0.1)
    if config.single_gene:
        _collate_fn = collate_fn_sg
    else:
        _collate_fn = collate_fn
    train_dl = DataLoader(
                dataset['train'],
                batch_size=config.dataset.batch_size,
                shuffle=False,
                #sampler=sampler,
                collate_fn=_collate_fn, #collate_fn,
                #**kwargs,
                num_workers=8,
                prefetch_factor=8,
                )
    val_dl = DataLoader(
                dataset['test'],
                batch_size=2*config.dataset.batch_size,
                shuffle=False,
                #sampler=sampler,
                collate_fn = _collate_fn,
                num_workers=8,
                prefetch_factor=8,
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

    num_training_steps = len(train_dl) * config.trainer.max_epochs

    # Set zero weight decay for some params
    #if 'optimizer_param_grouping' in model.hparams.train:
    #    add_optimizer_hooks(model, **model.hparams.train.optimizer_param_grouping)
    # Normal parameters
    #all_params = list(model.parameters())
    #params = [p for p in all_params if not hasattr(p, "_optim")]
    #optimizer = utils.instantiate(registry.optimizer, model.hparams.optimizer, params)
    #del model.hparams.optimizer._name_
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.0)
    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    #model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dl, eval_dl, lr_scheduler
    #     )

    if not config.single_gene:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    if accelerator.is_main_process:
        progress_bar = tqdm(range(num_training_steps), initial = 0 * len(train_dl))

    # Start model training and defining the training loop
    if config.trainer.restart:
        logger.info(f"Loading saved state from {config.trainer.restart}")
        accelerator.load_state(config.trainer.restart)
    accumulation_steps = config.train.global_batch_size
    save_preds = config.trainer.save_preds
    for epoch in range(0, config.trainer.max_epochs):
        model.train()
        acc_loss, acc_mlm_loss, acc_value_loss = 0.0, 0.0, 0.0
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            # Trainingi
            if dist.get_world_size() > 1:
                output = model(**batch, return_dict=True)
                loss, mlm_loss, value_loss, _, _ = output
            else:
                # loss = model._shared_step(batch, batch_idx, prefix="train")
                raise Exception("need distributed")
            # Backward pass
            accelerator.backward(loss / accumulation_steps)
            acc_loss += loss
            acc_mlm_loss += mlm_loss
            acc_value_loss += value_loss
            if (batch_idx + 1) % accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters
                optimizer.step()
                # lr_scheduler.step()
                if accelerator.is_main_process:
                    if not config.single_gene:
                        progress_bar.update(accumulation_steps)
                    else:
                        progress_bar.update(8)
                    accelerator.log(
                        {'loss': acc_loss / accumulation_steps, 'mlm_loss': acc_mlm_loss / accumulation_steps,
                         'value_loss': acc_value_loss / accumulation_steps, 'grad_norm': get_grad_norm(model),
                         'param_norm': get_param_norm(model)})
                optimizer.zero_grad()
                acc_loss, acc_mlm_loss, acc_value_loss = 0.0, 0.0, 0.0
        accelerator.save_state(os.path.join('ft_model',str(epoch)))
        model.eval()
        with torch.no_grad():
            total_val_loss=[]
            all_logits, all_labels, ctrls = [], [], []
            for batch_idx, batch in tqdm(enumerate(val_dl)):
                loss, mlm_loss, value_loss, value_logits, value_labels = model(**batch, return_dict=True)
                total_val_loss.append(loss)
                if save_preds:
                    all_logits.append(value_logits.cpu())
                    all_labels.append(value_labels.cpu())
                    ctrls.append(batch['ctrl'].cpu())
                accelerator.log({'val_loss': loss})
                pcc_ctrl.update(batch['ctrl'].to(device)/4.0-F.sigmoid(value_logits), batch['ctrl'].to(device)/4.0-value_labels)
                pcc.update(value_logits,value_labels)  
            torch.save(all_logits,os.path.join(f'val_preds_{str(epoch)}_gpu{dist.get_rank()}_logits.pt'))
            torch.save(all_labels,os.path.join(f'val_labels_{str(epoch)}_gpu{dist.get_rank()}_labels.pt'))
            torch.save(ctrls, os.path.join(f'val_ctrls_{str(epoch)}_gpu{dist.get_rank()}_ctrls.pt'))
            accelerator.log({'val_epoch_loss': sum(total_val_loss)/len(val_dl), 'val_epoch_pcc': pcc.compute().cpu(), 'val_epoch_pcc_ctrl': pcc_ctrl.compute().cpu()})
            pcc.reset()
            pcc_ctrl.reset()


    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    accelerator.end_training()

if __name__ == "__main__":
    main()
