import logging
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
#from utils.training import count_parameters #, move_to
import hydra

from accel_model import SequenceModule
from src.utils import registry
import src.utils as utils
from src.utils.optim_groups import add_optimizer_hooks
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(args.fsdp, num_gpus // args.fsdp),
                                                    mesh_dim_names=('dp', 'cp'))
    cp_mesh, dp_mesh = device_mesh['cp'], device_mesh['dp']
    print(dist.get_rank(), cp_mesh, dist.get_process_group_ranks(cp_mesh.get_group()))
    print(dist.get_rank(), dp_mesh, dist.get_process_group_ranks(dp_mesh.get_group()))

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    if config.train.seed is not None:
        torch.manual_seed(config.train.seed)
    model = SequenceModule(config)
    print(model)
    train_dl, eval_dl = model.train_dataloader(), model.val_dataloader()

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

    # Start model training and defining the training loop
    model.train()
    world_size = dist.get_worldsize()
    print('Rank/World Size',dist.get_rank(),'/',world_size)
    for epoch in range(0,1):
        for batch_idx, batch in tqdm(enumerate(train_dl)):
            # Training
            print(f'forward on {dist.get_rank()}')
            loss = model.module._shared_step(batch, batch_idx, prefix="train")
            #batch = move_to(batch, device)
            print(f'backward on {dist.get_rank()}')
            loss.backward()
            #print(f'optimize on {dist.get_rank()}')
            #optimizer.step()
            #lr_scheduler.step()

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

if __name__ == "__main__":
    main()
