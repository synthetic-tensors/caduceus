length=$((2**19))
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train_accel_gpu.py   experiment=hg38/hg38   callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500   dataset.context_parallel=false dataset.max_length=$length   dataset.batch_size=1   dataset.mlm=true   dataset.mlm_probability=0.15   dataset.rc_aug=false   model=caduceus   model.config.d_model=128   model.config.n_layer=4   model.config.bidirectional=true   model.config.context_parallel=false model.config.bidirectional_strategy=add   model.config.bidirectional_weight_tie=true   model.config.rcps=false   optimizer.lr="8e-3"   train.global_batch_size=1   trainer.max_steps=10000   +trainer.val_check_interval=10000   wandb=null
