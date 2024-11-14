length=$((2**20))
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 train_accel_gpu.py   \
	dataset.shuffle=false \
	experiment=hg38/hg38  \
       	callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
	dataset.max_length=$length   \
	dataset.batch_size=1   \
	dataset.mlm=true   \
	dataset.mlm_probability=0.15   \
	dataset.rc_aug=false   \
	model=caduceus   \
	model.config.d_model=256 \
	model.config.ssm_cfg.headdim=16   \
	model.config.n_layer=16   \
	model.config.bidirectional=true   \
	dataset.context_parallel=true \
	model.config.ssm_cfg.context_parallel=true \
	model.config.bidirectional_strategy=add   \
	model.config.bidirectional_weight_tie=true   \
	model.config.rcps=false   \
	optimizer.lr="8e-3"   \
	train.global_batch_size=1   \
	trainer.max_steps=10000   \
	+trainer.val_check_interval=10000   \
	wandb=null
