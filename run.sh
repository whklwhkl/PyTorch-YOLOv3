python -m torch.distributed.launch \
	--nproc_per_node=4 \
	train.py \
		--model_def config/yolov3-custom.cfg \
		--data_config config/custom.data \
		--pretrained_weights weights/darknet53.conv.74 \
		--n_cpu 4 \
		--checkpoint_interval 5 \
		--evaluation_interval 5 \
		--batch_size 8 \
		--img_size 608 \
		--gradient_accumulations 1
