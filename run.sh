python train.py \
	--model_def config/yolov3-custom.cfg \
	--data_config config/custom.data \
	--pretrained_weights weights/darknet53.conv.74 \
	--n_cpu 2 \
	--checkpoint_interval 5 \
	--evaluation_interval 5 \
	--batch_size 8
