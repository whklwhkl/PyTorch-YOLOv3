CUDA_VISIBLE_DEVICES=3 python train.py \
		--model_def config/yolov3-custom.cfg \
		--data_config config/custom.data \
		--pretrained_weights weights/yolov3.weights \
		--n_cpu 4 \
		--checkpoint_interval 5 \
		--evaluation_interval 5 \
		--batch_size 8 \
		--img_size 416 \
		--gradient_accumulations 2
