MODEL_DEF=config/yolov3.cfg
WEIGHTS_PATH=weights/yolov3.weights


det:
	python detect.py \
		--weights_path ${WEIGHTS_PATH} \
		--model_def ${MODEL_DEF}

serving:
	python server.py \
		--weights_path ${WEIGHTS_PATH} \
		--model_def ${MODEL_DEF}
