import os
import sys
import time
import datetime

from flask import Flask, request, jsonify
app = Flask('yolov3')

from io import BytesIO
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose, Resize

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
parser.add_argument("--port", default=6666)
opt = parser.parse_args()
print(opt)


from models import *
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))
model.eval()  # Set in evaluation mode

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
classes = load_classes(opt.class_path)  # Extracts class labels from file

transforms = Compose([Resize([opt.img_size]*2), ToTensor()])
@app.route('/det', methods=['POST'])
def det():
    f = request.files['img']
    imb = BytesIO(f.read())
    img = Image.open(imb)
    input_imgs = transforms(img).cuda()

    with torch.no_grad():
        detections = model(input_imgs[None])
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
    ret = {}
    if detections is not None:
        detections = rescale_boxes(detections, opt.img_size, img.size[::-1])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            ret.setdefault('dets', []).append(
                {'label':classes[int(cls_pred)],
                 'conf':cls_conf.item(),
                 'x1y1x2y2':[x1.item(), y1.item(), x2.item(), y2.item()]
                 })
    return jsonify(ret)

app.run(host="0.0.0.0", debug=True, port=opt.port)
