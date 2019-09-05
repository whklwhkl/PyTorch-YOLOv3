import argparse


ap = argparse.ArgumentParser()
ap.add_argument('--train')  # train.txt
ap.add_argument('--val')    # valid.txt
ap.add_argument('--outputs')# data/custom
args = ap.parse_args()

from glob import glob

import shutil
import os.path as osp
from tqdm import tqdm


from tqdm import tqdm
import os


args.outputs = osp.expanduser(args.outputs)


def maybe_create_folder(folder_name):
    if osp.exists(folder_name)==False:
        os.mkdir(folder_name)


maybe_create_folder(osp.join(args.outputs, 'images'))
maybe_create_folder(osp.join(args.outputs, 'labels'))


for split, saveAs in zip(map(osp.expanduser, [args.val, args.train]),
                         ['valid.txt', 'train.txt']):
    useful = []
    for l in tqdm(open(split).readlines(), desc=osp.basename(split)):
        image = osp.basename(l.strip())
        label = osp.splitext(image)[0]+'.txt'
        label_path = l.strip().replace('images', 'labels').replace('.jpg', '.txt')
        if osp.exists(label_path)==False:
            continue
        p = []
        for line in open(label_path):
            if line.split()[0]=='0':    # '0' for person in COCO
                p.append(line)
        if len(p)==0:
            continue
        image_path = osp.join(args.outputs, 'images', image)
        useful.append(image_path)
        shutil.copy(l.strip(), image_path)
        with open(osp.join(args.outputs, 'labels', label), 'w') as fw:
            for line in p:
                fw.writelines(line)
    with open(osp.join(args.outputs, saveAs), 'w') as fw:
        for u in useful:
            print(u.strip(), file=fw)
