img_path_file = 'data/coco/trainvalno5k.txt'
from PIL import Image
from tqdm import tqdm


for l in tqdm(open(img_path_file).readlines()):
    try:
        im = Image.open(l.strip()).convert('RGB')
    except OSError:
        print(l)
