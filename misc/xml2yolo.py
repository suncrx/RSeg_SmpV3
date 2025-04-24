# Usage:
# python xml2yolo.py --img_path data/images --xml_path data/Annotations --out_path data/label

import os
import glob
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_all_classes(xml_path):
    xml_fns = glob.glob(os.path.join(xml_path, '*.xml'))
    class_names = []
    for xml_fn in xml_fns:
        tree = ET.parse(xml_fn)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls = obj.find('name').text
            class_names.append(cls)
    return sorted(list(set(class_names)))


def list_all_images(directory):
    imgs = []
    path = Path(directory)
    for file in path.rglob('*.jpg'):
        if file.is_file():
            imgs.append(file)
    return imgs


def convert_annotation(img_path, xml_path, class_names, out_path):
    output = []
    # make output dir
    os.makedirs(out_path, exist_ok=True)

    #im_fns = glob.glob(os.path.join(img_path, '*.jpg'))
    im_fns = list_all_images(img_path)

    for im_fn in tqdm(im_fns):
        if os.path.getsize(im_fn) == 0:
            continue
        xml_fn = os.path.join(xml_path, os.path.splitext(os.path.basename(im_fn))[0] + '.xml')
        if not os.path.exists(xml_fn):
            continue
        img = Image.open(im_fn)
        height, width = img.height, img.width
        tree = ET.parse(xml_fn)
        root = tree.getroot()
        anno = []
        xml_height = int(root.find('size').find('height').text)
        xml_width = int(root.find('size').find('width').text)
        if height != xml_height or width != xml_width:
            print((height, width), (xml_height, xml_width), im_fn)
            continue
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls_id = class_names.index(cls)
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)
            cx = (xmax + xmin) / 2.0 / width
            cy = (ymax + ymin) / 2.0 / height
            bw = (xmax - xmin) * 1.0 / width
            bh = (ymax - ymin) * 1.0 / height
            anno.append('{} {} {} {} {}'.format(cls_id, cx, cy, bw, bh))
        
        if len(anno) > 0:
            output.append(im_fn)
            #txt_out_path = 'data/labels/'            
            txt_filename = os.path.basename(im_fn).replace('.jpg', '.txt')  
            txt_file_path = os.path.join(out_path, txt_filename)              
            with open(txt_file_path, 'w') as f:
                f.write('\n'.join(anno))

'''
    random.shuffle(output)
    train_num = int(len(output) * 0.9)
    with open(os.path.join(out_path, 'train.txt'), 'w') as f:
        f.write('\n'.join(output[:train_num]))
    with open(os.path.join(out_path, 'val.txt'), 'w') as f:
        f.write('\n'.join(output[train_num:]))
'''


def parse_args():
    parser = argparse.ArgumentParser('generate annotation')
    parser.add_argument('--img_path', type=str, help='input image directory')
    parser.add_argument('--xml_path', type=str, help='input xml directory')
    parser.add_argument('--out_path', type=str, help='output directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    args.img_path = 'D:\\GeoData\\DLData\\SurfaceDefects\\NEU-DET\\val\\images'
    args.xml_path = 'D:\\GeoData\\DLData\\SurfaceDefects\\NEU-DET\\val\\annotations'
    args.out_path = 'D:\\GeoData\\DLData\\SurfaceDefects\\NEU-DET\\val\\labels'

    class_names = get_all_classes(args.xml_path)
    print(class_names)
    convert_annotation(args.img_path, args.xml_path, class_names, args.out_path)

