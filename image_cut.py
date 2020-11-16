import json
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import copy
import os
import progressbar
import PIL
from PIL import Image
import numpy as np
import argparse

from image_to_feature import Image2Feat

parser = argparse.ArgumentParser()
parser.add_argument('--ID', default=-1, type=int, help='ID')
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES']=str(args.ID - 2)

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def IMG_CUT_X(bbox_list):
    arr = []
    for bbox in bbox_list:
        arr.append({'value': bbox[0],
                    'tag': '('})
        arr.append({'value': bbox[0] + bbox[2],
                    'tag': ')'})
    arr = sorted(arr, key=lambda x: x['value'])
    left = 0
    right = 0
    res = []
    for i, x in enumerate(arr):
        if x['tag'] == '(':
            left += 1
        else:
            right += 1
        if left and left == right:
            if i < len(arr) - 1:
                res.append(arr[i]['value'] + (arr[i+1]['value'] - arr[i]['value']) / 2)
    return res

def IMG_CUT_Y(bbox_list):
    arr = []
    for bbox in bbox_list:
        arr.append({'value': bbox[1],
                    'tag': '('})
        arr.append({'value': bbox[1] + bbox[3],
                    'tag': ')'})
    arr = sorted(arr, key=lambda x: x['value'])
    left = 0
    right = 0
    res = []
    for i, x in enumerate(arr):
        if x['tag'] == '(':
            left += 1
        else:
            right += 1
        if left and left == right:
            if i < len(arr) - 1:
                res.append(arr[i]['value'] + (arr[i+1]['value'] - arr[i]['value']) / 2)
    return res

def crop_X(path, p_list, out_dir):
    if len(p_list) == 0:
        return []
    img = Image.open(path)
    W = img.width
    H = img.height
    img_list = []
    p_list = [0] + p_list + [W]
    dir_list = []
    for i in range(len(p_list)-1):
        box = (p_list[i], 0, p_list[i+1], H)
        sub = str(path.split('/')[-1].split('.')[0])
        make_path(out_dir + sub)
        img.crop(box).save(out_dir + sub + '/' + (sub + '-X-%d' % i) + '.jpg')
        dir_list.append(out_dir + sub + '/' + (sub + '-X-%d' % i) + '.jpg')
        img.crop(box).save(out_dir + 'img-X' + ('-%d' % i) + '.jpg')
        print('Saved.')
    return dir_list

def crop_Y(path, p_list, out_dir):
    if len(p_list) == 0:
        return []
    img = Image.open(path)
    W = img.width
    H = img.height
    img_list = []
    p_list = [0] + p_list + [H]
    dir_list = []
    for i in range(len(p_list)-1):
        box = (0, p_list[i], W, p_list[i+1])
        sub = str(path.split('/')[-1].split('.')[0])
        make_path(out_dir + sub)
        img.crop(box).save(out_dir + sub + '/' + (sub + '-Y-%d' % i) + '.jpg')
        dir_list.append(out_dir + sub + '/' + (sub + '-Y-%d' % i) + '.jpg')
        img.crop(box).save('img-Y' + ('-%d' % i) + '.jpg')
        print('Saved.')
    return dir_list

def get_bboxs(img_id, json_data):
    bboxs = []
    c_ids = []
    for anno in json_data['annotations']:
        if anno['image_id'] == img_id:
            bboxs.append(anno['bbox'])
            c_ids.append(anno['category_id'])
    return bboxs, c_ids

# Load VG Classes
data_path = '/root/data/BUTD-config/data/genome/1600-400-20'
yaml_path = '/root/data/BUTD-config/configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml'
weights_path = '/root/collection/pretrained-models/BUTD/faster_rcnn_from_caffe_attr.pkl'

I2F = Image2Feat(
        data_path=data_path,
        yaml_path=yaml_path,
        weights_path=weights_path
    )

img_root = '/root/data/COCO/val2014/'
img_list = sorted(os.listdir(img_root))
json_path = '/root/data/COCO/annotations/instances_val2014.json'
out_dir = '/root/data/COCO/CUT/image/'

feature_dir = '/root/data/COCO/CUT/feature/BUTD/'
question_dir = '/root/data/COCO/CUT/question/'

with open(json_path, 'r') as f:
    json_data = json.load(f)

categories = dict()
for c in json_data['categories']:
    categories[int(c['id'])] = c['name']

length = 2000

cur_idx = 0
progress = progressbar.ProgressBar(maxval=length).start()
for idx, img_name in enumerate(img_list):
    if cur_idx >= length:
        print('From %d Images' % idx)
        break
    progress.update(cur_idx + 1)
    img_path = img_root + img_name
    img_id = int(img_name.split('.')[0].split('_')[-1])
    bbox_list, c_id_list = get_bboxs(img_id, json_data)
    c_name_list = [categories[c_id] for c_id in c_id_list]

    p_X = IMG_CUT_X(bbox_list)
    p_Y = IMG_CUT_Y(bbox_list)
    X_dir_list = crop_X(img_path, p_X, out_dir+'X/')
    Y_dir_list = crop_Y(img_path, p_Y, out_dir+'Y/')
    
    if len(X_dir_list) == 0 or len(Y_dir_list) == 0:
        continue
    cur_idx += 1
    img = cv2.imread(img_path)
    instances, BUTD_features, _ = I2F.doit(img)
    pred = instances.to('cpu')

    class_list = [cls.item() for cls in pred.pred_classes]
    attr_list = [attr.item() for attr in pred.attr_classes]
    objects = dict()

    for c_name in c_name_list:
        objects[c_name] = {'attr': set()}

    # filter classes not appear in COCO
    for i in range(min(len(class_list), I2F.NUM_OBJECTS)):
        cls = str(I2F.get_class_name(class_list[i]))
        attr = str(I2F.get_attr_name(attr_list[i]))
        if cls in c_name_list:
            if cls in objects.keys():
                objects[cls]['attr'].add(attr)
            else:
                objects[cls] = {'attr': set([attr])}
    for k in objects.keys():
        objects[k]['attr'] = list(objects[k]['attr'])

    Question = {'add': [], 'not': [], 'equ': []}
    Question = I2F.generate_Q(objects, Question, obj_th1=1, obj_th2=1)
    
    prefix = '.'.join(img_name.split('.')[:-1])
    suffix = '.json'
    with open(question_dir + prefix + suffix, 'w') as f:
        json.dump(Question, f)

    suffix = '.npz'

    X_features_list = []
    for X_dir in X_dir_list:
        X_img = cv2.imread(X_dir)
        _, X_BUTD_features, _ = I2F.doit(X_img)
        X_features_list.append(X_BUTD_features.to('cpu').numpy())

    Y_features_list = []
    for Y_dir in Y_dir_list:
        Y_img = cv2.imread(Y_dir)
        _, Y_BUTD_features_, _ = I2F.doit(Y_img)
        Y_features_list.append(Y_BUTD_features.to('cpu').numpy())
    np.savez_compressed(feature_dir + prefix + suffix, 
            origin=BUTD_features.to('cpu').numpy(),
            X=X_features_list,
            Y=Y_features_list
        )
progress.finish()