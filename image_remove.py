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

from image_to_feature import Image2Feat

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_bboxs(img_id, json_data):
    bboxs = []
    c_ids = []
    for anno in json_data['annotations']:
        if anno['image_id'] == img_id:
            bboxs.append(anno['bbox'])
            c_ids.append(anno['category_id'])
    return bboxs, c_ids

def isEdgeOverlap(s1, e1, s2, e2):
    return not (e1 <= s2 or e2 <= s1)

def isBoxOverlap(box1, box2):
    # box: top-left (x, y, w, h)
    x_overlap = isEdgeOverlap(box1[0], box1[0]+box1[2], box2[0], box2[0]+box2[2])
    y_overlap = isEdgeOverlap(box1[1], box1[1]+box1[3], box2[1], box2[1]+box2[3])
    return x_overlap and y_overlap 

def noOverlap(bbox_list, c_name_list):
    selected_bbox_list = []
    selected_c_name_list = []
    list_len = len(bbox_list)
    for i in range(list_len):
        j = 0
        while (j < list_len):
            if j != i and isBoxOverlap(bbox_list[i], bbox_list[j]):
                break
            j += 1
        if j >= list_len:
            selected_bbox_list.append(bbox_list[i])
            selected_c_name_list.append(c_name_list[i])
    return selected_bbox_list, selected_c_name_list


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
out_dir = '/root/data/COCO/REMOVE/image/'
feature_dir = '/root/data/COCO/REMOVE/feature/'
question_dir = '/root/data/COCO/REMOVE/question/'

with open(json_path, 'r') as f:
    json_data = json.load(f)

categories = dict()
for c in json_data['categories']:
    categories[int(c['id'])] = c['name']

length = 2000
progress = progressbar.ProgressBar(maxval=length).start()
cur_idx = 0
for idx, img_name in enumerate(img_list):
    if cur_idx >= length:
        print('From %d Images' % idx)
        break
    progress.update(cur_idx + 1)
    img_path = img_root + img_name
    img_id = int(img_name.split('.')[0].split('_')[-1])
    bbox_list, c_id_list = get_bboxs(img_id, json_data)
    if len(bbox_list) < 2:
        continue
    c_name_list = [categories[c_id] for c_id in c_id_list]

    selected_bbox_list, selected_c_name_list = noOverlap(bbox_list, c_name_list)
    if len(selected_bbox_list) < 3 or len(set(selected_c_name_list)) < 2:
        continue
    #print(set(selected_c_name_list))
    cur_idx += 1
    img = cv2.imread(img_path)
    (H, W, _) = img.shape
    instances, BUTD_features, _ = I2F.doit(img)
    pred = instances.to('cpu')

    class_list = [cls.item() for cls in pred.pred_classes]
    attr_list = [attr.item() for attr in pred.attr_classes]
    objects = dict()

    for c_name in c_name_list:
        objects[c_name] = {'attr': set()}

    # select classes in COCO
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

    final_Q_list = []
    masked_feat_list = []

    for k, bbox in enumerate(selected_bbox_list):
        c_name = selected_c_name_list[k]
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = 1
        tmp_img = copy.deepcopy(img)
        for i in range(3):
            tmp_img[:, :, i] = tmp_img[:, :, i] * (1 - mask) + mask * 255
        sub = str(img_name.split('.')[0])
        make_path(out_dir + 'masked/' + sub)

        key = c_name
        value = objects.pop(key, None)
        
        # generate questions
        Question1 = {'add': [], 'not': [], 'equ': []}
        if len(objects.keys()) == 1:
            obj = list(objects.keys())[0]
            Question1['add'].append(
                {'Q': [I2F.template1 + obj + I2F.end],
                'C': False}
                )
        Question1 = I2F.generate_Q(objects, Question1, obj_th1=2, obj_th2=2)
        
        
        Question2 = {'add': [], 'not': [], 'equ': []}
        Question2['add'].append(
            {'Q': [I2F.template1 + key + I2F.end],
            'C': False}
            )
        for obj in objects.keys():
            obj1 = key
            obj2 = obj
            if np.random.uniform() > 0.5:
                Question2 = I2F.add_Q1_Q3(Question2, obj1, obj2, False)
            else:
                Question2 = I2F.add_Q1_Q3(Question2, obj2, obj1, False)

        final_Q = {
            'same': Question1,
            'diff': Question2
        }
        final_Q_list.append(final_Q)

        objects[key] = value

        _, masked_BUTD_features, _ = I2F.doit(tmp_img)
        masked_feat_list.append(masked_BUTD_features.to('cpu').numpy())
        cv2.imwrite(out_dir + 'masked/' + sub + '/' + (sub+'-masked-%d' % k) + '.jpg', tmp_img)
        m = np.stack([mask * 255] * 3, -1)
        make_path(out_dir + 'mask/' + sub)
        cv2.imwrite(out_dir + 'mask/' + sub + '/' + (sub+'-mask-%d' % k) + '.jpg', m)
    #print(len(masked_feat_list))
    #print(len(final_Q_list))
    prefix = '.'.join(img_name.split('.')[:-1])
    suffix = '.npz'
    np.savez_compressed(feature_dir + 'masked/BUTD/' + prefix + suffix,
        origin=BUTD_features.to('cpu').numpy(),
        masked=masked_feat_list
        )
    suffix = '.json'
    with open(question_dir + prefix + suffix, 'w') as f:
        json.dump(final_Q_list, f)
progress.finish()