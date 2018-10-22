import json
import os

import cv2
import numpy as np
import torch
from torch.utils import data

BDD_BOX_CLASSES = [
    'rider', 'train', 'person', 'motor', 'traffic light', 'bus', 'truck',
    'traffic sign', 'car', 'bike'
]

BDD_SEG_CLASSES = ['lane', 'drivable area']


def get_label_map(labels):
    label_map = {}
    for label in labels:
        label_map[label] = len(label_map)
    return label_map


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


class BDDAnnotationTransform(object):

    def __init__(self):
        self.label_map = get_label_map(BDD_BOX_CLASSES)

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): BDD target json annotation as a python dict
            height (int): height
            width (int): width
        Returns
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []

        labels = target['labels']
        for label in labels:
            if 'box2d' in label:
                bbox = [
                    label['box2d']['x1'], label['box2d']['y1'],
                    label['box2d']['x2'], label['box2d']['y2']
                ]
                label_index = self.label_map[label['category']]
                final_box = list(np.array(bbox) / scale)
                final_box.append(label_index)
                res.append(final_box)  # [x1, y1, x2, y2, label_idx]

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class BDDDetection(data.Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=BDDAnnotationTransform(),
                 dataset_name='BDD'):
        self.image_dir = os.path.join(root, 'images/100k/val')
        self.label_file = os.path.join(root, 'labels/bdd100k_labels_images_val.json')
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self.labels = load_json(self.label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        target = self.labels[index]
        img = cv2.imread(os.path.join(self.image_dir, target['name']))

        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def __len__(self):
        return len(self.labels)
