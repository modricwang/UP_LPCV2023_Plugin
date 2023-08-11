import cv2
import random
import numpy as np
from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY

@AUGMENTATION_REGISTRY.register('seg_cutout')
class SegCutout(Augmentation):
    def __init__(self, n_holes=1, length=0.1, ignore_index=255):
        self.n_holes = n_holes
        self.length = length
        self.ignore_index = ignore_index

    def augment(self, data):
        image = data['image']
        label = data['gt_semantic_seg']
        h, w, _ = image.shape

        mask = np.ones((h, w), np.float32)
        dy = int(h * self.length)
        dx = int(w * self.length)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - dy // 2, 0, h)
            y2 = np.clip(y + dy // 2, 0, h)
            x1 = np.clip(x - dx // 2, 0, w)
            x2 = np.clip(x + dx // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
        mask_image = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image = image * mask_image

        # Apply the same cutout to the ground truth
        gt = label.astype(float)
        gt_mask = np.logical_not(mask.astype(bool))
        gt[gt_mask] = self.ignore_index

        data['image'] = image
        data['gt_semantic_seg'] = gt
        return data

@AUGMENTATION_REGISTRY.register('seg_random_flip_vertical')
class SegRandomVerticalFlip(Augmentation):
    def augment(self, data):
        image = data['image']
        label = data['gt_semantic_seg']
        flip = np.random.choice(2) * 2 - 1
        data['image'] = image[::flip, :, :]
        data['gt_semantic_seg'] = label[::flip, :]
        return data

@AUGMENTATION_REGISTRY.register('seg_rand_rotate_lpcv')
class RandRotate(Augmentation):
    def __init__(self, angle=20., prob=1.):
        self.angle = angle
        self.prob = prob

    def augment(self, data):
        if random.random() <= self.prob:
            image = data['image']
            label = data['gt_semantic_seg']
            angle = random.random() * self.angle - 10
            h, w = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            data['image'] = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
            data['gt_semantic_seg'] = cv2.warpAffine(label, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        return data
