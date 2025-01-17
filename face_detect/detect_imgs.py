"""
This code is used to batch detect images in a folder.
"""

import os
import sys
import cv2
import numpy as  np
import torch

from face_detect.vision.ssd.config.fd_config import define_img_size

input_size = 320
test_device = 'cpu'
net_type = 'slim'
threshold = 0.6
candidate_size = 1500

define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from face_detect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from face_detect.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

label_path = "./face_detect/models/voc-model-labels.txt"
test_device = test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if net_type == 'slim':
    model_path = "./face_detect/models/pretrained/version-slim-320.pth"
    # model_path = "./face_detect/models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "./face_detect/models/pretrained/version-RFB-320.pth"
    # model_path = "./face_detect/models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

def get_face_boundingbox(orig_image):
    """
        Description:
            In input image, detect face

        Args:
            orig_image: input BGR image.
    """
    boxes, labels, probs = predictor.predict(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB), candidate_size / 2, threshold)

    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    height, width, _ = orig_image.shape
    valid_face = np.logical_and(
        np.logical_and(boxes[:,0] >= 0, boxes[:,1] >= 0), 
        np.logical_and(boxes[:,2] < width, boxes[:,3] < height)
    )

    boxes = boxes[valid_face] 
    probs = probs[valid_face]

    return boxes, probs
