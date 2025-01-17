
import argparse
import cv2
import torch
import numpy as np
import ctypes
import os.path
import time

from face_detect.detect_imgs import get_face_boundingbox
from face_landmark.GetLandmark import get_face_landmark
from face_feature.GetFeature import get_face_feature
from face_pose.GetPose import get_face_pose

def GetImageInfo(image, faceMaxCount):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ### Detection
    start_time = time.time() * 1000
    boxes, scores = get_face_boundingbox(image)
    boxes = boxes[:faceMaxCount]
    scores = scores[:faceMaxCount]
    count = len(boxes)
    bboxes = []
    bscores = []
    for idx in range(count):
        bboxes.append(boxes[idx].data.numpy())
        bscores.append(scores[idx].data.numpy())

    ### Landmark
    start_time = time.time() * 1000
    landmarks = [] ### np.zeros((count, 136), dtype=np.float32)
    for idx in range(count):
        landmarks.append(get_face_landmark(gray_image, boxes[idx]).data.numpy())

    ### Pose
    poses = []
    for idx in range(count):
        poses.append(get_face_pose(boxes[idx], landmarks[idx]))

    ### Feature
    start_time = time.time() * 1000
    features = []
    alignimgs = []
    for idx in range(count):
        alignimg, feature = get_face_feature(image, landmarks[idx])
        features.append(feature)
        alignimgs.append(alignimg)
    # print("Feature extraction time = %s ms" % (time.time() * 1000 - start_time))
    
    return count, bboxes, bscores, landmarks, alignimgs, features
    
def get_similarity(feat1, feat2):
    return (np.sum(feat1 * feat2) + 1) * 50

if __name__ == '__main__':
    threshold = 75
    test_directory = 'test'

    efn = os.getcwd() + "/test/1.jpg"
    img = cv2.imread(efn, cv2.IMREAD_COLOR)
    count, boxes, scores, landmarks, alignimgs, features1 = GetImageInfo(img, 5)
    
    vfn = os.getcwd() + "/test/2.png"
    img = cv2.imread(vfn, cv2.IMREAD_COLOR)
    count, boxes, scores, landmarks, alignimgs, features2 = GetImageInfo(img, 5)

    score = get_similarity(features1[0], features2[0])
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('score = ', score)
    if score > threshold:
        print('same person')
    else:
        print('different person')
