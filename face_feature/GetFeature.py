
import cv2
import numpy as np
import torch
from face_feature.irn50_pytorch import irn50_pytorch
from face_util.faceutil import align_vertical

import ctypes

model_feature = irn50_pytorch("./face_feature/irn50_pytorch.npy")
model_feature.eval()
feature_align_image = np.zeros([128, 128, 3], dtype=np.uint8)

def get_face_feature(image, landmark):
    landmark_vec = (ctypes.c_float * len(landmark))(*landmark)
    align_vertical(image, image.shape[1], image.shape[0], feature_align_image, 128, 128, 3, landmark_vec, 48, 64, 40)
    # cv2.imwrite("D:/align.png", feature_align_image)
    feature_align_image_proc = feature_align_image / 256
    feature_align_image_proc = torch.from_numpy(feature_align_image_proc.astype(np.float32))
    feature_align_image_proc = feature_align_image_proc.permute(2, 0, 1)
    feature_align_image_proc = feature_align_image_proc.unsqueeze(0)
    feature_out = model_feature(feature_align_image_proc)
    feature_out = torch.nn.functional.normalize(feature_out)[0, :]
    return feature_align_image, feature_out.data.numpy()
