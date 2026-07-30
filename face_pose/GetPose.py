import numpy as np
from numpy.ctypeslib import as_ctypes_type
from face_util.faceutil import getPose68
import ctypes

def get_face_pose(box, landmark):
    fw = box[2] - box[0]
    fh = box[3] - box[1]
    sz = pow(fw * fh, 0.5)
    cx = (box[2] + box[0]) / 2
    cy = (box[3] + box[1]) / 2
    x1 = cx - sz / 2
    y1 = cy - sz / 2

    rx = ctypes.c_float(0)
    ry = ctypes.c_float(0)
    rz = ctypes.c_float(0)

    landmark_vec = (ctypes.c_float * len(landmark))(*landmark)
    getPose68(x1, y1, sz, landmark_vec, 68, ctypes.pointer(rx), ctypes.pointer(ry), ctypes.pointer(rz))
    return np.array([rx.value, ry.value, rz.value], dtype=np.float32)
