
import cv2
import numpy as np
import torch
from face_landmark.MobileFaceNet import MobileFaceNet

model_landmark = MobileFaceNet(input_size=64, embedding_size=136)
model_landmark.load_state_dict(torch.load("./face_landmark/vfl_1.02_578_6.734591484069824.pth.tar", map_location=torch.device('cpu'))['state_dict'])
model_landmark.eval()

def get_face_landmark(gray_img, bounding_box):
    image = gray_img
    box = bounding_box

    nHeight, nWidth = image.shape

    rLeftMargin = 0.05
    rTopMargin = 0.00
    rRightMargin = 0.05
    rBottomMargin = 0.10

    rW = box[2] - box[0]
    rH = box[3] - box[1]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    sz = pow(rW * rH, 0.5)
    rX = cx - sz / 2
    rY = cy - sz / 2
    rW = sz
    rH = sz
    
    #get image range to get face landmark from face rect
    iExFaceX = int(rX - rLeftMargin * rW)
    iExFaceY = int(rY - rTopMargin * rH)
    iExFaceW = int((1 + (rLeftMargin + rRightMargin)) * rW)
    iExFaceH = int((1 + (rTopMargin + rBottomMargin)) * rH)

    iExFaceX = np.clip(iExFaceX, 0, nWidth - 1)
    iExFaceY = np.clip(iExFaceY, 0, nHeight - 1)
    iExFaceW = np.clip(iExFaceX + iExFaceW, 0, nWidth - 1) - iExFaceX
    iExFaceH = np.clip(iExFaceY + iExFaceH, 0, nHeight - 1) - iExFaceY

    #crop face image in range to face landmark
    image = image[iExFaceY:iExFaceY+iExFaceH, iExFaceX:iExFaceX+iExFaceW]
    #normalize croped face image
    image = cv2.resize(image, (64, 64), cv2.INTER_LINEAR)
    # cv2.imwrite("D:/crop.png", image)
    image = image / 256
    image = torch.from_numpy(image.astype(np.float32))
    #convert mask_align_image from type [n,n] to [1,1,n,n]
    image = image.unsqueeze(0).unsqueeze(0)

    #get landmark fron croped face image
    landmark = model_landmark(image)
    #reshape face landmark and convert to image coordinates
    landmark = landmark.reshape(68, 2)
    landmark[:,0] = landmark[:,0] * iExFaceW + iExFaceX
    landmark[:,1] = landmark[:,1] * iExFaceH + iExFaceY
    
    landmark = landmark.reshape(-1)

    return landmark