<div align="center">
<img alt="" src="https://github.com/Faceplugin-ltd/FaceRecognition-Javascript/assets/160750757/657130a9-50f2-486d-b6d5-b78bcec5e6e2.png" width=200/>
</div>

# Face Recognition SDK for Windows and Linux - Fully On Premise
## Overview
The world's 1st **Completely Free and Open Source** `Face Recognition SDK` from [Faceplugin](https://faceplugin.com/) for developers to integrate face recognition capabilities into applications. Supports real-time, high-accuracy face recognition with deep learning models.
<br>This is `on-premise face recognition SDK` which means everything is processed in your phone and **NO** data leaves the device
<br>You can use this SDK on Windows and Linux
<br><br>**Please contact us if you need the SDK with higher accuracy.**
<br></br>

## Key Features
- **Real-Time Face Recognition**: Can detect and recognize faces from live video streams. Currently only supports face recognition from an image.
- **High Accuracy**: Built with deep learning models trained on large datasets.
- **Cross-Platform**: Compatible with Windows and Linux.
- **Flexible Integration**: Easy-to-use APIs for seamless integration into any project.
- **Scalable**: Works on local devices, cloud, or embedded systems.
- **Python SDK**: Comprehensive support for Python with extensive documentation and examples.

## Applications
This **Face Recognition SDK** is ideal for a wide range of applications, including:
- **Time Attendance Systems**: Monitor arrivals and depatures using face recognition.
- **Security Systems**: Access control and surveillance.
- **User Authentication**: Biometric login and multi-factor authentication.
- **Smart Devices**: Integration into IoT devices for smart home or office applications.
- **Augmented Reality**: Enhance AR applications with real-time facial recognition.
- **Retail**: Personalized marketing and customer analytics.

## Installation
Please download anaconda on your computer and install it.
We used Windows machine without GPU for testing.

### create anaconda environment 
conda create -n facesdk python=3.9

### activate env
conda activate facesdk

### install dependencies
pip install -r requirements.txt

### compare 1.jpg and 2.png in the test directory.
python run.py

## APIs and Parameters

**GetImageInfo(image, faceMaxCount):** returns face bounding boxes, landmarks and feature embeddings<br>
**get_similarity(feat1, feat2):** returns similarity between two feature embeddings. 0 to 100<br>
**Threshold**: value to determine if two embeddings belong to same person, default = 75


## List of our Products

* **[FaceRecognition-LivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceRecognition-Android)**
* **[FaceRecognition-LivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceRecognition-iOS)**
* **[FaceRecognition-LivenessDetection-Javascript](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Javascript)**
* **[FaceLivenessDetection-Android](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Android)**
* **[FaceLivenessDetection-iOS](https://github.com/Faceplugin-ltd/FaceLivenessDetection-iOS)**
* **[FaceLivenessDetection-Linux](https://github.com/Faceplugin-ltd/FaceLivenessDetection-Linux)**
* **[FaceRecognition-LivenessDetection-React](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-React)**
* **[FaceRecognition-LivenessDetection-Vue](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Vue)**
* **[Face Recognition SDK](https://github.com/Faceplugin-ltd/Face-Recognition-SDK)**
* **[Liveness Detection SDK](https://github.com/Faceplugin-ltd/Face-Liveness-Detection-SDK)**
* **[ID Card Recognition](https://github.com/Faceplugin-ltd/ID-Card-Recognition)**

## Contact
<div align="left">
<a target="_blank" href="mailto:info@faceplugin.com"><img src="https://img.shields.io/badge/email-info@faceplugin.com-blue.svg?logo=gmail " alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://t.me/faceplugin"><img src="https://img.shields.io/badge/telegram-@faceplugin-blue.svg?logo=telegram " alt="faceplugin.com"></a>&emsp;
<a target="_blank" href="https://wa.me/+14422295661"><img src="https://img.shields.io/badge/whatsapp-faceplugin-blue.svg?logo=whatsapp " alt="faceplugin.com"></a>
</div>
