<div align="center">
  <img src="https://github.com/Faceplugin-ltd/FaceRecognition-Javascript/assets/160750757/657130a9-50f2-486d-b6d5-b78bcec5e6e2.png" alt="Face Recognition SDK Logo" width="200"/>
  
  # Open Source Face Recognition SDK
  
  **The world's first completely free and open-source face recognition SDK for Windows and Linux from [Faceplugin](https://faceplugin.com/)**
  
  [![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-blue.svg)](https://github.com/Faceplugin-ltd/Open-Source-Face-Recognition-SDK)
  [![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/license-Open%20Source-green.svg)](https://github.com/Faceplugin-ltd/Open-Source-Face-Recognition-SDK)
  [![Privacy](https://img.shields.io/badge/privacy-On--Premise%20Only-brightgreen.svg)](https://faceplugin.com/)
</div>

---

## 🚀 Overview

The **Open Source Face Recognition SDK** by [Faceplugin](https://faceplugin.com/) is a powerful, privacy-focused solution for integrating face recognition capabilities into your applications. Built with deep learning models, this SDK provides high-accuracy face detection and recognition while ensuring complete data privacy through on-premise processing.

### ✨ Key Features

- 🔒 **100% On-Premise**: All processing happens locally - no data leaves your device
- 🎯 **High Accuracy**: Powered by state-of-the-art deep learning models
- ⚡ **Real-Time Processing**: Fast face detection and recognition capabilities
- 🔧 **Easy Integration**: Simple Python APIs for seamless development
- 🌐 **Cross-Platform**: Compatible with Windows and Linux systems
- 📱 **GPU Optional**: Works efficiently on CPU-only systems
- 🆓 **Completely Free**: Open source with no licensing fees

### 🎯 Current Capabilities

- Face detection and bounding box extraction
- Facial landmark detection
- Feature embedding generation
- Face similarity comparison
- Support for multiple image formats (JPG, PNG, etc.)

---

## 🛠️ Installation

### Prerequisites

- **Python 3.9 or higher**
- **Anaconda** (recommended for dependency management)
- **Windows or Linux** operating system

### Setup Instructions

1. **Install Anaconda** (if not already installed)
   ```bash
   # Download from: https://www.anaconda.com/products/distribution
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n facesdk python=3.9
   conda activate facesdk
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the installation**
   ```bash
   python run.py
   ```

---

## 📖 Quick Start

### Basic Usage

```python
from face_recognition_sdk import FaceRecognition

# Initialize the SDK
face_sdk = FaceRecognition()

# Process an image
image_path = "path/to/your/image.jpg"
face_info = face_sdk.GetImageInfo(image_path, faceMaxCount=10)

# Compare two faces
similarity = face_sdk.get_similarity(feature1, feature2)
```

### Example: Face Comparison

```python
# Compare two images
image1 = "test/1.jpg"
image2 = "test/2.png"

# Get face information from both images
faces1 = face_sdk.GetImageInfo(image1, faceMaxCount=1)
faces2 = face_sdk.GetImageInfo(image2, faceMaxCount=1)

if faces1 and faces2:
    # Compare the first face from each image
    similarity = face_sdk.get_similarity(faces1[0]['embedding'], faces2[0]['embedding'])
    print(f"Similarity: {similarity}%")
    
    # Check if it's the same person (threshold = 75)
    is_same_person = similarity >= 75
    print(f"Same person: {is_same_person}")
```

---

## 🔧 API Reference

### Core Functions

#### `GetImageInfo(image_path, faceMaxCount)`
Extracts face information from an image.

**Parameters:**
- `image_path` (str): Path to the input image
- `faceMaxCount` (int): Maximum number of faces to detect

**Returns:**
- List of dictionaries containing:
  - `bbox`: Face bounding box coordinates
  - `landmarks`: Facial landmark points
  - `embedding`: Feature embedding vector

#### `get_similarity(feature1, feature2)`
Compares two face feature embeddings.

**Parameters:**
- `feature1` (array): First face embedding
- `feature2` (array): Second face embedding

**Returns:**
- Similarity score (0-100), where higher values indicate greater similarity

### Configuration

- **Default Threshold**: 75 (for determining if two faces belong to the same person)
- **Supported Formats**: JPG, PNG, BMP, TIFF
- **Face Detection**: Automatic detection of multiple faces per image

---

## 🎯 Use Cases

This SDK is ideal for various applications:

### 🔐 Security & Authentication
- **Access Control Systems**: Secure entry points with face recognition
- **User Authentication**: Biometric login for applications
- **Surveillance**: Real-time monitoring and alerting

### 👥 Business Applications
- **Time & Attendance**: Automated employee check-in/check-out
- **Customer Analytics**: Retail customer tracking and analytics
- **Smart Offices**: Automated visitor management

### 📱 Mobile & IoT
- **Smart Devices**: Integration with IoT devices
- **Mobile Apps**: Face recognition in mobile applications
- **Augmented Reality**: AR applications with facial recognition

---

## 🏢 Enterprise Solutions

For higher accuracy requirements and enterprise features, contact us for our commercial SDK offerings:

- **Enhanced Accuracy Models**: Superior recognition performance
- **Liveness Detection**: Anti-spoofing capabilities
- **Multi-Platform Support**: Android, iOS, Web, and more
- **Technical Support**: Professional assistance and documentation
---

## 📚 Related Products

Explore our complete suite of face recognition and biometric solutions:

### Mobile SDKs
- [Android (Java/Kotlin)](https://github.com/Faceplugin-ltd/FaceRecognition-Android)
- [iOS (Objective-C/Swift)](https://github.com/Faceplugin-ltd/FaceRecognition-iOS)
- [React Native](https://github.com/Faceplugin-ltd/FaceRecognition-React-Native)
- [Flutter](https://github.com/Faceplugin-ltd/FaceRecognition-Flutter)

### Web & Desktop
- [JavaScript](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Javascript)
- [React](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-React)
- [Vue.js](https://github.com/Faceplugin-ltd/FaceRecognition-LivenessDetection-Vue)
- [.NET MAUI](https://github.com/Faceplugin-ltd/FaceRecognition-.Net)
- [.NET WPF](https://github.com/Faceplugin-ltd/FaceRecognition-WPF-.Net)

### Specialized Solutions
- [Liveness Detection](https://github.com/Faceplugin-ltd/Face-Liveness-Detection-SDK)
- [Palm Recognition](https://github.com/Faceplugin-ltd/Palm-Recognition)
- [ID Card Recognition](https://github.com/Faceplugin-ltd/ID-Card-Recognition)
- [Document Liveness Detection](https://github.com/Faceplugin-ltd/ID-Document-Liveness-Detection)

---

## 🤝 Support & Contact
While there are many ways to support this project, starring ⭐️ this GitHub repository is one of the simplest and most impactful. It increases discoverability and helps the project reach a wider audience. Thank you for your support 🙏
<div align="center">
  <a href="mailto:info@faceplugin.com">
    <img src="https://img.shields.io/badge/Email-info@faceplugin.com-blue.svg?logo=gmail" alt="Email"/>
  </a>
  <a href="https://t.me/faceplugin">
    <img src="https://img.shields.io/badge/Telegram-@faceplugin-blue.svg?logo=telegram" alt="Telegram"/>
  </a>
  <a href="https://wa.me/+14692784822">
    <img src="https://img.shields.io/badge/WhatsApp-faceplugin-green.svg?logo=whatsapp" alt="WhatsApp"/>
  </a>
</div>

### 📞 Get in Touch
- **Email**: [info@faceplugin.com](mailto:info@faceplugin.com)
- **WhatsApp**: [+1 (469) 278-4822](https://wa.me/+14692784822)
- **Website**: [faceplugin.com](https://faceplugin.com/)

---

<div align="center">
  <sub>Made with ❤️ by <a href="https://faceplugin.com">Faceplugin</a></sub>
</div>
