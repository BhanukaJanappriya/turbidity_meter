# 💧 Image-Based Water Turbidity Estimation

This is a 2nd-year **image processing project** that presents a low-cost, image-based approach for estimating **water turbidity** using fundamental image processing techniques in Python. It offers a practical alternative to expensive lab devices like nephelometers by analyzing images of water samples to determine turbidity levels.

---

## 📌 Project Description

Using computer vision and image statistics, this system:

- Converts water sample images into grayscale  
- Applies Gaussian blur and Canny edge detection  
- Computes brightness, intensity variance, edge density  
- Uses red channel analysis for turbidity estimation  
- Predicts turbidity using an exponential model based on NTU calibration

---

## 🔧 Features

- Preprocessing using grayscale + blur  
- Brightness, min/max, and edge metrics  
- Red channel exponential model for NTU estimation  
- Calibration using known turbidity values  
- Visual outputs: image display, histograms  
- User-friendly Python functions and CLI usage  

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/BhanukaJanappriya/turbidity_meter.git
cd turbidity_meter
