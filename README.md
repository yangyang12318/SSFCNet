
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official dataset and resources for the paper: **"Fine-Grained Heterogeneous Change Detection in Complex Disaster Response with Wavelet-Based Spatial-Frequency Coupled Learning"**.

## 📢 News
* **[Date]**: The dataset is now available via Google Drive!
* **[Status]**: The training and testing code will be released upon the acceptance of the paper.

---

## 👥 Authors & Affiliations

**Yang Yang<sup>a</sup>, Jun Pan<sup>a,e,f,*</sup>, Qiqi Zhu<sup>b</sup>, Rui Xu<sup>a</sup>, Xiaoyu Yu<sup>c</sup>, Junli Li<sup>d</sup>, Mi Wang<sup>a,e,f</sup>**

* <sup>a</sup> State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China
* <sup>b</sup> School of Geography and Information Engineering, China University of Geosciences, Wuhan 430079, China
* <sup>c</sup> Hangzhou International Innovation Institute, Beihang University, Hangzhou 311115, China
* <sup>d</sup> National Key Laboratory of Ecological Security and Sustainable Development in Arid Region, Xinjiang Institute of Ecology and Geography, Chinese Academy of Sciences, Urumqi 830011, China
* <sup>e</sup> Oriental Space Port Research Institute, Yantai, 265100, China
* <sup>f</sup> Hubei Luojia Laboratory, 430079, Wuhan, China

---

## 📖 Introduction

In complex disaster response scenarios, detecting fine-grained changes using heterogeneous data sources (e.g., Optical and SAR) is critical but challenging. This paper proposes a novel **Wavelet-Based Spatial-Frequency Coupled Learning** framework to address these challenges.

### Model Architecture
Our method utilizes a Spatial-Frequency Coupled Encoder and a Multi-scale Inverse Wavelet Decoder to effectively fuse information and detect changes.


<img width="1094" height="893" alt="image" src="https://github.com/user-attachments/assets/cdd0af38-2336-4b24-95a9-1a236a9b1532" />
*Figure 1: The overall architecture of the proposed method, featuring the Spatial-Frequency Collaborative Extraction Module and Cross-Channel 3D Fusion Module.*

---

## 📂 Dataset

We have constructed a self-made dataset specifically designed for heterogeneous change detection in disaster scenarios. The dataset covers significant events such as the **Islahiye, Turkey Earthquake** and the **Derna, Libya Flood**.

### Dataset Features
* **Pre-event Data (T1):** Optical Imagery (Google Earth)
* **Post-event Data (T2):** SAR Imagery (Capella-X)
* **Labels:** Expert visual interpretation and ground truth masks.


<img width="934" height="961" alt="image" src="https://github.com/user-attachments/assets/0d1b3629-d5c4-477f-9aa3-0a7158cf3f75" />
*Figure 2: Examples of the dataset. (A) Earthquake Case Study in Islahiye, Turkey. (B) Flood Case Study in Derna, Libya. The bottom rows show detailed comparisons between Optical, SAR, and Labels.*

### 📥 Download
The dataset is available for academic research purposes. You can download it via the link below:

[**🔗 Google Drive Link: Download Dataset**](https://drive.google.com/drive/folders/1KDdnoqa-a-s-RZTE7b-1EDngcQPbAgUn?usp=drive_link)

---
If you have any questions or suggestions, plased to contact me. Email: yangyang1@whu.edu.cn or panjun1215@whu.edu.cn
