<img width="1094" height="893" alt="image" src="https://github.com/user-attachments/assets/c1094951-d4e8-4c69-8a65-976086a7b352" /># Fine-Grained Heterogeneous Change Detection in Complex Disaster Response with Wavelet-Based Spatial-Frequency Coupled Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official dataset and resources for the paper: **"Fine-Grained Heterogeneous Change Detection in Complex Disaster Response with Wavelet-Based Spatial-Frequency Coupled Learning"**.

## 📢 News
* **[Date]**: The dataset is now available via Google Drive!
* **[Status]**: The training and testing code will be released upon the acceptance of the paper.

---

## 👥 Authors & Affiliations

**Yang Yang$^a$, Jun Pan$^{a,e,f,*}$, Qiqi Zhu$^b$, Rui Xu$^a$, Xiaoyu Yu$^c$, Junli Li$^d$, Mi Wang$^{a,e,f}$**

* $^a$ State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China
* $^b$ School of Geography and Information Engineering, China University of Geosciences, Wuhan 430079, China
* $^c$ Hangzhou International Innovation Institute, Beihang University, Hangzhou 311115, China
* $^d$ National Key Laboratory of Ecological Security and Sustainable Development in Arid Region, Xinjiang Institute of Ecology and Geography, Chinese Academy of Sciences, Urumqi 830011, China
* $^e$ Oriental Space Port Research Institute, Yantai, 265100, China
* $^f$ Hubei Luojia Laboratory, 430079, Wuhan, China

\* *Corresponding Author: Jun Pan*

---

## 📖 Introduction

In complex disaster response scenarios, detecting fine-grained changes using heterogeneous data sources (e.g., Optical and SAR) is critical but challenging. This paper proposes a novel **Wavelet-Based Spatial-Frequency Coupled Learning** framework to address these challenges.

### Model Architecture
Our method utilizes a Spatial-Frequency Coupled Encoder and a Multi-scale Inverse Wavelet Decoder to effectively fuse information and detect changes.

![Model Architecture](![Uploading image.png…]
*Figure 1: The overall architecture of the proposed method, featuring the Spatial-Frequency Collaborative Extraction Module and Cross-Channel 3D Fusion Module.*

---

## 📂 Dataset

We have constructed a self-made dataset specifically designed for heterogeneous change detection in disaster scenarios. The dataset covers significant events such as the **Islahiye, Turkey Earthquake** and the **Derna, Libya Flood**.

### Dataset Features
* **Pre-event Data (T1):** Optical Imagery (Landsat OLI)
* **Post-event Data (T2):** SAR Imagery (Capella)
* **Labels:** Expert visual interpretation and ground truth masks.

![Dataset Preview](dataset_preview.jpg)
*Figure 2: Examples of the dataset. (A) Earthquake Case Study in Islahiye, Turkey. (B) Flood Case Study in Derna, Libya. The bottom rows show detailed comparisons between Optical, SAR, and Labels.*

### 📥 Download
The dataset is available for academic research purposes. You can download it via the link below:

[**🔗 Google Drive Link: Download Dataset**](https://drive.google.com/drive/folders/1KDdnoqa-a-s-RZTE7b-1EDngcQPbAgUn?usp=drive_link)

---

## 💻 Code Availability

The implementation of our proposed model, including training scripts and testing protocols, is currently under review.

**The full source code will be made publicly available in this repository immediately after the paper is accepted.**

Please star 🌟 this repository to stay updated!

---

## 📝 Citation

If you find this work or dataset useful for your research, please consider citing our paper (BibTeX will be updated upon publication):

```bibtex
@article{yang2024fine,
  title={Fine-Grained Heterogeneous Change Detection in Complex Disaster Response with Wavelet-Based Spatial-Frequency Coupled Learning},
  author={Yang, Yang and Pan, Jun and Zhu, Qiqi and Xu, Rui and Yu, Xiaoyu and Li, Junli and Wang, Mi},
  journal={Submitted to Journal Name},
  year={2024}
}
