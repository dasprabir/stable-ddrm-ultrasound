# Stable DDRM for Ultrasound Imaging

This repository contains the code and results for my Master’s Thesis project:  
**“Super-Resolution in Ultrasound Imaging using Diffusion Models”**  

The work explores how **Denoising Diffusion Restoration Models (DDRM)** can be adapted for ultrasound RF/B-mode data to improve image resolution and quality.

> 🎓 Research conducted at [IRIT – MINDS Team](https://www.irit.fr/en/departement/dep-signals-and-images/minds-team/), Université Toulouse III – Paul Sabatier  
> Supervisors: **Dr. Duong-Hung Pham** & **Prof. Denis Kouamé**

---

## 📌 Main Results

The key experiments and results are documented in this notebook:  
👉 [**My_Result.ipynb**](./My_Result.ipynb)

It includes:
- DDRM reconstructions on **simulated** and **in-vivo** ultrasound datasets  
- Comparisons with **Wiener/Tikhonov** deconvolution baselines  
- B-mode visualization with envelope detection + log compression  
- Evaluation metrics: **PSNR** and **SSIM**

---

## ✨ Features

- **DDRM sampling** adapted to ultrasound inverse problems  
- **Physics-informed degradations** using PSF and BCCB operators  
- Classical **Wiener/Tikhonov baselines** for fair comparison  
- Configurations for multiple datasets:
  - PICMUS  
  - Besson  
  - In-vivo carotid  
  - Phantom  

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/dasprabir/stable-ddrm-ultrasound.git
cd stable-ddrm-ultrasound
