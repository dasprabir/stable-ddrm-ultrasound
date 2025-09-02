
**“Ultrasound Image Deconvolution via Block Circulant Spectral Decomposition in Diffusion Models”**

*A physics-guided diffusion approach for ultrasound image deconvolution *

This repository contains the code and results of my Master’s Thesis Internship project:  
  

The work proposes **Fourier-DDRM**, a variant of Denoising Diffusion Restoration Models (DDRM) where the expensive SVD projection is replaced by a **Block-Circulant with Circulant Blocks (BCCB) / Fourier spectral decomposition**. This approach accelerates ultrasound image restoration while preserving physics consistency and improving reconstruction quality.

> 🎓 Internship at [IRIT – MINDS Team](https://www.irit.fr/departement/ics/minds/), Université Toulouse III – Paul Sabatier  
> Academic Supervisors: **Dr. Duong-Hung Pham**, **Prof. Denis Kouamé**, **Dr. Julien Fade**

---

##  Motivation

Ultrasound B-mode images are degraded by convolution with the **point spread function (PSF)** and corrupted by noise:

```

Y = H  X + N

````

Classical Wiener/Tikhonov filters oversmooth fine structures. Deep learning methods require supervised paired data and may distort speckle.  
**Diffusion models** offer strong generative priors, but standard DDRM relies on SVD decompositions that are unstable for oscillatory RF PSFs and computationally heavy.

---

##  Contributions

- **Fourier-DDRM**: replaces SVD by FFT-based BCCB projections (`H = F† Λh F`), reducing runtime and memory.  
- **Physics-informed priors**: explicitly integrate ultrasound forward model into the diffusion sampling process.  
- **Comprehensive evaluation**: on simulated phantoms and in vivo mouse kidney data.  
- **Ablations & Efficiency**: analysis of runtime, stability, and contrast/resolution trade-offs.  

---

## 📊 Key Results

### Simulation Data
- Fourier-DDRM achieves **higher PSNR/SSIM** than Tikhonov and vanilla DDRM.  
- Example metrics (BSNR 20 dB):  
  - Tikhonov: 22.7 dB PSNR, SSIM 0.63  
  - DDRM: 24.9 dB PSNR, SSIM 0.72  
  - **Fourier-DDRM**: 26.3 dB PSNR, SSIM 0.78  

### In Vivo Data
- Improves **Resolution Gain (RG)** compared to Tikhonov and DDRM.  
- Slight decrease in **Contrast Ratio (CR)**, as pretrained diffusion priors may treat speckle as texture.  
- Fourier-DDRM offers better **edge sharpness** and lower runtime.  

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/dasprabir/stable-ddrm-ultrasound.git
cd stable-ddrm-ultrasound
````

Set up the environment:

```bash
conda create -n stable-ddrm python=3.10 -y
conda activate stable-ddrm
pip install -r requirements.txt
```

Minimal requirements:

```
numpy
scipy
matplotlib
pywt
jupyter
torch>=2.0
torchvision
```

---

## Usage

### Run the main results notebook

```bash
jupyter notebook My_Result.ipynb
```

### Example: deblurring with configs

```bash
python main.py --ni \
  --config configs/deblur_besson.yml \
  --doc imagenet_ood \
  --timesteps 20 \
  --eta 0.85 \
  --deg deblur_bccb \
  --sigma_0 15
```



---

## 📁 Repository Structure

```
configs/           # YAML configs for experiments
datasets/          # Data loaders
exp/datasets/      # Dataset files (tracked with Git LFS)
functions/, models/, runners/   # Core DDRM + Fourier operators
guided_diffusion/  # UNet & diffusion backbone
main.py            # Main training/inference script
My_Result.ipynb    # 📌 Main results notebook
```
---

## 📂 MATLAB Codes

The repository includes a [`Matlabfiles/`](./Matlabfiles) directory with supporting MATLAB scripts:

- **ComputeMetrics_Visualization/**  
  Scripts for computing PSNR, SSIM, Contrast Ratio (CR), Resolution Gain (RG) and visualizing RF → B-mode conversions.  
  Key scripts: `computeMetrics.m`, `resolution_gain.m`, `ContrastRatio.m`, `rf2bmode.m`.

- **Psf_Estimation/**  
  Scripts for ultrasound Point Spread Function (PSF) estimation and demos.  
  Key scripts: `calling_psf.m`, `demo.m`, `psf_demo_new_data.m`.


---

##  Data

* **Simulated phantoms** (scatterers convolved with anisotropic PSFs + noise)
* **In vivo mouse kidney B-mode** acquisitions

> Large datasets are excluded from the repo and tracked with [Git LFS](https://git-lfs.github.com).

---

##  Future Work

* **Speckle-aware fine-tuning** of diffusion priors to improve contrast ratio (CR).
* **Domain adaptation** of noise model (RF vs B-mode, multiplicative vs additive).
* **Depth-adaptive Λh** for non-stationary PSFs without resorting to full SVD.

---

##  Citation

If you use this work, please cite:

```bibtex
@misc{das2025fourierddrm,
  title  = {Ultrasound Image Deconvolution via Block Circulant Spectral Decomposition in Diffusion Models},
  author = {Prabir Kumar Das},
  year   = {2025},
  note   = {Master’s Thesis Internship, IRIT/MINDS}
}
```


##  Acknowledgements

* Supervisors: Dr. Duong-Hung Pham, Prof. Denis Kouamé, Dr. Julien Fade
* MINDS team members Vassili & Arthur for technical discussions
* IRIT & Centrale Méditerranée for providing resources


