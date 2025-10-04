# 🧠 Generative Adversarial Network (GAN) Collection
*A complete journey from simple GANs to advanced architectures for face and text-to-image synthesis.*

---

## 📘 Overview
This repository aggregates multiple **GAN-based projects** implemented using **PyTorch** and **TensorFlow**, showcasing the evolution from foundational adversarial models to advanced architectures such as **DCGAN**, **Pix2Pix**, **StackGAN**, and **StyleGAN2-ADA**.  

Each sub-project includes code, documentation, dataset references, and visuals to help learners and practitioners understand how adversarial training can generate, reconstruct, and manipulate images.

---

## 🗂️ Project Index

| No. | Project | Architecture | Domain | Highlights |
|-----|----------|---------------|---------|-------------|
| 1 | Generative Adversarial Networks in Slanted Land | Simple GAN | 2×2 grayscale generation | Intro to GAN mechanics |
| 2 | Generating Fake Faces with DCGAN | DCGAN | CelebA Faces | Large-scale face synthesis |
| 3 | Front Face Generator using Pix2Pix | Pix2Pix | Image-to-Image (Conditional GAN) | Side-to-frontal face generation |
| 4 | StackGAN | StackGAN Stage-I & II | Text-to-Image | Bird image synthesis from captions |
| 5 | StyleGAN2 with Adaptive Discriminator Augmentation (ADA) | StyleGAN2-ADA | High-Res Image Synthesis | Advanced augmentation & training stabilization |

---

## 1️⃣ Generative Adversarial Networks in Slanted Land
A minimal GAN demonstration inspired by the video *A Friendly Introduction to GANs*.
This project introduces the **generator–discriminator game** using 2×2 black-and-white image generation.

**Key Concepts**
- Random noise → binary image generation  
- One-layer Generator and Discriminator  
- Understanding adversarial loss and convergence behavior

---

## 2️⃣ Generating Fake Faces with DCGAN
This project uses the **Deep Convolutional GAN (DCGAN)** architecture to generate synthetic human faces trained on the **CelebA dataset**.

**Features**
- Convolutional Generator & Discriminator  
- BatchNorm + ReLU/LeakyReLU activations  
- Latent vector sampling & image visualization  
- Code walkthrough available via educational video

---

## 3️⃣ Front Face Generator using Pix2Pix
A **conditional GAN** that transforms side or angled facial images into **frontal faces** using the **Pix2Pix** model.  

**Pipeline**
1. Select experiment type and define parameters  
2. Load pre-trained Pix2Pix model  
3. Align and preprocess input image  
4. Perform inference to generate frontal face  
5. Visualize before/after comparison  

**Requirements**
- Python 3.x  
- TensorFlow / PyTorch  
- OpenCV  
- Jupyter Notebook for experiments

---

## 4️⃣ StackGAN
Two-stage **Text-to-Image synthesis** pipeline that generates high-resolution images from natural language descriptions.  

**Dataset:** Caltech-UCSD Birds-200-2011  
**Embeddings:** char-CNN-RNN  

**Project Structure**
```
weights/          # Trained model weights
test/             # Stage-I generated images
results_stage2/   # Final images refined by Stage-II
```

**Key Features**
- Stage-I GAN: Coarse image synthesis from embeddings  
- Stage-II GAN: High-fidelity image refinement  
- char-CNN-RNN text embeddings  
- Customizable training pipeline for other text-image datasets

---

## 5️⃣ StyleGAN2 with Adaptive Discriminator Augmentation (ADA)
Implementation of **StyleGAN2-ADA**, an advanced model capable of generating **photorealistic, high-resolution** images with **data-efficient training**.

**Core Innovations**
- Adaptive Discriminator Augmentation (ADA)  
- Improved latent space regularization  
- Style mixing and stochastic variation  
- Suitable for small dataset training without overfitting  

**Applications**
- Portrait & artistic style generation  
- Domain adaptation (e.g., anime, sketches)  
- Latent vector interpolation for creative exploration  

---

## 🧩 Environment Setup

### Dependencies
Install global dependencies before running any notebook:
```bash
pip install torch torchvision tensorflow opencv-python pillow tqdm matplotlib
```

### Dataset Preparation
Each sub-project includes dataset references. Download and organize as:
```
data/
 ├── celeba/
 ├── cub_200_2011/
 ├── custom_faces/
 └── embeddings/
```

---

## 📈 Results Summary

| Model | Dataset | Output Resolution | Distinctive Feature |
|--------|----------|------------------|--------------------|
| Simple GAN | Synthetic | 2×2 | Minimal GAN concept |
| DCGAN | CelebA | 64×64 | Deep Conv generator |
| Pix2Pix | Custom | 256×256 | Image-to-image translation |
| StackGAN | CUB-200-2011 | 256×256 | Text-to-image synthesis |
| StyleGAN2-ADA | Custom | 1024×1024 | High-res adaptive GAN |

---

## 🧠 Learning Objectives
By exploring this repository, you’ll learn:
- The **adversarial training loop** and loss dynamics.  
- How architectural innovations improve stability and output quality.  
- Dataset preparation and preprocessing techniques.  
- Evaluation via visual inspection and qualitative metrics (FID, IS).  
- Extending GANs to conditional, text-driven, and style-transfer tasks.

---

## 🚀 How to Run
Each folder includes its own notebook (`.ipynb`) or script with setup instructions:
```bash
cd project_folder
jupyter notebook main.ipynb
```

---

## 📚 References
- Goodfellow et al., *Generative Adversarial Networks*, NeurIPS 2014  
- Radford et al., *DCGAN* (2015)  
- Isola et al., *Pix2Pix* (2017)  
- Zhang et al., *StackGAN* (2017)  
- Karras et al., *StyleGAN2-ADA* (2020)  


**Acknowledgments:**  
Special thanks to the open-source community and dataset providers for enabling experimentation with GAN architectures.
