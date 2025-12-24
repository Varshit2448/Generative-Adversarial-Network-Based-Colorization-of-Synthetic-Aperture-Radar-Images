# Generative Adversarial Network Based Colorization of Synthetic Aperture Radar Images

> A GAN-based pipeline to colorize Synthetic Aperture Radar images by translating VV polarization data into RGB representations, preserving spatial structure and enhancing visual details through an encoder–decoder architecture with adversarial training.

---

## 1. Problem Statement

Manual interpretation of Synthetic Aperture Radar (SAR) images is challenging due to grayscale representation and low visual clarity. This project automates SAR image colorization to enhance interpretability, aiding analysts in remote sensing, agriculture, and defense applications.

---

## 2. Objectives

- Develop a GAN-based model to colorize SAR images.
- Preserve structural details and spatial integrity.
- Improve edge sharpness and texture fidelity.
- Provide a reproducible training and inference pipeline.

---

## 3. Dataset Description

- Source: Custom SAR VV polarization images and corresponding RGB images (https://www.kaggle.com/datasets/shatish2403/sar-opticalvvvhhhhvrgb).
- Total Images: ~5,000 paired samples.
- Split: 70% Train, 20% Validation, 10% Test.
- Preprocessing: 
  - Resize images to 120×120
  - Normalize pixel values to [0,1]
  - Convert VV images to single-channel grayscale, RGB images to 3 channels.

---

## 4. Model Architecture

### Generator (Encoder–Decoder)
- Encoder: Convolutional layers with LeakyReLU activations.
- Bottleneck: Residual blocks to preserve structure.
- Decoder: Transposed convolution layers with skip connections.
- Output: 3-channel RGB image.
- Loss Function: Combination of L1, SSIM, Sobel edge, and adversarial loss.

### Discriminator (PatchGAN)
- Convolutional layers with LeakyReLU and BatchNormalization.
- Classifies local patches for sharper texture generation.
- Loss Function: Binary Crossentropy.

<img width="617" height="672" alt="image" src="https://github.com/user-attachments/assets/2a931dfd-300a-46c9-8245-fc9c2330ee7b" />


---

## 5. Training Details

- Epochs: 150 (initial training), 40 (fine-tuning)
- Batch Size: 64 (initial), 1–16 (fine-tuning)
- Learning Rate: 2e-4 (initial), 5e-5 (fine-tuning)
- Optimizer: Adam with β1 = 0.5
- Hardware: GPU (NVIDIA)
- Techniques: Encoder freezing during fine-tuning, edge-aware loss, skip connections.

---

## 6. Evaluation Metrics

- L1 Loss
- Structural Similarity Index (SSIM)
- Edge preservation (Sobel / Laplacian metrics)
- Visual inspection of generated RGB images against ground truth.

---

## 7. Results

- Generated RGB images show enhanced visual clarity and preserved SAR structures.
- Fine-tuning improved edge sharpness and reduced color bleeding.
- Visual outputs closely match ground truth in terms of textures and contours.

---

## 8. How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/username/sar-colorization-gan

# 2. Install dependencies
pip install -r requirements.txt

#3. Usage of pretrained model
generator = tf.keras.models.load_model(
    "generator_model.keras",
    compile=False
)

```

## 9. References

1.Liang, Yihuai, et al. "Unpaired medical image colorization using generative adversarial network." Multimedia Tools and Applications 81.19 (2022): 26669-26683.
2.Chen, Yu, et al. "Exploring efficient and effective generative adversarial network for thermal infrared image colorization." Complex & intelligent systems 9.6 (2023): 7015-7036.
