---

# **Face Aging with Conditional GANs**

<p align="center">
  <img src="path/to/qualitative_results_image.png" alt="Qualitative Results" width="600"/>
</p>

---

### **Overview**
This project explores the application of **Conditional Generative Adversarial Networks (CGANs)** for realistic face aging. By leveraging the UTKFace dataset and incorporating advanced architectural elements like attention blocks and perceptual loss, the model synthesizes age progression and regression while preserving individual identity.

### **Features**
- **Data Preprocessing**: Automatic resizing, normalization, and age-label categorization.
- **Model Design**:
  - **Generator**: Noise and label-conditioned synthesis with attention blocks.
  - **Discriminator**: Real/fake classification and feature matching with attention mechanisms.
- **Loss Functions**:
  - Adversarial Loss
  - Perceptual Loss (using VGG16)
  - Feature Matching Loss
- **Visualization**: Epoch-wise loss curves, age distribution, and qualitative results.

---

### **Visualizations**
#### **Sample Age Transformations**
<p align="center">
  <img src="path/to/epoch_wise_loss_and_data_visualization_during_training.png" alt="Sample Transformations" width="800"/>
</p>

#### **Training Loss Curves**
<p align="center">
  <img src="path/to/training_losses_plotted_on_line_graph_with_generator_and_discriminator_loss.png" alt="Training Losses" width="600"/>
</p>

#### **Perceptual and Feature Loss**
<p align="center">
  <img src="path/to/perceptual_and_feature_loss_plotted_with_line_graph.png" alt="Perceptual and Feature Loss" width="600"/>
</p>

---

### **Installation**
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/face-aging-cgan.git
   cd face-aging-cgan
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Dataset**
   - Download the [UTKFace dataset](https://susanqq.github.io/UTKFace/).
   - Extract and place the dataset in the `data/` directory.
   - Ensure the directory structure is as follows:
     ```
     data/
     ├── UTKFace/
     │   ├── 0_0_0_20170110222108075.jpg
     │   ├── ...
     ```

---

### **Usage**
#### **Train the Model**
Run the following command to train the Conditional GAN:
```bash
python train.py --dataset_path data/UTKFace --epochs 20 --batch_size 64
```

#### **Visualize Results**
To generate and save qualitative results:
```bash
python generate.py --model_path saved_models/generator_final.h5 --output_dir results/
```

#### **Plot Loss Curves**
Visualize training loss curves and perceptual loss:
```bash
python plot_losses.py --log_dir logs/
```

---

### **Project Structure**
```
face-aging-cgan/
├── data/                   # Dataset folder
├── models/                 # Model architectures
├── utils/                  # Utility scripts (e.g., data preprocessing, plotting)
├── train.py                # Training script
├── generate.py             # Inference and image generation
├── plot_losses.py          # Script to plot training losses
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

### **Results**
Our CGAN model demonstrates effective age synthesis across categories:
- **Child**: 0–12 years
- **Young**: 13–30 years
- **Middle-aged**: 31–50 years
- **Old**: 51+ years

The results highlight:
- Realistic texture and feature progression.
- Preservation of identity across age transformations.

---

### **Future Work**
- Extending the model for real-time age synthesis.
- Incorporating multimodal inputs like gender and ethnicity.
- Evaluating on diverse datasets for better generalization.

---

### **References**
- Goodfellow, Ian, et al. "Generative adversarial nets." *NIPS*, 2014.
- Karras, Tero, et al. "A style-based generator architecture for generative adversarial networks." *CVPR*, 2019.
- Zhang, Z., et al. "Age progression/regression by conditional adversarial autoencoder." *CVPR*, 2017.

---

### **Contributors**
- **Bilal Shakeel** - Research and Implementation
- **Muhammad Uziar** - Model Design and Optimization
- **Hassan Abbas** - Data Visualization and Documentation

---

### **License**
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
