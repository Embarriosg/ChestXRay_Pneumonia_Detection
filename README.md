# Chest X-Ray Pneumonia Detection

A deep learning project implementing and comparing Convolutional Neural Networks (CNN) and Transfer Learning approaches (ResNet-18) for automated pneumonia detection from chest radiographs, with Grad-CAM visualization for model interpretability.

## Project Overview

Pneumonia represents a significant global health challenge, particularly affecting vulnerable populations. This project develops and evaluates machine learning models capable of classifying chest X-ray images as either normal or indicative of pneumonia. The investigation compares a custom CNN architecture against a transfer learning approach utilizing ResNet-18, while implementing Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explanation of model predictions.

## Key Results

- **Custom CNN**: Achieved 89.58% test accuracy with 94.1% sensitivity and 82.1% specificity (AUC-ROC: 0.948)
- **ResNet-18 Transfer Learning**: Achieved 92.47% test accuracy with 97.9% sensitivity and 82.1% specificity (AUC-ROC: 0.962)
- **Grad-CAM Visualization**: Successfully identified clinically relevant lung regions corresponding to pathological features

## Repository Structure

```
ChestXRay_Pneumonia_Detection/
├── data/
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
├── src/
│   └── data/
│       └── dataloaders.py
├── results/
│   └── figures/
│       ├── Normal_GradCAM_Batch.png
│       ├── Overrall_model_performance.png
│       ├── Pneumonia_GradCAM.jpeg
│       ├── Pneumonia_GradCAM_Batch.png
│       ├── Reduction_missed_pneumonia.png
│       ├── cnn_evaluation.png
│       ├── cnn_training_curves.png
│       ├── neumonia-normal.png
│       ├── resnet_evaluation.png
│       └── resnet_training_curves.png
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_model.ipynb
│   ├── 03_resnet_transfer_learning.ipynb
│   ├── 04_gradcam_visualization.ipynb
│   └── 05_results_analysis.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset Information

The project utilizes a chest X-ray dataset comprising:
- **Training Set**: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Test Set**: 624 images (234 Normal, 390 Pneumonia)
- **Validation Set**: 16 images (excluded due to insufficient size)

The dataset exhibits class imbalance favoring pneumonia cases, which was considered in model training and evaluation strategies.
[1] P. T. Mooney, "Chest X-Ray Images (Pneumonia)," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.

## Methodology

### Custom CNN Architecture

A three-block convolutional neural network was designed from scratch, featuring:
- Progressive channel expansion (32 → 64 → 128 filters)
- MaxPooling layers for spatial dimension reduction
- Fully connected layers with dropout regularization (0.5)
- Aggressive data augmentation (random rotation, horizontal flipping)

### ResNet-18 Transfer Learning

A two-phase training strategy was implemented:

**Phase 1 - Feature Extraction**: Convolutional backbone frozen, training only the custom classification head for initial adaptation.

**Phase 2 - Fine-Tuning**: Selective unfreezing of layer4 and fully connected layers with reduced learning rate (5×10⁻⁵) for specialized adaptation to chest X-ray features.

Conservative data augmentation was applied, leveraging ImageNet pre-trained weights for initialization.

### Grad-CAM Implementation

Gradient-weighted Class Activation Mapping was applied to the final convolutional layer (layer4) of ResNet-18 to generate visual explanations. The technique highlights image regions most influential in classification decisions, enabling validation that the model attends to clinically relevant anatomical features.

## Technical Requirements

```
python>=3.8
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
opencv-python>=4.6.0
tqdm>=4.64.0
Pillow>=9.0.0
```

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ChestXRay_Pneumonia_Detection.git
cd ChestXRay_Pneumonia_Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Training Custom CNN
```bash
jupyter notebook notebooks/02_cnn_model.ipynb
```

### Training ResNet-18
```bash
jupyter notebook notebooks/03_resnet_transfer_learning.ipynb
```

### Generating Grad-CAM Visualizations
```bash
jupyter notebook notebooks/04_gradcam_visualization.ipynb
```

## Model Performance Comparison

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | False Negatives |
|-------|----------|-------------|-------------|---------|-----------------|
| Custom CNN | 89.58% | 94.1% | 82.1% | 0.948 | 23 |
| ResNet-18 | 92.47% | 97.9% | 82.1% | 0.962 | 8 |

The ResNet-18 transfer learning approach demonstrated superior performance, particularly in sensitivity (pneumonia detection), reducing false negatives by 65% compared to the custom CNN architecture.

## Key Findings

The investigation yielded several important insights:

The transfer learning approach substantially outperformed the custom CNN, validating the utility of pre-trained representations even when source and target domains differ significantly (natural images versus medical radiographs). The ResNet-18 model achieved clinically relevant sensitivity (97.9%), approaching thresholds suitable for screening applications where minimizing missed diagnoses is paramount.

Grad-CAM visualization confirmed that learned representations correspond to anatomically plausible pathological features rather than spurious dataset artifacts. The model consistently attended to lung consolidation regions, opacities, and infiltrates characteristic of pneumonia manifestation.

Both architectures exhibited identical specificity (82.1%), suggesting systematic challenges in normal radiograph classification that transcend architectural choices. This pattern may reflect inherent annotation ambiguities or the presence of borderline cases requiring nuanced clinical judgment.

## Limitations

Several constraints warrant acknowledgment:

The absence of a properly sized validation set necessitated using the test set for model selection, introducing potential optimization bias. The class imbalance favoring pneumonia cases may have contributed to the asymmetry between sensitivity and specificity. The binary classification framework represents a simplification of clinical differential diagnosis requirements. Dataset provenance and annotation methodology lack comprehensive documentation regarding inter-rater reliability and demographic characteristics.

## Future Directions

Recommended enhancements include:

Acquisition of larger, balanced datasets with proper train-validation-test partitioning and multiple annotator consensus labels. Expansion to multi-class classification incorporating common differential diagnoses such as congestive heart failure, atelectasis, and pleural effusion. Integration of patient metadata for multimodal learning approaches. Implementation of uncertainty quantification techniques to provide confidence estimates for individual predictions. External validation on independent datasets from diverse institutions and geographic regions. Exploration of advanced architectures including Vision Transformers and hybrid CNN-Transformer models.

## References

**CNN Architecture**:
O'Shea, K., & Nash, R. (2015). An Introduction to Convolutional Neural Networks. arXiv:1511.08458.

**ResNet Architecture**:
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

**Grad-CAM Visualization**:
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. International Conference on Computer Vision (ICCV).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

The dataset utilized in this project was sourced from publicly available medical imaging repositories. The implementation leverages PyTorch and torchvision libraries for deep learning development, with visualization support from matplotlib, seaborn, and OpenCV.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the GitHub repository or contact the project maintainer.

---

**Note**: This project is intended for research and educational purposes. The models have not undergone clinical validation and should not be used for actual medical diagnosis without appropriate regulatory approval and expert oversight.
