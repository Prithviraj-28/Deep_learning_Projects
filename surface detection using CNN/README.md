# Industrial Surface Crack Detection using CNN

An end-to-end Deep Learning solution for automated **Industrial Surface Crack Detection** built with **TensorFlow / Keras**. This system processes surface images, automatically splits them into training, validation, and test datasets, applies real-time data augmentation, trains a custom 4-block Convolutional Neural Network (CNN), evaluates performance with detailed metrics, and performs single-image inference.

---

## 📌 Features

- **Automated Industrial Dataset Organization**: Automatically scans `CrackDataset/Positive` and `CrackDataset/Negative`, creates standard `train`, `validation`, and `test` folder structures, and performs clean, randomized splitting (70% Train, 15% Validation, 15% Test) with safe file cleanup handling.
- **Real-Time Data Augmentation**: Enhances model robustness using image normalization, random rotations ($\pm 15^\circ$), shifts, zoom variations, and horizontal flips.
- **Custom Deep CNN Architecture**: Features 4 Convolutional blocks (32, 64, 128, 256 filters) coupled with **Batch Normalization** for activation stabilization, **Max Pooling** for spatial downsampling, and **Dropout** regularization (50% and 30%) to prevent overfitting.
- **Production-Grade Training Callbacks**:
  - `EarlyStopping`: Halts training when validation loss stops improving to avoid overfitting.
  - `ModelCheckpoint`: Automatically saves the best model checkpoint (`Best_Crack_Detection_Model.keras`) based on peak validation accuracy.
  - `ReduceLROnPlateau`: Dynamically reduces the learning rate when validation loss reaches a plateau.
- **Comprehensive Evaluation**: Generates training vs. validation Accuracy and Loss plots, evaluates performance on unseen test data, and produces a full **Confusion Matrix** and **Classification Report** (Precision, Recall, F1-Score).
- **Single Image Inference Pipeline**: Includes a visual inference function to test individual surface images and plot predictions directly.

---

## 📁 Project Structure

```
marvellous_cnn_surface_crack_detection/
├── CrackDataset/
│   ├── Positive/                       # Raw images containing surface cracks
│   └── Negative/                       # Raw images without surface cracks
├── Processed_CrackDataset/             # Auto-generated 70/15/15 split
│   ├── train/
│   │   ├── Crack/
│   │   └── NoCrack/
│   ├── validation/
│   │   ├── Crack/
│   │   └── NoCrack/
│   └── test/
│       ├── Crack/
│       └── NoCrack/
├── Best_Crack_Detection_Model.keras    # Model checkpoint saved during training
├── Final_Marvellous_Crack_Detection_Model.keras # Final trained model
├── Marvellous_CNN_Surface_Crack_Detection.py    # Main training & evaluation script
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
- **Python**: 3.8+
- **Pip**: Latest version

### 2. Setup Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the included `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

Organize your raw image dataset inside the `CrackDataset` folder before running the script:

```
CrackDataset/
├── Positive/   <-- Place all images showing surface cracks here
└── Negative/   <-- Place all images with clean/smooth surfaces here
```

> **Supported Image Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

When you execute the script, it will automatically construct `Processed_CrackDataset` with a 70% / 15% / 15% split across `train`, `validation`, and `test` directories.

---

## ⚙️ Model Architecture & Hyperparameters

### Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Input Shape** | $128 \times 128 \times 3$ | RGB Image Resolution |
| **Batch Size** | 32 | Images per gradient update |
| **Max Epochs** | 15 | Maximum full training passes |
| **Optimizer** | Adam | Adaptive moment estimation |
| **Loss Function** | Binary Crossentropy | Loss metric for binary classification |
| **Random Seed** | 42 | Ensures reproducible data splits |

### Network Summary
1. **Input Layer**: $128 \times 128 \times 3$
2. **Conv Block 1**: `Conv2D(32, 3x3, relu)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D(2x2)`
3. **Conv Block 2**: `Conv2D(64, 3x3, relu)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D(2x2)`
4. **Conv Block 3**: `Conv2D(128, 3x3, relu)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D(2x2)`
5. **Conv Block 4**: `Conv2D(256, 3x3, relu)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `MaxPooling2D(2x2)`
6. **Flatten Layer**: Reshapes feature maps to a 1D vector
7. **Dense Layer 1**: `Dense(256, relu)` $\rightarrow$ `Dropout(0.5)`
8. **Dense Layer 2**: `Dense(128, relu)` $\rightarrow$ `Dropout(0.3)`
9. **Output Layer**: `Dense(1, sigmoid)` (Outputs probability: 0 for No Crack, 1 for Crack)

---

## 🚀 How to Run

Execute the main script to start data preparation, model training, evaluation, and single-image testing:

```bash
python Marvellous_CNN_Surface_Crack_Detection.py
```

### Execution Flow:
1. **Dataset Verification**: Validates raw image paths in `CrackDataset`.
2. **Dataset Splitting**: Populates `Processed_CrackDataset` (cleans up any previous runs cleanly).
3. **Augmentation & Loading**: Prepares `ImageDataGenerator` data streams for train/val/test sets.
4. **Visualization**: Displays sample training images with labels.
5. **Model Compilation & Training**: Runs CNN training with callbacks monitoring `val_loss` and `val_accuracy`.
6. **Plotting**: Displays training vs. validation accuracy and loss graphs.
7. **Testing & Metrics**: Evaluates model on unseen test dataset and prints the **Confusion Matrix** & **Classification Report**.
8. **Single Image Prediction**: Predicts and displays a sample image from the test set.

---

## 📈 Outputs & Saved Models

Upon successful execution, the following files are saved in the project root:
- `Best_Crack_Detection_Model.keras`: Saved during training whenever validation accuracy reaches a new high.
- `Final_Marvellous_Crack_Detection_Model.keras`: The final model saved after training completes.

---

## 🤝 Acknowledgments

Developed as part of the **Marvellous Infosystems** Deep Learning curriculum for Industrial Surface Crack Detection.
