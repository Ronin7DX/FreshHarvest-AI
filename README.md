# 🍓 FreshHarvest AI Freshness Inspector

An AI-powered fruit freshness inspection system built for **FreshHarvest Logistics**, a cold storage and distribution company in California. This project automates the manual quality inspection process using deep learning and computer vision.

---

## 📌 Problem Statement

FreshHarvest Logistics faced operational issues due to inconsistent manual quality inspections:
- Human errors from inconsistent lighting and worker fatigue
- Customer complaints about spoiled produce reaching retailers
- Financial losses from refund requests and declining brand reputation

## 💡 Solution

A CNN-based image classification system integrated into the warehouse conveyor belt infrastructure. High-speed cameras capture fruit images in real time, and the model classifies them as **Fresh** or **Spoiled** instantly.

---

## 🍎 Supported Fruits

| Fruit | Fruit | Fruit | Fruit |
|-------|-------|-------|-------|
| 🍌 Banana | 🍋 Lemon | 🍈 Lulo | 🥭 Mango |
| 🍊 Orange | 🍓 Strawberry | 🍅 Tamarillo | 🍅 Tomato |

---

## 📁 Project Structure

```
FreshHarvest_App/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
│
├── FreshHarvest_Week1_Task1_Preprocessing.py    # Data preprocessing
├── FreshHarvest_Week1_Task2_CNN_Training.py     # Custom CNN training
├── FreshHarvest_Week1_Task3_Optimization.py     # Regularization & tuning
├── FreshHarvest_Week2_Task1_TransferLearning.py # ResNet50 transfer learning
│
├── freshharvest_resnet50.pt            # Saved best model (ResNet50)
├── freshharvest_splits.pt             # Train/Val/Test split indices
│
└── README.md                          # Project documentation
```

---

## 🧠 Model Architecture

### Week 1 — Custom CNN (Baseline)
- 5 Convolutional blocks (Conv → ReLU → MaxPool)
- Fully connected head: 25088 → 512 → 2
- 14.4M parameters
- **Best Test Accuracy: 90.67%** (30 epochs)

### Week 2 — ResNet50 Transfer Learning ✅ Final Model
- Pretrained on ImageNet (frozen backbone → fine-tuned layer4)
- Custom head: Linear(2048→256) → ReLU → Dropout(0.4) → Linear(256→2)
- Two-phase training: Feature Extraction + Fine-tuning
- **Final Test Accuracy: 99.96%** (only 10 epochs!)

---

## 📊 Results Summary

| Model | Epochs | Test Accuracy |
|-------|--------|---------------|
| Custom CNN (no regularization) | 30 | 90.67% |
| Custom CNN (with regularization) | 30 | ~92%+ |
| **ResNet50 Transfer Learning** | **10** | **99.96% ✅** |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- NVIDIA GPU (optional but recommended)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/FreshHarvest_App.git
cd FreshHarvest_App
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🖥️ Streamlit App Features

- **Drag & Drop** image upload (JPG, PNG, WebP)
- **Real-time prediction** — Fresh ✅ or Spoiled ❌
- **Confidence bars** showing probability for each class
- **Action message** — dispatch or remove from conveyor
- Clean dark-themed UI built with custom CSS

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| PyTorch 2.x | Deep learning framework |
| Torchvision | ResNet50 pretrained model |
| Streamlit | Web application |
| Pillow | Image processing |
| NumPy | Numerical operations |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Visualizations |

---

## 📈 Training Details

### Dataset: FRUIT-16K
- 16 folders (8 fruits × 2 classes)
- Split: 70% Train / 15% Validation / 15% Test
- Augmentations: RandomCrop, ColorJitter, Flips, Rotation

### Regularization Techniques Applied
- ✅ Batch Normalization
- ✅ Dropout (50%)
- ✅ Weight Decay (L2 = 1e-4)
- ✅ Early Stopping (patience = 7)

---

## 🔮 How to Load the Model

```python
import torch
import torch.nn as nn
from torchvision import models

checkpoint = torch.load('freshharvest_resnet50.pt', weights_only=False)

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.4),
    nn.Linear(256, 2),
)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"Model loaded! Test Accuracy: {checkpoint['test_acc']:.2f}%")
```

---

## 👨‍💻 Author

**Kavinesh L**
Deep Learning Internship Project — FreshHarvest Logistics
Virtual Internship | Computer Vision | PyTorch

---

## 📄 License

This project is for internship and educational purposes.
