# 🧠 UTKFace Ethnicity Classification with CNN (PyTorch)

This project focuses on building a Convolutional Neural Network (CNN) using **PyTorch** to classify facial images into 5 ethnicity categories using the **UTKFace dataset**. The model is trained on image data with labels extracted directly from the filenames.

---

## 📂 Dataset: UTKFace

- 20,000+ in-the-wild facial images
- Labels include: **Age**, **Gender**, and **Ethnicity**
- File name format: `[age]_[gender]_[ethnicity]_[date&time].jpg`
- For this project, only **ethnicity** is used as the target variable
- Link to Data https://www.kaggle.com/datasets/jangedoo/utkface-new/data

> **Ethnicity Classes:**
> - 0 → White  
> - 1 → Black  
> - 2 → Asian  
> - 3 → Indian  
> - 4 → Others

---

## 💡 Project Objective

To train a CNN to **accurately predict the ethnicity** of individuals from facial images using:
- Data preprocessing & balancing
- Custom PyTorch Dataset class
- Weighted random sampling for fair training
- Visualization of predictions and performance metrics

---

## 🏗️ Architecture

- 3 Convolutional Layers (Conv2D + ReLU + MaxPooling)
- Flatten → Fully Connected Layers
- Dropout Layer (0.5) to reduce overfitting
- Final Linear Layer with 5 output classes

---

## 🧪 Evaluation

- **Training loss** and **accuracy** over 30 epochs
- **Confusion matrix heatmap** for test predictions
- **Visual examples**: Correct vs Incorrect predictions by ethnicity
- **Classification report** with precision, recall, F1-score

---

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/utkface-ethnicity-cnn.git
cd utkface-ethnicity-cnn

# Install dependencies
pip install -r requirements.txt

# Download UTKFace dataset and place inside /data/UTKFace
