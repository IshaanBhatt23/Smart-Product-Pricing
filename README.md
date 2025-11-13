# Smart Product Pricing Prediction 🚀

This repository contains the code for a machine learning solution to the **Smart Product Pricing Challenge**, which achieved a **rank of 629 out of 23,000 (Top 2.73%)**.

The goal of this project is to **predict the price of e-commerce products** using a **multimodal approach**, leveraging both **textual descriptions** and **product images**. This implementation combines advanced text-based feature engineering with deep-learning image embeddings to build a **high-performance ensemble model**.

---

## 🔥 Features

### 📝 Text-Based Features

This solution engineers a rich set of text-based attributes:

* **TF-IDF Representation**
  Unigram and bigram TF-IDF for core text analysis.

* **Item Pack Quantity (IPQ) Extraction**
  Extracts quantity values (e.g., *Pack of 12*, *6 Count*) using regex.

* **Keyword Detection**
  Binary indicators for value-defining keywords:

  * *Quality*: premium, organic, heavy-duty
  * *Bundling*: set, bundle, kit
  * *Condition*: refurbished, new, generic

* **Text Metadata**

  * Character count
  * Word count
  * Uppercase ratio

---

### 🖼️ Image-Based Features

* **EfficientNetB0 Embeddings**
  A pre-trained EfficientNetB0 model is used to extract dense numerical vectors from product images.

* **High-Dimensional Feature Extraction**
  Captures visual quality, complexity, material, and product structure.

---

## ⚙️ Methodology

### 1. **Data Pre-processing**

* Clean and engineer all text features (IPQ, Keywords, Metadata, TF-IDF).
* Process product images using EfficientNetB0 to generate embeddings.

### 2. **Feature Fusion**

Combine:

* Sparse + numerical text features
* Dense image embeddings
  → into a single multimodal feature matrix.

### 3. **Model Training**

Uses a two-model **ensemble**:

* **LightGBM (LGBMRegressor)**
* **HistGradientBoostingRegressor**

Target is transformed using:
`log(1 + price)`

### 4. **Persistence**

Trained models and preprocessed datasets are saved as `.pkl` files.

### 5. **Prediction**

Load saved files → run predictions → ensemble → convert back to original price scale.

---

## 🚀 Getting Started

### **Prerequisites**

* Python 3.7+
* Git

---

## 📥 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/amazon-price-prediction.git
cd amazon-price-prediction
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### Step 1 — Place Your Data

Create a folder named **Dataset** in the root:

```
Dataset/
 ├── train.csv
 └── test.csv
```

Add image folders inside it if required.

---

### Step 2 — Train the Model (run once)

```bash
python train_and_save_model.py
```

This will generate:

* `lightgbm_model.pkl`
* `histgb_model.pkl`
* `X_test_processed.pkl`
* etc.

---

### Step 3 — Generate Predictions

```bash
python load_and_predict.py
```

This will create:

```
submission.csv
```

---

## 📂 Project Structure

```
.
├── Dataset/
│   ├── train.csv
│   └── test.csv
├── .gitignore
├── train_and_save_model.py     # Pre-processing + training
├── load_and_predict.py         # Load models + prediction
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🎯 Advanced Techniques

### 🔹 Multimodal Feature Fusion

Combining text + image features gives the model a holistic understanding of the product—something neither modality could achieve alone.

### 🔹 Ensemble Modeling

Uses two strong gradient boosting models:

* LightGBM
* HistGradientBoosting

This improves robustness and accuracy.

### 🔹 Hyperparameter Tuning with Optuna

Optuna was used for extensive tuning of the LightGBM model.

A multi-hour search was conducted focusing on LGBM due to competition time constraints.

---

## 📄 License

This project is licensed under the **MIT License**.
