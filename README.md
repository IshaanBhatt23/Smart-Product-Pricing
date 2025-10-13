# Smart Product Pricing Prediction 🚀

This repository contains the code for a machine learning solution to the "Smart Product Pricing Challenge." The project's goal is to predict the price of e-commerce products using only their textual descriptions and product images. This implementation focuses on building a robust model using advanced text-based feature engineering.

-----

## Features

This solution engineers a rich set of features from the raw product text to capture signals related to pricing:

  * **TF-IDF Representation:** Core text analysis using Term Frequency-Inverse Document Frequency for both unigrams and bigrams.
  * **Item Pack Quantity (IPQ) Extraction:** Uses regular expressions to intelligently parse and extract crucial quantity information (e.g., "Pack of 12", "6 Count") into a numerical feature.
  * **Keyword Detection:** Creates binary features for the presence of specific value-indicating keywords related to:
      * **Quality:** `premium`, `organic`, `heavy-duty`, etc.
      * **Bundling:** `set`, `bundle`, `kit`, etc.
      * **Condition:** `refurbished`, `new`, `generic`, etc.
  * **Text Metadata:** Simple yet effective features like character count, word count, and the ratio of uppercase letters.

-----

## Methodology ⚙️

The solution follows a multi-stage machine learning pipeline:

1.  **Data Pre-processing:** Text is cleaned, and all features (IPQ, Keywords, Metadata, TF-IDF) are engineered.
2.  **Feature Fusion:** The various numerical and sparse features are combined into a single feature matrix.
3.  **Model Training:** A **LightGBM (LGBMRegressor)** model is trained on the fused feature set. To handle the skewed price distribution, the model is trained to predict `log(1 + price)`.
4.  **Persistence:** The trained model is saved to a `.pkl` file to separate the training and prediction workflows.
5.  **Prediction:** The saved model is loaded to generate predictions on the test set, which are then converted back to the original price scale.

-----

## Getting Started

Follow these instructions to set up the project and run the code on your local machine.

### Prerequisites

  * Python 3.7+
  * Git

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/amazon-price-prediction.git
    cd amazon-price-prediction
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate
    ```

3.  **Create a `requirements.txt` file:**
    Before installing, it's best practice to freeze your current working libraries into a file.

    ```bash
    pip freeze > requirements.txt
    ```

    *This creates a `requirements.txt` file. You should commit this file to your repository.*

4.  **Install the required libraries:**
    *(If you are setting this up on a new machine, you would use this command)*

    ```bash
    pip install -r requirements.txt
    ```

### Usage

The workflow is split into two main steps: training the model and generating predictions.

1.  **Place your data:** Create a folder named `Dataset` in the root directory and place `train.csv` and `test.csv` inside it.

2.  **Train the model (Run once):**
    Execute the training script. This will pre-process the data, train the model with early stopping, and save the model and processed data to `.pkl` files.

    ```bash
    python train_and_save_model.py
    ```

    *This will create `lightgbm_model.pkl`, `X_test_processed.pkl`, and `test_df.pkl`.*

3.  **Generate Predictions (Run anytime):**
    Execute the prediction script. This will load the saved files and generate the final submission file in seconds.

    ```bash
    python load_and_predict.py
    ```

    *This will create `submission.csv` in the root folder.*

-----

## Project Structure

```
.
├── Dataset/
│   ├── train.csv
│   └── test.csv
├── .gitignore
├── train_and_save_model.py     # Script for pre-processing and training
├── load_and_predict.py         # Script for loading model and predicting
├── requirements.txt            # Project dependencies
└── README.md
```

-----

---

## Advanced Techniques Explored

To maximize the model's performance, several state-of-the-art techniques were implemented during the development process.

### Hyperparameter Tuning with Optuna ⚙️

Finding the optimal settings (hyperparameters) for a machine learning model is crucial for achieving peak performance. Instead of manual guesswork, this project utilized **Optuna**, a powerful automated tuning framework.

* An extensive, multi-hour search was conducted, running 30 distinct trials.
* In each trial, Optuna intelligently selected a new combination of parameters, trained a LightGBM model, and evaluated its performance.
* The final model was then trained using the single best set of hyperparameters discovered by this automated process, ensuring it was tuned for maximum accuracy on this specific dataset.

### Model USed

* A powerful top-tier gradient boosting model was explored: **LightGBM**
* The model was trained independently on the same rich, multimodal feature set.

-----

## License

This project is licensed under the MIT License.
