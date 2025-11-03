## 📘 Project Overview

This project demonstrates an end-to-end machine learning pipeline for customer segmentation and classification — helping businesses understand customer behavior and personalize marketing strategies.
We first segment customers using unsupervised learning (K-Means) and then train a supervised classifier (XGBoost/RandomForest) to predict which segment a new customer belongs to.
Finally, we deploy the trained classifier in a Streamlit web application for real-time prediction.

## 🧩 Business Objective
Businesses often have large amounts of customer data but limited insight into customer types.
This project aims to:

Identify distinct customer groups based on their demographic and spending behavior.
Predict which segment a new customer belongs to.
Enable personalized marketing strategies and better decision-making.

## ⚙️ Project Workflow

1️⃣ Problem Definition
Define the business goal and ML problem (unsupervised + supervised learning).

2️⃣ Data Collection
Use the Mall Customer Segmentation Dataset (Kaggle).
Columns include:
CustomerID
Gender
Age
Annual Income (k$)
Spending Score (1–100)

3️⃣ Data Preprocessing
Handle missing values
Encode categorical features (e.g., Gender → 0/1)
Scale numeric features using StandardScaler
Create new behavioral features:
Spending_Efficiency = Spending Score / (Annual Income + 1)
Income_Spend_Interaction = Annual Income × Spending Score

4️⃣ Customer Segmentation (Unsupervised)
Apply K-Means clustering to group customers.
Determine optimal k using Elbow Method and Silhouette Score.
Label clusters (e.g., low-spending, average, premium).

5️⃣ Customer Type Classification (Supervised)
Use cluster labels as target (y) to train a classifier.
Model used:
Random Forest


Evaluate accuracy, precision, recall, and F1-score.

Save best model using joblib.

6️⃣ Model Deployment (Streamlit App)
Deploy the classifier model via Streamlit:
User enters Gender, Age, Income, and Spending Score.
The app predicts their customer segment.
Tabs and sidebar make the UI clean and interactive.

## 🗂 Project Structure
customer_segmentation_classification/
│
├── App/
│   └── app.py                       # Streamlit web app for model deployment
│
├── Data/
│   ├── raw/                         # Raw dataset (Mall Customers)
│   └── processed/                   # Cleaned & transformed data
│
├── Model/
│   ├── scaler.pkl                   # StandardScaler for input normalization
│   ├── kmeans.pkl                   # KMeans clustering model
│   ├── classifier.pkl               # Final classification model (XGBoost/RandomForest)
│   └── features.pkl                 # Saved feature names
│
├── Notebooks/
│   ├── 01_data_preprocessing.ipynb  # Data cleaning & feature engineering
│   ├── 02_clustering.ipynb          # K-Means clustering for segmentation
│   └── 03_classification.ipynb      # Supervised learning for classification
│
├── Src/
│   ├── data_preprocessing.py        # Data preprocessing functions
│   ├── clustering_model.py          # Unsupervised clustering code
│   ├── classification_model.py      # Supervised model training
│   └── utils.py                     # Helper functions and utilities
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Ignored files (venv, cache, etc.)



## 🧰 Tools & Libraries
| Stage              | Libraries Used              |
| ------------------ | --------------------------- |
| **Data Handling**  | Pandas, NumPy               |
| **Visualization**  | Matplotlib, Seaborn, Plotly |
| **ML Models**      | Scikit-learn, XGBoost       |
| **Model Storage**  | Joblib                      |
| **App Deployment** | Streamlit                   |
| **Environment**    | Python 3.9+                 |


## 🧠 Key Features
✅ Unsupervised + Supervised ML Pipeline
✅ Behavior-based Feature Engineering
✅ Model Evaluation & Optimization
✅ Interactive Streamlit App Deployment
✅ Clean Modular Code Structure


## 🎨 Streamlit App Features

Sidebar: “About the Project” section
Tabs Layout:

🧍 Customer Input – Enter new customer details

🔮 Prediction – View predicted segment

📊 Model Info – Understand model details

Clean UI Design: modern colors, rounded cards, hover effects

Real-time prediction output

## 📊 Model Evaluation Example
| Model               | Accuracy   | F1-Score   |
| ------------------- | ---------- | ---------- |
| Logistic Regression | 0.82       | 0.79       |
| Random Forest       | 0.90       | 0.89       |
| XGBoost             | **0.94 ✅** | **0.93 ✅** |

## 📦 Future Enhancements

* 🔁 Add real-time model monitoring (MLflow / Evidently)

* ☁️ Deploy app using Render, Railway, or Hugging Face Spaces

* 💾 Connect to database for storing customer data

* 📊 Add visualization of new customer’s position among clusters

![App Preview](./Customer_Segmentation.JPG)
