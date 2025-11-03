## 🌟 Project Overview 
This project demonstrates a complete end-to-end Machine Learning pipeline for customer segmentation and classification. It's designed to help businesses gain deep insight into customer behavior to enable personalized and highly effective marketing strategies.

The pipeline involves two core ML phases:
 * Unsupervised Segmentation: Uses K-Means clustering to group customers based on demographics and spending habits.
 * Supervised Classification: Trains a classifier (specifically XGBoost/Random Forest) using the cluster labels to predict which segment a new customer belongs to.
Finally, the trained model is deployed via a Streamlit web application for real-time prediction.

## 🎯 Business Objectives 
Businesses often have customer data but lack clear, actionable insights into customer types. This project addresses that by aiming to:
 * Identify distinct customer groups based on their demographic and spending behavior.
 * Predict which segment a new, unseen customer belongs to with high accuracy.
 * Enable personalized marketing strategies and better, data-driven decision-making.

## 🚀 Streamlit App Features & Demo
The application provides a clean, interactive interface with real-time prediction output. 
 * Interactive Input: Users can enter Gender, Age, Annual Income ($\text{k}$), and Spending Score ($\text{1–100}$) via sliders and dropdowns.
 * Prediction Tab: View the predicted customer segment in real-time.Clean UI:
 * Modern design featuring tabs and a sidebar for a clean user experience.
Key Libraries Used
| Development | Python 3.9+
| ML/Data Science | Scikit-learn
| App Framework | Streamlit
| Model Persistence | Joblib


## 💻 Getting Started: Installation and Setup 
Prerequisites
 * Python 3.9+1.
1. Clone the Repository
   git clone [https://github.com/Alexmarco-gif/Customer-Segmentation-ml-app.git](https://github.com/Alexmarco-gif/Customer-Segmentation-ml-app.git)
   cd customer-segmentation-ml-app
2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies:
     [Create the environment]:
        python -m venv venv
     [Activate the environment (macOS/Linux)]:
        source venv/bin/activate
    [Activate the environment (Windows)]:
        venv\Scripts\activate
3. Install Dependencies
All required dependencies are listed in the requirements.txt file.
  pip install -r requirements.txt
4. Run the Streamlit App
Navigate to the App/ directory and run the application:
cd App/
streamlit run app.py
Your web browser should automatically open the Streamlit application at http://localhost:8501.

## 👩‍💻 Code Process: How the ML Pipeline Works 
The project is structured into modular code in Src and documented in the Notebooks.
1. Data Collection & Preprocessing (01_data_preprocessing.ipynb)
   * Dataset: Uses the Mall Customer Segmentation Dataset (Kaggle).
   * Data Preparation: Handles missing values and encodes categorical features (e.g., Gender $\rightarrow$ 0/1).
   * Scaling: Numeric features are scaled using StandardScaler.
   * Feature Engineering: Created behavior-based features:
     $Spending\_Efficiency = \frac{Spending\_Score}{Annual\_Income + 1}$$Income\_Spend\_Interaction = Annual\_Income \times Spending\_Score$
2. Segmentation (Unsupervised) (02_clustering.ipynb)
   * Model: K-Means clustering is applied to group customers.
   * Optimization: The optimal number of clusters ($k$) is determined using the Elbow Method and Silhouette Score.
   * Labeling: Cluster labels are used as the target variable ($y$) for the subsequent classification step.
   * Persistence: The trained KMeans model is saved as kmeans.pkl.
3. Classification (Supervised) (03_classification.ipynb)
   * Models: Random Forest and XGBoost are used and evaluated.
   * Evaluation: XGBoost achieved the best overall performance.
     Model | Accuracy | F1-Score
     Random Forest | 0.90 | 0.89
     | XGBoost | 0.94 ✅ | 0.93 ✅
   * Persistence: The final classification model is saved as classifier.pkl using joblib.
4. Model Deployment (App/app.py)
 * Framework: Streamlit is used for the web interface.
 * Prediction: The app loads the saved models and uses user input to generate real-time customer segment predictions.



## 🧰 Tools & Libraries 
| Stage | Libraries Used
| Data Handling | Pandas, NumPy
| Visualization | Matplotlib, Seaborn, Plotly
| ML Models |Scikit-learn, XGBoostModel 
| Storage | Joblib
| App Deployment | Streamlit 
| Environment | Python 3.9+

## 📁 Project Structure
customer_segmentation_classification/
├── App/
│   └── app.py                      # Streamlit web app for model deployment
├── Data/
│   ├── raw/                        # Raw dataset
│   └── processed/                  # Cleaned & transformed data
├── Model/                          # Stores saved models and scaler
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── classifier.pkl
├── Notebooks/                      # EDA and ML training scripts
├── Src/                            # Modular Python functions
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
