🫀 **Heart Disease Prediction Using Machine Learning**

📄 **Project Overview**

This repository contains a comprehensive machine-learning project aimed at predicting the likelihood of heart disease in patients using clinical and lifestyle data. The project involves data preprocessing, exploratory data analysis (EDA), data visualizations, model development, and performance evaluation of multiple classifiers.

🎯** Objective**

To develop a machine learning model that predicts the presence of heart disease based on patient demographics, medical history, and test results.

📊 **Dataset**

Features: 13 attributes related to patient demographics, medical history, and diagnostic results.

Target Variable: Binary classification (1 = Presence of heart disease, 0 = No heart disease).

🛠️ **Methodology**

This project is implemented in Python using Jupyter Notebook and follows a systematic machine-learning pipeline:

**1. Data Preprocessing**

Data Inspection: Checked dataset structure, statistical summaries, and for duplicates.

Handling Missing Values: Verified and handled any null or missing data.

Feature Scaling: Standardized continuous features using StandardScaler to normalize data for improved model performance.

Feature Selection:

Pearson Correlation Analysis for continuous features.

Chi-Squared Test for categorical variables to identify the most informative features.

**2. Exploratory Data Analysis (EDA)**

Distribution of Target Variable: Visualized the balance of heart disease cases.

Correlation Heatmap: Identified key features strongly correlated with heart disease.

Histograms & Pairplots: Explored feature distributions and relationships.

Feature Importance: Highlighted the most influential features for prediction.

**3. Model Development & Evaluation
**
Train-Test Split: Dataset split into 80% training and 20% testing.

Three machine-learning models were trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

**4. Performance Metrics**

Models were evaluated using the following metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

**5. Model Comparison**

**6. Visualizations**

Bar Plots: Compared model accuracy, precision, recall, and F1-score.

Confusion Matrix: Visualized classification outcomes (true/false positives/negatives).

ROC Curve: Illustrated true positive and false positive rates.

📚 **Technologies Used**

Python

Pandas, NumPy (Data Manipulation)

Matplotlib, Seaborn (Visualizations)

Scikit-learn (Model Development & Evaluation)

Jupyter Notebook

🚀 **Future Work**

Explore advanced models (XGBoost, LightGBM).

Deploy the best model using Flask or FastAPI.

🤝 **Contributions**

Contributions are welcome! Feel free to fork this repository and submit pull requests.

📧 **Contact**

If you have any questions or suggestions, feel free to reach out via LinkedIn or open an issue.

⭐ If you found this project useful, give it a star and share it with others!



