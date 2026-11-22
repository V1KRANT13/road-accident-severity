# 🚗 Road Accident Severity Prediction

A Machine Learning project that predicts the severity of road accidents using accident-related features.

The project uses multiple classification algorithms and selects the best-performing model based on Macro F1 Score.

## 🌐 Live Application

**Streamlit App:** [Click here to access the application](YOUR_STREAMLIT_APP_URL)

Upload the accident dataset (`RTA Dataset.csv`) through the application to train the models and predict accident severity.

## 📌 Project Overview

The objective is to predict accident severity into three categories:

- Slight Injury
- Serious Injury
- Fatal Injury

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib & Seaborn
- Scikit-learn
- Streamlit

## 🤖 Machine Learning Models

The following algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

**Evaluation Metrics:**
- Accuracy
- Macro F1 Score
- Precision
- Recall
- Confusion Matrix

Macro F1 Score is used as the primary metric because the target classes may be imbalanced.

## 📂 Project Structure

```text
Road-Accident-Severity/
│
├── app.py
├── requirements.txt
├── README.md
└── road_accident_severity_final.ipynb
```

## 💻 Run Locally

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd road-accident-severity
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

To deploy your own instance:

1. Push the project files to GitHub.
2. Visit https://share.streamlit.io/
3. Connect your GitHub account.
4. Select this repository.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

## 📊 Project Highlights

- End-to-end Machine Learning workflow.
- Data preprocessing and feature engineering.
- Comparison of multiple classification algorithms.
- Model evaluation using appropriate classification metrics.
- Interactive Streamlit web application.
- Online deployment using Streamlit Community Cloud.

---

**Developed as a Machine Learning project.**
