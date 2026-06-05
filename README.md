# Telco Customer Churn Predictor

An end-to-end machine learning project that predicts whether a telecom customer will cancel their subscription, deployed as an interactive web app.

🔗 **[Live App](https://customer-churn-prediction-ycsvjm4qh4sfjzludzs3cx.streamlit.app)** | 📓 **[Notebook](https://github.com/michelle-lahde/customer-churn-prediction/telco-customer-churn-/customer-churn-prediction.ipynb)**

---

## Overview

Customer churn is one of the biggest challenges in the telecom industry, and losing a customer costs a lot more than retaining one. This project uses real telecom data to build a machine learning model that predicts which customers are at risk of leaving; giving businesses important insights and the chance to intervene early.

---

## Key Findings

- Customers on **month-to-month contracts** churn at nearly 3x the rate of those on two-year contracts
- **New customers** (under 10 months tenure) are at the highest risk of churning
- Customers paying **higher monthly charges ($60–$120)** churn more than those on lower plans
- Customers paying around **$20/month** almost never churn

---

## Project Structure

```
telco-customer-churn/
│
├── customer-churn-prediction.ipynb   # Full analysis notebook
├── app.py                            # Streamlit web app
├── telco_churn.csv                   # Dataset
├── requirements.txt                  # Dependencies
└── README.md
```

---

## How It Works

### 1. Exploratory Data Analysis / EDA
Explored 7,043 customer records across 21 features including contract type, tenure, monthly charges, and internet service type. Visualized churn distribution and key relationships between features.

### 2. Data Preparation
- Removed irrelevant columns (customer ID)
- Converted text columns to numeric using one-hot encoding
- Handled missing values in TotalCharges
- Split data into 80% training / 20% testing

### 3. Model Building
Trained a **Logistic Regression** model using scikit-learn.

| Metric | Score |
|--------|-------|
| Accuracy | 79% |
| Precision (stayed) | 83% |
| Precision (churned) | 62% |

### 4. Deployment
Built and deployed an interactive web app using Streamlit where users can input customer details and get an instant churn prediction with probability score.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
| Scikit-learn | Machine learning |
| Streamlit | Web app deployment |

---

## Dataset

The dataset is the **IBM Telco Customer Churn** dataset, publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains information about 7,043 telecom customers including demographics, services subscribed, and whether they churned.

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/michelle-lahde/customer-churn-prediction/telco-customer-churn-

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
