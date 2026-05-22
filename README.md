# Credit Card Fraud Monitoring Dashboard

An **interactive dashboard** that simulates a **real-time credit card transaction monitoring system** using **machine learning**.  
Detects potential fraudulent transactions with alerts and anomaly visualization.

## Key Features

- **Live Transaction Simulation**: Generates realistic transactions with occasional fraudulent patterns.
- **Random Forest Model**: ML model trained to detect fraud in credit card transactions.
- **Threshold Tuning**: Fraud probability threshold is adjustable to optimize detection.
- **Imbalanced Data Handling**: Used **SMOTE** to balance fraud vs non-fraud cases during training.
- **Fraud Alerts**: Flags high-risk transactions with recommended actions.
- **Anomaly Visualization**: Visual bar charts of key features for flagged transactions.
- Built with **Streamlit**, **Pandas**, **NumPy**, **Scikit-learn**, and **Joblib**.

## Demo

This dashboard uses **simulated transaction data** for demonstration purposes.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sakshirsharma022-debug/fraud-monitoring-dashboard.git
cd fraud monitoring dashboard
pip install -r requirements.txt
streamlit run app.py
