import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import joblib


st.set_page_config(page_title="Fraud Intelligence Center", layout="wide")

@st.cache_resource
def load_assets():
    with open("fraud_model.pkl", "rb") as f:
        model = joblib.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_assets()


def generate_live_data(n=10):
    
    data = pd.DataFrame(np.random.randn(n, len(feature_cols)), columns=feature_cols)
    data['Amount'] = np.random.uniform(10, 2000, n)
    
    if np.random.random() > 0.5:
        fraud_idx = np.random.randint(0, n)
        data.loc[fraud_idx, 'V17'] = np.random.uniform(-25.0, -15.0)
        data.loc[fraud_idx, 'V14'] = np.random.uniform(-18.0, -8.0)  
        data.loc[fraud_idx, 'V12'] = np.random.uniform(-15.0, -5.0)
        data.loc[fraud_idx, 'Amount'] = np.random.uniform(3000, 5000)
    
    return data


st.title("Global Bank: Fraud Monitoring Feed")
st.markdown("Internal Security Tool: Monitoring live bank transactions for unusual patterns.")


m1, m2 = st.columns(2)
m1.metric("System Status", "ACTIVE", delta="Secure")
m2.metric("Model Threshold", "0.20", delta="Optimized")


st.divider()


if st.button("Fetch New Transactions from Feed"):
    with st.spinner('Analyzing patterns...'):
        time.sleep(1) 
        
        
        live_feed = generate_live_data(15)
        
    
        X = scaler.transform(live_feed[feature_cols])
        probs = model.predict_proba(X)[:, 1]
        
        live_feed['Fraud_Probability'] = probs
        live_feed['Status'] = ["FLAG" if p > 0.2 else "CLEAR" for p in probs]

        
        st.subheader("Recent Activity")
        
        
        def highlight_fraud(row):
            return ['background-color: #AD2831' if row.Status == "FLAG" else '' for _ in row]

        styled_df = live_feed.style.apply(highlight_fraud, axis=1).format({"Amount": "${:.2f}", "Fraud_Probability": "{:.2%}"})
        st.dataframe(styled_df, use_container_width=True)

        
        flagged = live_feed[live_feed['Status'] == "FLAG"]
        if not flagged.empty:
            st.warning(f"Security Alert: {len(flagged)} suspicious transaction(s) intercepted.")
            
        
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("### Anomaly Signature")
                analysis_features = ['V10', 'V12', 'V14', 'V17']
                anomaly_means = flagged[analysis_features].mean()
                st.bar_chart(anomaly_means, color="#0078D4")
            with col_b:
                st.write("### Recommended Action")
                st.info(f"""
                - **Transaction ID:** {np.random.randint(1000, 9999)}
                - **Primary Trigger:** Outlier in V17 components
                - **Action:** Temporary Card Freeze & SMS Verification sent.
                """)
        else:
            st.success("No suspicious patterns detected in the last batch.")

else:
    st.info("Click the button above to start monitoring the live transaction stream.")

st.caption("(Demo system using simulated transaction data for model evaluation)")
