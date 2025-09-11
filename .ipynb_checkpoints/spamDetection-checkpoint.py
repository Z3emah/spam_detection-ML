import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json


model = joblib.load("best_pipeline.pkl")


st.markdown("<h2 style = 'text-align: left; color: orange;'> SMS Spam Detector </h2>", unsafe_allow_html = True)
st.markdown("<h3 style = 'text-align: left; color: gray;'> Enter a message below and click 'Classify' to check if it is a spam or not </h3>", unsafe_allow_html = True)
st.divider()


message = ""
uploaded_file = st.file_uploader("Upload a text file", type = ["txt", "text/plain"])


custom_threshold = 0.4


if uploaded_file is not None:
    message = uploaded_file.read().decode("utf-8")
    st.info("File uploaded successfully")
user_input = st.text_area("Or Enter Message Here", height = 150, value = message)


    
if st.button("Classify"):
    if user_input :
        message = user_input
        st.info("Analyzing message...")
        prediction = model.predict_proba([message])
     
        spam_confidence = prediction[0][1]
        ham_confidence = prediction[0][0]
        
        label = "🚫 Spam" if spam_confidence >= custom_threshold else "✅ Not Spam"
       
        
        if label == "🚫 Spam":
            st.error(label)
        else:
            st.success(label)

        with st.expander("Show Detailed Probabilities"):
            st.write(f"Confidence for Spam: {round(spam_confidence * 100)}%")
            st.write(f"Confidence for Not Spam: {round(ham_confidence * 100)}%")
            st.markdown("Classification is based on a custom threshold")
            
    else:
        st.warning("please enter a message to Classify")
        st.stop()


    
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)
    
metrics_df = pd.DataFrame(metrics).transpose()

metrics_df_filtered = metrics_df[["precision", "recall", "f1-score", "support"]]
metrics_df_filtered = metrics_df_filtered.rename(index= {"1" : "Spam", "0" : "Not Spam"})
metrics_df_filtered = metrics_df_filtered.loc[["Spam", "Not Spam"]]

with st.sidebar:
    st.subheader("Disclaimer")
    st.markdown("<h2 style = 'color: red;'>This is a machine learning demonstration for educational purposes.\n It is not a professional spam filter. Messages entered are not stored </h2>", unsafe_allow_html = True)
    
    st.header("Model Details")
    st.subheader("Loaded Model Pipeline")
    st.write(model.steps)
    
    st.markdown(f"**Default Custom Threshold:** {custom_threshold}")
    st.subheader("Model Settings")
    custom_threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.4, 0.05)
    
    st.subheader("Model Performance")
    st.markdown(f"****Overall Accuracy:**** {metrics["accuracy"]:.2f}")
    st.dataframe(metrics_df_filtered.round(2))
