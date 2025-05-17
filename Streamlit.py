import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="MediAware", layout="wide")

# Load model and scaler
model = load_model("final_health_recommender_model.h5")
scaler = joblib.load("scaler.pkl")

# Load data
try:
    df = pd.read_csv('updated_family_health_dataset.csv')
except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure the CSV file is in the correct directory.")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
        .block-container { padding: 2rem; background-color: #fffbde; }
        h1, h2, h3, h4, .stMetric label, .stMetric div {
            color: #096B68; text-align: center;
        }
        .metric-center {
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 28px;
            color: #096B68;
            font-weight: bold;
        }
        .small-icon { font-size: 18px; }
        .patient-table {
            font-size: 14px;
            color: #096B68;
            margin: auto;
            width: 45%;
            border-collapse: collapse;
        }
        .patient-table th {
            text-align: left;
            padding: 6px;
            font-size: 16px;
            background-color: #fffbde;
        }
        .patient-table td {
            padding: 6px;
            font-weight: bold;
        }
        .risk-list {
            display: flex;
            justify-content: center;
            padding: 0;
            list-style-type: none;
        }
        .risk-list li {
            margin: 0 10px;
            font-size: 16px;
            color: #096B68;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='color:#096B68;'>MediAware</h1>
<h4 style='color:#096B68;'>Analytical System for Predicting Diseases for Families</h4>
""", unsafe_allow_html=True)

# -------- 👨‍⚕️ Doctor Input Section --------
st.markdown("<h2 style='color:#096B68;'>🔍 Patient & Family Analysis</h2>", unsafe_allow_html=True)

with st.form("search_form"):
    pat_id = st.number_input("Enter Patient ID (Person_ID):", step=1, min_value=0, key="input_id")
    submitted = st.form_submit_button("Search")

if submitted and pat_id:
    try:
        patient_row = df[df['Person_ID'] == pat_id].iloc[0]
        family_id = patient_row['Family_ID']
        family_df = df[df['Family_ID'] == family_id].copy()

        st.markdown("<h3 style='color:#096B68;'>🧍General Patient Info</h3>", unsafe_allow_html=True)

        st.markdown(f"""
        <table class="patient-table">
          <tr><th>🧍 Age</th><td>{int(patient_row['Age'])}</td></tr>
          <tr><th>👤 Gender</th><td>{patient_row['Gender']}</td></tr>
          <tr><th>⚖️ BMI</th><td>{round(patient_row['bmi'], 1)}</td></tr>
          <tr><th>🩸 Diabetes</th><td>{'Yes' if patient_row['Has_Diabetes'] else 'No'}</td></tr>
          <tr><th>💓 Hypertension</th><td>{'Yes' if patient_row['Has_Hypertension'] else 'No'}</td></tr>
          <tr><th>❤️ Heart Disease</th><td>{'Yes' if patient_row['Has_Heart_Disease'] else 'No'}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='color:#096B68;'>👨‍👩‍👧‍👦 Family Full Record</h3>", unsafe_allow_html=True)
        st.dataframe(family_df)

        try:
            gender_val = 0 if patient_row['Gender'] == 'Female' else 1
            input_data = np.array([[patient_row['Age'], patient_row['bmi'], gender_val,
                                    patient_row['Has_Diabetes'], patient_row['Has_Hypertension'], patient_row['Has_Heart_Disease']]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            predicted_class = np.argmax(prediction)
            label_map = ["✅ Safe", "⚠️ Moderate Concern", "🚨 High Concern"]
            st.markdown("<h3 style='color:#096B68;'>🧠 AI Suggestions</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#096B68;'>Prediction: {label_map[predicted_class]}</h4>", unsafe_allow_html=True)
            st.markdown("<ul class='risk-list'>" + "".join([
                f"<li>{label}: <b>{round(prob, 2)}</b></li>" for label, prob in zip(label_map, prediction)
            ]) + "</ul>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.stop()

        # ---- Family Summary
        st.markdown("<h3 style='color:#096B68;'>👨‍👩‍👧‍👦 Family Summary</h3>", unsafe_allow_html=True)

        def safe_predict(row):
            try:
                gender_val = 0 if row['Gender'] == 'Female' else 1
                input_data = np.array([[row['Age'], row['bmi'], gender_val,
                                        row['Has_Diabetes'], row['Has_Hypertension'], row['Has_Heart_Disease']]])
                input_scaled = scaler.transform(input_data)
                return model.predict(input_scaled)[0].tolist()
            except Exception as e:
                st.error(f"❌ Prediction error for Person_ID {row['Person_ID']}: {e}")
                return [0, 0, 0]

        family_df['Prediction_Probs'] = family_df.apply(safe_predict, axis=1)
        family_df['Final_Recommendation'] = family_df.apply(lambda row: ["✅ Safe", "⚠️ Moderate Concern", "🚨 High Concern"][np.argmax(row['Prediction_Probs'])], axis=1)

        rec_chart = family_df['Final_Recommendation'].value_counts().reset_index()
        rec_chart.columns = ['Risk Level', 'Count']

        fig_recs = px.pie(
            rec_chart,
            names='Risk Level',
            values='Count',
            title="Family AI Risk Classification Summary",
            color_discrete_sequence=["#096B68", "#90D1CA", "#129990"]
        )
        fig_recs.update_traces(
            textinfo='label+percent',
            pull=[0.05] * len(rec_chart),
            marker=dict(line=dict(color='#fffbde', width=2))
        )
        fig_recs.update_layout(
            height=400,
            width=600,
            title_font_size=18,
            legend_font_size=12,
            margin=dict(t=40, b=40, l=40, r=40),
            paper_bgcolor='#fffbde'
        )
        st.plotly_chart(fig_recs)

    except IndexError:
        st.error("Patient ID not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
