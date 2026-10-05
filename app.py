import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Bhutan Student Wellbeing ML Project", layout="wide")

# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------
@st.cache_data
def generate_synthetic_data():
    np.random.seed(42)
    n = 800
    
    districts = [
        "Thimphu", "Paro", "Punakha", "Chhukha", "Sarpang", 
        "Tashigang", "Wangdue Phodrang", "Samdrup Jongkhar"
    ]
    
    age = np.random.randint(18, 27, size=n)
    gender = np.random.choice(["Female", "Male", "Other"], size=n, p=[0.5, 0.48, 0.02])
    district = np.random.choice(districts, size=n)
    residence = np.random.choice(["Urban", "Rural"], size=n, p=[0.6, 0.4])
    
    sleep_hours = np.round(np.random.normal(6.5, 1.2, size=n), 1)
    sleep_hours = np.clip(sleep_hours, 3.0, 10.0)
    
    study_hours = np.round(np.random.normal(5.0, 1.8, size=n), 1)
    study_hours = np.clip(study_hours, 1.0, 12.0)
    
    physical_activity = np.random.randint(0, 8, size=n)
    stress_level = np.random.randint(1, 11, size=n)
    social_support = np.random.randint(1, 11, size=n)
    financial_pressure = np.random.randint(1, 11, size=n)
    academic_pressure = np.random.randint(1, 11, size=n)
    
    screen_time = np.round(np.random.normal(4.5, 1.5, size=n), 1)
    screen_time = np.clip(screen_time, 1.0, 10.0)
    
    awareness = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.3, 0.5, 0.2])
    
    # Synthetic target logic with noise
    score = (
        (stress_level * 0.3) + 
        (academic_pressure * 0.25) + 
        (financial_pressure * 0.2) - 
        (social_support * 0.25) - 
        (sleep_hours * 0.2) + 
        np.random.normal(0, 1, size=n)
    )
    
    support_need = np.where(score > 1.5, "Support recommended", "No immediate flag")
    
    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "district": district,
        "residence": residence,
        "sleep_hours": sleep_hours,
        "study_hours_per_day": study_hours,
        "physical_activity_days": physical_activity,
        "stress_level": stress_level,
        "social_support": social_support,
        "financial_pressure": financial_pressure,
        "academic_pressure": academic_pressure,
        "screen_time_hours": screen_time,
        "mental_health_awareness": awareness,
        "support_need": support_need
    })
    
    return df

df = generate_synthetic_data()

# ---------------------------------------------------------
# MODEL TRAINING PIPELINE
# ---------------------------------------------------------
@st.cache_resource
def train_pipeline(data):
    X = data.drop("support_need", axis=1)
    y = data["support_need"].map({"No immediate flag": 0, "Support recommended": 1})
    
    numeric_cols = [
        "age", "sleep_hours", "study_hours_per_day", 
        "physical_activity_days", "stress_level", "social_support", 
        "financial_pressure", "academic_pressure", "screen_time_hours"
    ]
    categorical_cols = ["gender", "district", "residence", "mental_health_awareness"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, categorical_cols)
    ])
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_pipeline(df)

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("Mental Health and Student Wellbeing in Bhutan")
st.subheader("A Basic Machine-Learning Application Using Streamlit")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Explore Data", "Prediction Demo", "Model Results"
])

# TAB 1: OVERVIEW
with tab1:
    st.markdown("""
    ### Project Overview
    This student project demonstrates a complete beginner-friendly machine-learning workflow using Python, scikit-learn, and Streamlit.
    
    > **Important:** This project uses synthetic data only. It does not contain information collected from real Bhutanese students and must not be used as a medical, psychological, screening, or diagnostic tool.
    
    #### Project Objective
    Can basic lifestyle, academic-pressure, and social-support variables predict a synthetically generated "support recommended" label?
    
    The purpose is to teach students how to:
    - Generate and explore a dataset
    - Prepare numeric and categorical variables
    - Divide data into training and testing sets
    - Train a classification model
    - Evaluate model performance
    - Build an interactive Streamlit interface
    - Discuss limitations and responsible AI use
    """)

# TAB 2: EXPLORE DATA
with tab2:
    st.markdown("### Dataset Exploration")
    st.dataframe(df.head(10))
    
    selected_col = st.selectbox(
        "Select variable to visualize:", 
        ["stress_level", "sleep_hours", "academic_pressure", "social_support", "district"]
    )
    
    fig, ax = plt.subplots(figsize=(8, 4))
    if df[selected_col].dtype == "object":
        sns.countplot(data=df, x=selected_col, hue="support_need", ax=ax)
        plt.xticks(rotation=45)
    else:
        sns.histplot(data=df, x=selected_col, hue="support_need", kde=True, ax=ax)
    
    st.pyplot(fig)

# TAB 3: PREDICTION DEMO
with tab3:
    st.markdown("### Interactive Prediction Demo")
    st.write("Enter a fictional profile to see the model prediction:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 18, 30, 21)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        district = st.selectbox("District", [
            "Thimphu", "Paro", "Punakha", "Chhukha", "Sarpang", 
            "Tashigang", "Wangdue Phodrang", "Samdrup Jongkhar"
        ])
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        
    with col2:
        sleep_hours = st.slider("Sleep Hours / Day", 3.0, 10.0, 7.0)
        study_hours = st.slider("Study Hours / Day", 1.0, 12.0, 5.0)
        physical_activity = st.slider("Physical Activity Days / Week", 0, 7, 3)
        screen_time = st.slider("Screen Time Hours / Day", 1.0, 10.0, 4.0)
        
    with col3:
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10, 5)
        financial_pressure = st.slider("Financial Pressure (1-10)", 1, 10, 5)
        social_support = st.slider("Social Support (1-10)", 1, 10, 5)
        awareness = st.selectbox("Mental Health Awareness", ["Low", "Medium", "High"])

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "district": district,
        "residence": residence,
        "sleep_hours": sleep_hours,
        "study_hours_per_day": study_hours,
        "physical_activity_days": physical_activity,
        "stress_level": stress_level,
        "social_support": social_support,
        "financial_pressure": financial_pressure,
        "academic_pressure": academic_pressure,
        "screen_time_hours": screen_time,
        "mental_health_awareness": awareness
    }])

    if st.button("Run Prediction"):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        
        if pred == 1:
            st.error(f"Result: **Support recommended** (Confidence: {proba:.2%})")
        else:
            st.success(f"Result: **No immediate flag** (Confidence: {1 - proba:.2%})")

# TAB 4: MODEL RESULTS
with tab4:
    st.markdown("### Model Evaluation Metrics")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    col2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    col3.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col4.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    col5.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
    
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                xticklabels=["No Flag", "Support"], 
                yticklabels=["No Flag", "Support"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)
