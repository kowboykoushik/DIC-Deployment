# app.py
import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
# Assuming 'df' is your original dataset
df = pd.read_csv("hiv.csv")

# Perform preprocessing
df_processed, df_unprocessed = preprocess(df)

# Title of the app
st.title("AIDS Mortality Prediction using Clinical Trial Data")

# Add a sidebar for user inputs
st.sidebar.header("User Input Options")

# User selects input option
input_option = st.sidebar.radio("Select Input Option", ("CSV File", "Single Record"))

# Display some information about the dataset
st.sidebar.subheader("Training Dataset Information")
st.sidebar.write(f"Shape of the dataset: {df.shape}")
st.sidebar.write(f"Number of classes: {len(df['target'].unique())}")

# Allow user to choose a model
selected_model = st.sidebar.selectbox("Select a Model", ["Linear Regression Model", "K Nearest Neighbors Model", "Support Vector Machine Model", "Decision Tree Classifier Model", "Random Forest Classifier Model", "XGBoost Model"])

# Define a dictionary mapping model names to their file paths
model_paths = {
    "Linear Regression Model": "lr_model.pkl",
    "K Nearest Neighbors Model": "knn_model.pkl",
    "Support Vector Machine Model": "svm_model.pkl",
    "Decision Tree Classifier Model": "dt_model.pkl",
    "Random Forest Classifier Model": "rf_model.pkl",
    "XGBoost Model": "xgb_model.pkl"
}

# Load the selected model
model_path = model_paths.get(selected_model)
if model_path is not None:
    loaded_model = joblib.load(model_path)
else:
    st.warning("Please select a model.")

st.header("User Input")
if input_option == "Single Record":
    # Add user input features for a single record using text fields and select boxes
    pid = st.text_input("Enter pid", "0")
    time = st.text_input("Enter time", "0")
    trt = st.selectbox("Select trt", ["ZDV only", "ZDV + ddI", "ZDV + Zal", "ddI only"])
    age = st.text_input("Enter age", "0")
    wtkg = st.text_input("Enter wtkg", "0")
    hemo = st.selectbox("Select hemo", ["no", "yes"])
    homo = st.selectbox("Select homo", ["no", "yes"])
    drugs = st.selectbox("Select drugs", ["no", "yes"])
    karnof = st.slider("Select karnof", 0, 100, 50)
    oprior = st.selectbox("Select oprior", ["no", "yes"])
    z30 = st.selectbox("Select z30", ["no", "yes"])
    zprior = st.selectbox("Select zprior", ["yes"])
    preanti = st.text_input("Enter preanti", "0")
    race = st.selectbox("Select race", ["white", "non-white"])
    gender = st.selectbox("Select gender", ["female", "male"])
    strat = st.text_input("Enter strat", "0")
    symptom = st.selectbox("Select symptom", ["asymp", "symp"])
    treat = st.selectbox("Select treat", ["ZDV only", "others"])
    offtrt = st.selectbox("Select offtrt", ["no", "yes"])
    cd40 = st.text_input("Enter cd40", "0")
    cd420 = st.text_input("Enter cd420", "0")
    cd80 = st.text_input("Enter cd80", "0")
    cd820 = st.text_input("Enter cd820", "0")
    str2 = st.selectbox("Select str2", ["naive", "experienced"])

    # Combine user inputs into a DataFrame
    user_inputs = pd.DataFrame({
        "pid": [pid],
        "time": [time],
        "trt": [trt],
        "age": [age],
        "wtkg": [wtkg],
        "hemo": [hemo],
        "homo": [homo],
        "drugs": [drugs],
        "karnof": [karnof],
        "oprior": [oprior],
        "z30": [z30],
        "zprior": [zprior],
        "preanti": [preanti],
        "race": [race],
        "gender": [gender],
        "strat": [strat],
        "symptom": [symptom],
        "treat": [treat],
        "offtrt": [offtrt],
        "cd40": [cd40],
        "cd420": [cd420],
        "cd80": [cd80],
        "cd820": [cd820],
        "str2": [str2],
    })

    # Display the user inputs
    st.subheader("User Input Features")
    st.write(user_inputs)

elif input_option == "CSV File":
    # Allow user to upload a CSV file
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        user_inputs = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.subheader("Uploaded CSV Data")
        st.write(user_inputs)

    else:
        st.warning("Please upload a CSV file.")

# Make predictions using the loaded model
if (input_option == "CSV File" and uploaded_file is not None): 
    # Perform preprocessing on the uploaded data
    preprocess_inputs, unpreprocess_inputs = preprocess(user_inputs)
    st.write("Preprocessed Input")
    st.write(preprocess_inputs)
    prediction = loaded_model.predict(preprocess_inputs)

    # Display predictions and user inputs in a table
    st.subheader("Model Prediction and User Inputs")
    st.write("Note: In case any column or field has wrong data then the record may not be predicted")
    st.write("Note: In case your dataset has ground truth in column target then it can be compared with predicted value under column Prediction") 
    result_df = unpreprocess_inputs
    result_df["Prediction"] = prediction
    predictor_mapping = {0: 'censoring', 1: 'failure'}
    # Map values in the 'Status' column
    result_df["Prediction"] = result_df["Prediction"].map(predictor_mapping)
    st.write(result_df)

    # Comparison Bar Chart
    if 'target' in user_inputs.columns:
        st.header("Comparison Bar Chart")
        fig_compare, ax_compare = plt.subplots(figsize=(10, 6))
        comparison_data = result_df[['target', 'Prediction']].value_counts().unstack()
        comparison_data.plot(kind='bar', stacked=True, ax=ax_compare)
        ax_compare.set_xlabel("Class")
        ax_compare.set_ylabel("Count")
        ax_compare.set_title("Actual vs Predicted Comparison")
        st.pyplot(fig_compare)

    def compute_distances(input_instance, dataset, metric='euclidean'):
        distances = np.linalg.norm(dataset - input_instance, axis=1, ord=2)
        return distances

    st.header("Distance Plot")
    #distances = compute_distances(preprocess_inputs.values, df_processed.values)
    all_distances = [compute_distances(instance, df_processed.drop('target', axis=1).values) for instance in preprocess_inputs.values]

    # Concatenate all distances into a single array
    distances = np.concatenate(all_distances)

    # Binning distances
    bins = np.arange(0, np.max(distances) + 1, 1)  # Adjust the bin size based on your data
    binned_counts, bin_edges = np.histogram(distances, bins=bins)
    
    # Display the pie chart
    st.header("Binned Distance Pie Chart")
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(binned_counts, labels=[f'{edge:.1f}-{edge_next:.1f}' for edge, edge_next in zip(bin_edges[:-1], bin_edges[1:])], autopct='%1.1f%%', startangle=140)
    ax_pie.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    ax_pie.set_title("Binned Distances Pie Chart")
    st.pyplot(fig_pie)
if input_option == "Single Record": 
    # Perform preprocessing on the uploaded data
    preprocess_inputs, unpreprocess_inputs = preprocess(user_inputs)
    st.write("Preprocessed Input")
    st.write(preprocess_inputs)
    prediction = loaded_model.predict(preprocess_inputs)

    # Display predictions and user inputs in a table
    st.subheader("Model Prediction and User Inputs")
    st.write("Note: In case any column or field has wrong data then the record may not be predicted")
    st.write("Note: In case your dataset has ground truth in column target then it can be compared with predicted value under column Prediction") 
    result_df = unpreprocess_inputs
    result_df["Prediction"] = prediction
    predictor_mapping = {0: 'censoring', 1: 'failure'}
    # Map values in the 'Status' column
    result_df["Prediction"] = result_df["Prediction"].map(predictor_mapping)
    st.write(result_df)

    def compute_distances(input_instance, dataset, metric='euclidean'):
        distances = np.linalg.norm(dataset - input_instance, axis=1, ord=2)
        return distances

    st.header("Distance Plot")
    distances = compute_distances(preprocess_inputs.values, df_processed.values)
    # Binning distances
    bins = np.arange(0, np.max(distances) + 1, 1)  # Adjust the bin size based on your data
    binned_counts, bin_edges = np.histogram(distances, bins=bins)
    
    # Display the pie chart
    st.header("Binned Distance Pie Chart")
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(binned_counts, labels=[f'{edge:.1f}-{edge_next:.1f}' for edge, edge_next in zip(bin_edges[:-1], bin_edges[1:])], autopct='%1.1f%%', startangle=140)
    ax_pie.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    ax_pie.set_title("Binned Distances Pie Chart")
    st.pyplot(fig_pie)
