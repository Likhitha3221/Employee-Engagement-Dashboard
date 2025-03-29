import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Title and description
st.title("People Analytics Dashboard")
st.write("""
    Welcome to the interactive People Analytics dashboard! Upload your dataset, visualize trends, and make predictions using machine learning.
    Explore employee data to gain insights on attrition, job satisfaction, and more!
""")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Show the first few rows of the dataframe
    st.write("### Data Overview")
    st.dataframe(df.head())

    # Basic summary and information about the dataset
    st.write("### Dataset Summary")
    st.write(df.describe())
    st.write("### Column Data Types")
    st.write(df.dtypes)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Sidebar for selecting graph types
    st.sidebar.header("Select Graphs and Features")
    graph_type = st.sidebar.selectbox("Choose Graph Type", ['Histogram', 'Scatter Plot', 'Box Plot'])

    if graph_type == 'Scatter Plot':
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        scatter_fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(scatter_fig)

    elif graph_type == 'Histogram':
        column = st.sidebar.selectbox("Select Column for Histogram", df.columns)
        hist_fig = px.histogram(df, x=column, nbins=20, title=f"Distribution of {column}")
        st.plotly_chart(hist_fig)

    elif graph_type == 'Box Plot':
        column = st.sidebar.selectbox("Select Column for Box Plot", df.columns)
        box_fig = px.box(df, y=column, title=f"Box Plot of {column}")
        st.plotly_chart(box_fig)

    # Predictive Analytics: Logistic Regression Model
    st.write("### Predictive Analytics: Machine Learning Model")
    st.write("""
        Use the logistic regression model to predict outcomes such as employee attrition.
        Select the target column (e.g., 'AttritionYes') and analyze the model's performance.
    """)

    # Selecting target column
    target_column = st.sidebar.selectbox("Select Target Column for Prediction", df.columns)

    # Preprocessing: Encoding categorical variables and splitting data
    label_encoder = LabelEncoder()

    # Encoding non-numeric columns
    df_encoded = df.apply(lambda col: label_encoder.fit_transform(col) if col.dtypes == 'object' else col)

    # Features and target variable
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]

    # Train-test split (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Display model performance
    st.write("### Model Evaluation")
    st.text(classification_report(y_test, y_pred))

    # Feature importance (for models like Decision Trees, but here we show coefficients)
    st.write("### Model Feature Coefficients (Importance)")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.coef_[0]
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    st.write(feature_importance)

    # Interactive Filter to predict attrition (or other target column)
    st.write("### Predict Attrition (or other target)")
    input_data = {}

    for col in df.columns:
        if col != target_column:  # Skip the target column
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
            else:
                input_data[col] = st.slider(f"Select {col}", min_value=int(df[col].min()), max_value=int(df[col].max()), value=int(df[col].mean()))

    # Prepare input data for prediction
    input_df = pd.DataFrame(input_data, index=[0])
    input_df_encoded = input_df.apply(lambda col: label_encoder.fit_transform(col) if col.dtypes == 'object' else col)

    # Make a prediction for the input data
    prediction = model.predict(input_df_encoded)
    prediction_proba = model.predict_proba(input_df_encoded)

    # Display prediction results
    if prediction[0] == 1:
        st.write("The employee is likely to leave (Attrition: Yes).")
    else:
        st.write("The employee is likely to stay (Attrition: No).")

    st.write(f"Probability of Attrition: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Staying: {prediction_proba[0][0]:.2f}")

else:
    st.write("Please upload a CSV file to get started!")

