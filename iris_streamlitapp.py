import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset from a CSV file
df = pd.read_csv('iris.csv')

# Rename the 'target' column to 'species'
df = df.rename(columns={'target': 'species'})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'].values, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_pca, y_train)

# Make predictions
y_pred = model.predict(X_test_pca)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Create the Streamlit app
st.title("Iris Dataset App")
st.write("This app allows you to visualize the Iris dataset and make predictions based on the input features.")

# Add a dropdown menu for selecting the number of components
num_components = st.slider("Number of components", 1, 4, 2)

# Perform PCA with the selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(df.drop('species', axis=1))

# Add a scatter plot of the data
fig, ax = plt.subplots(figsize=(8, 8))
pd.plotting.scatter_matrix(df.drop('species', axis=1), ax=ax, alpha=0.7)
st.pyplot(fig)

# Add a text input for entering the input features
input_features = st.text_input("Enter the input features (separated by commas):")

# Convert the input features to a list
if input_features:
    input_features = [float(x) for x in input_features.split(',')]
else:
    input_features = []

# Scale the input features
if input_features:
    input_features = scaler.transform([input_features])

# Make a prediction
if input_features:
    prediction = model.predict(pca.transform(input_features))[0]
else:
    prediction = None

# Display the prediction
st.write(f"Prediction: {prediction}")