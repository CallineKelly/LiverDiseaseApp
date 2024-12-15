import streamlit as st # build interactive web apps
import pandas as pd #data manipulation & analysis
from imblearn.over_sampling import SMOTE
from collections import Counter # counting element & check class distribution
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,mean_squared_error, classification_report
import matplotlib.pyplot as plt #plotting graphs
import numpy as np #provides operation on arrays & numerical data

# load clean dataset
data = pd.read_csv("https://raw.githubusercontent.com/CallineKelly/LiverDiseaseApp/18e9277e249c8fc91be4b78edaaf60bea76f139a/Indian%20Liver%20Patient%20Dataset%20(ILPD)-%20clean.csv")

# display the first few rows to check the structure
print("Dataset Preview:\n", data.head())

# Seperate the target and features
X = data.drop('Selector', axis=1)
y = data['Selector']

# Before up-sampling, check class distribution
# Counter(y), count no. of samples in each class of target
print(f"Ori class distribution: {Counter(y)}")

# Define the oversampler by SMOTE
smote = SMOTE(random_state=42)

# Apply the oversampler for minor class
X_resampled, y_resampled = smote.fit_resample(X,y)

#After up-sampling, check the class distribution
print(f"Resemple class distribution: {Counter(y_resampled)}")

# Perform 5-fold cross validation
#split dataset into 5 folds, ensures data is shuffled before splitting, random state = reproducibility of splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Training the MLP model on the resampled data
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

accuracy_scores = []
mse_scores = []

# convert resampled data into arrays
X_resampled_np = np.array(X_resampled)
y_resampled_np = np.array(y_resampled)

# kf.split generate training & testing for each fold
for train_index, test_index in kf.split(X_resampled_np):
    X_train,X_test = X_resampled_np[train_index], X_resampled_np[test_index]
    y_train, y_test = y_resampled_np[train_index], y_resampled_np[test_index]

    #Train and predict
    mlp_model.fit(X_train, y_train)
    y_pred = (mlp_model.predict(X_test))

    # Evaluate model on the test data
    accuracy = mlp_model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    accuracy_scores.append(accuracy)
    mse_scores.append(mse)

print("\nClassification Report:\n",classification_report(y_test, y_pred))
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

print(f"Accuracy Scores: {accuracy_scores}")
print(f"MSE Scores: {mse_scores}")

# Visualize Accuracy vs MSE
folds = range(1, len(accuracy_scores) + 1)
plt.figure(figsize=(10, 5))

plt.plot(folds, accuracy_scores, label="Accuracy", marker="o", linestyle="-", color="blue")
plt.plot(folds, mse_scores, label="MSE", marker="o", linestyle="--", color="orange")

plt.title("Relationship Between Accuracy and MSE Across Folds")
plt.xlabel("Fold")
plt.ylabel("Metrics")
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.show()

# Streamlit App
st.title("Liver Disease Prediction Control Panel")

# 1. User Input for 10 Features
st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=40)
gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
tb = st.sidebar.number_input("Total Bilirubin", min_value=0.0, max_value=75.0, value=1.0)
db = st.sidebar.number_input("Direct Bilirubin", min_value=0.0, max_value=19.0, value=0.1)
alkphos = st.sidebar.number_input("Alkaline Phosphatase", min_value=63.0, max_value=2110.0, value=200.0)
sgpt = st.sidebar.number_input("Alamine Aminotransferase", min_value=10.0, max_value=2000.0, value=50.0)
sgot = st.sidebar.number_input("Aspartate Aminotransferase", min_value=10.0, max_value=4900.0, value=60.0)
tp = st.sidebar.number_input("Total Protein", min_value=2.7, max_value=9.6, value=6.8)
alb = st.sidebar.number_input("Albumin", min_value=0.9, max_value=5.5, value=3.5)
ag_ratio = st.sidebar.number_input("A/G Ratio", min_value=0.0, max_value=2.8, value=1.0)

# Create a DataFrame for prediction
user_input = pd.DataFrame([[age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]],
                          columns=['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio'])

# 2. Predict Output
if st.sidebar.button("Predict"):
    prediction = mlp_model.predict(user_input)
    result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease Detected"
    st.subheader("Prediction Output")
    st.write(result)

# 3. Classification Report
st.subheader("Classification Report")
st.text(f"Accuracy: {accuracy:.4f}")
st.text(f"F1-Score: {f1:.4f}")
st.text(f"Precision: {precision:.4f}")
st.text(f"Recall: {recall:.4f}")
st.text(f"Mean Squared Error (MSE): {mse:.4f}")

# 4. Visualize Graph
st.subheader("Accuracy vs MSE Across Folds")

fig, ax = plt.subplots(figsize=(10, 5))
folds = range(1, len(accuracy_scores) + 1)

ax.plot(folds, accuracy_scores, label="Accuracy", marker="o", linestyle="-", color="blue")
ax.plot(folds, mse_scores, label="MSE", marker="o", linestyle="--", color="orange")

ax.set_title("Relationship Between Accuracy and MSE Across Folds")
ax.set_xlabel("Fold")
ax.set_ylabel("Metrics")
ax.set_xticks(folds)
ax.legend()
ax.grid(True)

st.pyplot(fig)

# run in terminal 'streamlit run liver_disease_app.py'
