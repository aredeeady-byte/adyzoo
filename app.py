import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.title("Zoo Animal Classifier")

# 1. Load Data
@st.cache_data
def load_data():
    # Ensure zoo.csv is in the same folder as app.py
    return pd.read_csv("zoo.csv")

df = load_data()

# 2. Data Cleaning
def clean_values(val):
    if isinstance(val, str) and val.startswith("b'"):
        val = val[2:-1]
    if val == 'true': return 1
    if val == 'false': return 0
    return val

df_cleaned = df.map(clean_values)

# 3. Model Training
X = df_cleaned.drop(['animal', 'type'], axis=1)
y = df_cleaned['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Streamlit UI
st.subheader("Model Performance")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy:.2f}**")

st.subheader("Test a Random Animal")
if st.button("Predict Random Animal"):
    sample = X_test.sample(1)
    idx = sample.index[0]
    
    animal_name = df_cleaned.loc[idx, 'animal']
    actual_type = y.loc[idx]
    predicted_type = model.predict(sample)[0]
    
    st.write(f"**Animal:** {animal_name}")
    st.write(f"**Actual Type:** {actual_type}")
    st.write(f"**Predicted Type:** {predicted_type}")
    st.balloons()