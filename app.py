import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.title("Zoo Animal Classifier")

#Load Data
@st.cache_data
def load_data():
    return pd.read_csv("zoo.csv")

df = load_data()

#Data Cleaning
def clean_values(val):
    if isinstance(val, str) and val.startswith("b'"):
        val = val[2:-1]
    if val == 'true': return 1
    if val == 'false': return 0
    return val

df_cleaned = df.map(clean_values)

#Model Training
X = df_cleaned.drop(['animal', 'type'], axis=1)
y = df_cleaned['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
model = RandomForestClassifier(n_estimators=75, random_state=42)
model.fit(X_train, y_train)

st.subheader("Model Performance")
y_pred = model.predict(X_test)

# Calculate 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

st.write(f"Model Accuracy: **{accuracy:.2f}**")
st.write(f"Precision: **{precision:.2f}**")
st.write(f"Recall: **{recall:.2f}**")


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
    if actual_type==predicted_type:
        st.balloons()
        st.toast("Prediction Success ")
    else:
        st.snow()
        st.toast("Prediction Fail ")

#addon
st.subheader("Predict Your Own Animal")

# We create inputs based on the columns of your X dataset (excluding 'animal')
# Note: Ensure these features match exactly what your model was trained on
features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 
            'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 
            'domestic', 'catsize']

user_input = {}
cols = st.columns(4) # Create a 4-column grid for the checkboxes

for i, feature in enumerate(features):
    # 'legs' is usually a number, others are 0/1 (boolean)
    if feature == 'legs':
        user_input[feature] = cols[i % 4].number_input("Legs (0-8)", min_value=0, max_value=8, value=2)
    else:
        user_input[feature] = cols[i % 4].checkbox(feature)

# Convert user input to a DataFrame for the model
input_df = pd.DataFrame([user_input])

# Prediction button
if st.button("Predict Animal Type"):
    prediction = model.predict(input_df)[0]
    st.success(f"The predicted type for your animal is: **Type {prediction}**")
