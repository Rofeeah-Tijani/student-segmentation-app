import streamlit as st
import pandas as pd
import pickle

# Load saved model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App title
st.title("Student Segmentation App")

st.write("Enter student information to predict the cluster")

# User inputs
age = st.number_input("Age", min_value=10, max_value=30)

studytime = st.number_input(
    "Study Time",
    min_value=1,
    max_value=10
)

failures = st.number_input(
    "Failures",
    min_value=0,
    max_value=10
)

absences = st.number_input(
    "Absences",
    min_value=0,
    max_value=100
)

G1 = st.number_input(
    "G1 Score",
    min_value=0,
    max_value=20
)

G2 = st.number_input(
    "G2 Score",
    min_value=0,
    max_value=20
)

G3 = st.number_input(
    "G3 Score",
    min_value=0,
    max_value=20
)

# Predict button
# Predict button
if st.button("Predict Cluster"):

    # Create dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences],
        'G1': [G1],
        'G2': [G2],
        'G3': [G3]
    })

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict cluster
    prediction = model.predict(scaled_data)

    # Display result
    if prediction[0] == 0:

    st.success("🎓 High Performing Student")

    st.info("""
    Recommendation:
    Continue maintaining strong study habits
    and academic consistency.
    """)

elif prediction[0] == 1:

    st.warning("📘 Average Performing Student")

    st.info("""
    Recommendation:
    Increased study consistency and reduced
    distractions may improve performance.
    """)

else:

    st.error("⚠️ Student Needs Academic Support")

    st.info("""
    Recommendation:
    Additional tutoring, academic counseling,
    and structured study planning may help.
    """)
