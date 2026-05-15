# student-segmentation-app
# 🎓 Student Segmentation App

## 📌 Project Overview

The Student Segmentation App is a Machine Learning web application developed to group students into different categories based on their academic performance and study related behavior.

This project was built using **KMeans Clustering**, an unsupervised machine learning algorithm that identifies hidden patterns within data by grouping similar observations together.

The application allows users to enter student related information such as:

- Age
- Study Time
- Number of Failures
- Absences
- G1 Score
- G2 Score
- G3 Score

and then predicts the cluster the student belongs to.

The project demonstrates the complete workflow of a real world Machine Learning deployment process, beginning from data preprocessing and model training to cloud deployment and web application development.

---

# 🎯 Project Objectives

The major goal of this project is to:

- Analyze student academic behavior
- Identify patterns among students
- Group students into meaningful clusters
- Build an interactive prediction system
- Deploy the model online for public access

This project also serves as a practical introduction to Machine Learning deployment using Streamlit.

---

# 🧠 Understanding The Machine Learning Concept Used

## What is Clustering?

Clustering is an unsupervised machine learning technique used to group similar data points together based on their characteristics.

Unlike supervised learning, clustering does not require predefined labels.

Instead, the algorithm automatically discovers patterns within the dataset.

In this project, students with similar academic and behavioral characteristics are grouped together into clusters.

---

## Why KMeans Clustering?

KMeans was selected because:

- It is simple and efficient
- It performs well for numerical data
- It is widely used for segmentation problems
- It helps identify hidden groups within datasets

The algorithm works by:

1. Selecting the number of clusters
2. Initializing cluster centers
3. Measuring distances between observations and cluster centers
4. Assigning observations to the nearest cluster
5. Updating cluster centers repeatedly until convergence

---

# 📊 Dataset Information

The dataset used for this project is the Student Performance Dataset.

Features selected for clustering include:

| Feature | Description |
|---|---|
| age | Student age |
| studytime | Weekly study time |
| failures | Number of past class failures |
| absences | Number of school absences |
| G1 | First period grade |
| G2 | Second period grade |
| G3 | Final grade |

These variables were selected because they provide meaningful academic and behavioral information about students.

---

# ⚙️ Machine Learning Workflow

The project followed the complete Machine Learning pipeline below.

## 1. Data Selection

Relevant features related to student academic performance were selected from the dataset.

```python
dF = df[['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]

```

## 2. Data Standardization

The dataset was standardized using StandardScaler.

This step was necessary because the variables exist on different scales.

For example:

Age values are smaller
Absence values may be much larger
Grades exist on another scale entirely

Without standardization, variables with larger scales may dominate the clustering process.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(dF)

----

## 3. Model Training

KMeans clustering was then applied to the standardized dataset.

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)

model.fit(X_scaled)

The model grouped students into 3 different clusters based on similarities in their academic behavior and performance.


## 4. Cluster Prediction

After training the model, cluster labels were predicted for students.

clusters = model.predict(X_scaled)

## 5. Model Persistence

The trained model and scaler were saved using Pickle.

This allows the trained objects to be reused later without retraining the model every time the application starts.

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
🌐 Web Application Development

The user interface was developed using Streamlit.

The application accepts user input interactively and predicts the student cluster in real time.

Example functionalities include:

Interactive number inputs
Real time predictions
Cluster interpretation
User friendly dashboard
☁️ Deployment Process

The application was deployed online using Streamlit Community Cloud.

Deployment workflow:

Project files uploaded to GitHub
Streamlit Cloud connected to GitHub repository
Dependencies installed using requirements.txt
Application deployed publicly online

This deployment process transformed the project from a local notebook experiment into a real web accessible Machine Learning application.


## 🛠️ Technologies Used

| Technology      | Purpose                   |
| --------------- | ------------------------- |
| Python          | Programming language      |
| Pandas          | Data manipulation         |
| NumPy           | Numerical computations    |
| Scikit-learn    | Machine Learning          |
| Streamlit       | Web application framework |
| Pickle          | Model serialization       |
| GitHub          | Version control           |
| Streamlit Cloud | Deployment                |



🚀 Run The Project Locally
Install Dependencies
pip install -r requirements.txt
Run the Streamlit Application
streamlit run app.py

git clone <https://github.com/rofeeah-tijani/student-segmentation-app.git>

Deployment link: https://student-segmentation-app-sfdpsnxo9xubvwp2xnclbj.streamlit.app/
