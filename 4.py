import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Tennis Data Classification")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("The first 5 rows of the data are:")
    st.write(data.head())

    # Step 2: Prepare Data
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    st.write("The first 5 rows of the train data are:")
    st.write(X.head())
    
    st.write("The first 5 rows of the train output are:")
    st.write(y.head())

    # Step 3: Encode Data
    le_outlook = LabelEncoder()
    X.Outlook = le_outlook.fit_transform(X.Outlook)

    le_Temperature = LabelEncoder()
    X.Temperature = le_Temperature.fit_transform(X.Temperature)

    le_Humidity = LabelEncoder()
    X.Humidity = le_Humidity.fit_transform(X.Humidity)

    le_Windy = LabelEncoder()
    X.Windy = le_Windy.fit_transform(X.Windy)

    st.write("Now the train data is:")
    st.write(X.head())

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)
    st.write("Now the train output is:")
    st.write(y)

    # Step 4: Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Step 5: Train Model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Step 6: Evaluate Model
    accuracy = accuracy_score(classifier.predict(X_test), y_test)
    st.write(f"Accuracy of the model is: {accuracy:.2f}")
