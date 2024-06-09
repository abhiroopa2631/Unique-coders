import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('tennisdata.csv')

# Preprocess data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le_outlook = LabelEncoder()
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

le_temperature = LabelEncoder()
X['Temperature'] = le_temperature.fit_transform(X['Temperature'])

le_humidity = LabelEncoder()
X['Humidity'] = le_humidity.fit_transform(X['Humidity'])

le_windy = LabelEncoder()
X['Windy'] = le_windy.fit_transform(X['Windy'])

le_playtennis = LabelEncoder()
y = le_playtennis.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Define accuracy
accuracy = accuracy_score(classifier.predict(X_test), y_test)

# Streamlit app
st.title("Tennis Play Predictor")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

st.header("Input Features")

outlook = st.selectbox("Outlook", le_outlook.classes_)
temperature = st.selectbox("Temperature", le_temperature.classes_)
humidity = st.selectbox("Humidity", le_humidity.classes_)
windy = st.selectbox("Windy", le_windy.classes_)

# Encoding user input
input_data = pd.DataFrame({
    'Outlook': [le_outlook.transform([outlook])[0]],
    'Temperature': [le_temperature.transform([temperature])[0]],
    'Humidity': [le_humidity.transform([humidity])[0]],
    'Windy': [le_windy.transform([windy])[0]]
})

if st.button("Predict"):
    prediction = classifier.predict(input_data)
    result = le_playtennis.inverse_transform(prediction)
    st.write(f"Prediction: {'Play Tennis' if result[0] == 'Yes' else 'Dont Play Tennis'}")
