import streamlit as st
import pandas as pd
import numpy as np

# Display title
st.title("Tennis Play Predictor")

# Load data
@st.cache
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/your_username/your_repo/master/tennisdata.csv')
    return data

data = load_data()

# Preprocess data
data['Outlook'] = data['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rainy': 2})
data['Temperature'] = data['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
data['Humidity'] = data['Humidity'].map({'High': 0, 'Normal': 1})
data['Windy'] = data['Windy'].map({False: 0, True: 1})
data['PlayTennis'] = data['PlayTennis'].map({'No': 0, 'Yes': 1})

# Split data into features and target
X = data[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = data['PlayTennis']

# Train-test split
def train_test_split(X, y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = [i for i in range(n_samples) if i not in test_indices]
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes classifier
class NaiveBayesClassifier:
    def fit(self, X_train, y_train):
        self.priors = {}
        self.posteriors = {}
        self.classes = np.unique(y_train)
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.priors[c] = len(X_c) / len(X_train)
            self.posteriors[c] = X_c.mean(axis=0)
            
    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            probs = {c: np.prod(self._calculate_likelihood(X_test.iloc[i], self.posteriors[c])) * self.priors[c] for c in self.classes}
            y_pred.append(max(probs, key=probs.get))
        return y_pred
    
    def _calculate_likelihood(self, x, class_posterior):
        likelihood = []
        for i, val in enumerate(x):
            mu = class_posterior[i]
            p = (1 / (np.sqrt(2 * np.pi) * mu[1])) * np.exp(-0.5 * ((val - mu[0]) / mu[1]) ** 2)
            likelihood.append(p)
        return likelihood

# Train the model
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# Model accuracy
def accuracy_score(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model accuracy
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Display input features section
st.header("Input Features")

outlook = st.selectbox("Outlook", ['Sunny', 'Overcast', 'Rainy'])
temperature = st.selectbox("Temperature", ['Hot', 'Mild', 'Cool'])
humidity = st.selectbox("Humidity", ['High', 'Normal'])
windy = st.selectbox("Windy", [False, True])

# Encoding user input
input_data = pd.DataFrame({
    'Outlook': [outlook],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Windy': [windy]
})

# Predict and display result on button click
if st.button("Predict"):
    prediction = classifier.predict(input_data)
    st.write(f"Prediction: {'Play Tennis' if prediction[0] == 1 else 'Don't Play Tennis'}")
