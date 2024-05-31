import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title("Iris Dataset K-Nearest Neighbors Classifier")

    # Load the dataset
    dataset = load_iris()
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)
    
    # Train the KNN model
    kn = KNeighborsClassifier(n_neighbors=1)
    kn.fit(X_train, y_train)

    # Display the predictions
    st.write("Predictions on the test set:")
    for i in range(len(X_test)):
        x = X_test[i]
        x_new = np.array([x])
        prediction = kn.predict(x_new)
        st.write(f"TARGET = {y_test[i]} ({dataset['target_names'][y_test[i]]}), "
                 f"PREDICTED = {prediction[0]} ({dataset['target_names'][prediction[0]]})")
    
    # Display the accuracy
    accuracy = kn.score(X_test, y_test)
    st.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
