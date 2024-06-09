import streamlit as st
import numpy as np
import csv
import io

# Function to read data from CSV file
def read_data(file):
    content = file.read().decode('utf-8')
    datareader = csv.reader(io.StringIO(content))
    metadata = next(datareader)
    traindata = [row for row in datareader]
    return (metadata, traindata)

# Function to split dataset into training and testing sets
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    testset = list(dataset)
    i = 0
    while len(trainSet) < trainSize:
        trainSet.append(testset.pop(i))
    return [trainSet, testset]

# Naive Bayes classifier
def classify(data, test):
    total_size = data.shape[0]
    st.write("Training data size =", total_size)
    st.write("Test data size =", test.shape[0])
    target = np.unique(data[:, -1])
    count = np.zeros((target.shape[0]), dtype=np.int32)
    prob = np.zeros((target.shape[0]), dtype=np.float32)

    st.write("Target, Count, Probability")

    for y in range(target.shape[0]):
        for x in range(data.shape[0]):
            if data[x, data.shape[1] - 1] == target[y]:
                count[y] += 1
        prob[y] = count[y] / total_size  # Computes the probability of target
        st.write(f"{target[y]}\t{count[y]}\t{prob[y]}")

    prob0 = np.zeros((test.shape[1] - 1), dtype=np.float32)
    prob1 = np.zeros((test.shape[1] - 1), dtype=np.float32)
    accuracy = 0
    st.write("Instance, Prediction, Target")
    for t in range(test.shape[0]):
        for k in range(test.shape[1] - 1):
            count1 = count0 = 0
            for j in range(data.shape[0]):
                if test[t, k] == data[j, k] and data[j, data.shape[1] - 1] == target[0]:
                    count0 += 1
                elif test[t, k] == data[j, k] and data[j, data.shape[1] - 1] == target[1]:
                    count1 += 1
            prob0[k] = count0 / count[0]
            prob1[k] = count1 / count[1]

        probno = prob[0]
        probyes = prob[1]
        for i in range(test.shape[1] - 1):
            probno = probno * prob0[i]
            probyes = probyes * prob1[i]

        if probno > probyes:
            predict = target[0]
        else:
            predict = target[1]
        st.write(f"{t+1}\t{predict}\t{test[t, test.shape[1] - 1]}")

        if predict == test[t, test.shape[1] - 1]):
            accuracy += 1
    final_accuracy = (accuracy / test.shape[0]) * 100
    st.write(f"Accuracy: {final_accuracy}%")
    return

# Streamlit app
st.title("Naive Bayes Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    metadata, traindata = read_data(uploaded_file)
    splitRatio = st.slider("Split Ratio", 0.1, 0.9, 0.6)
    trainingset, testset = splitDataset(traindata, splitRatio)
    training = np.array(trainingset)
    testing = np.array(testset)

    st.write("------------------Training Data ------------------ ")
    st.write(trainingset)
    st.write("-------------------Test Data ------------------ ")
    st.write(testset)

    if st.button("Classify"):
        classify(training, testing)
