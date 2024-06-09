import streamlit as st
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

def neural_network_regression(X, y, epoch=1000, eta=0.2):
    X_normalized = X / np.amax(X, axis=0)
    y_normalized = y / 100
    
    input_neurons = X.shape[1]
    hidden_neurons = 3
    output_neurons = 1
    
    wh = np.random.uniform(size=(input_neurons, hidden_neurons))  # 2x3
    bh = np.random.uniform(size=(1, hidden_neurons))  # 1x3
    wout = np.random.uniform(size=(hidden_neurons, output_neurons))  # 3x1
    bout = np.random.uniform(size=(1, output_neurons))  # 1x1
    
    for _ in range(epoch):
        h_ip = np.dot(X_normalized, wh) + bh
        h_act = sigmoid(h_ip)
        o_ip = np.dot(h_act, wout) + bout
        output = sigmoid(o_ip)

        Eo = y_normalized - output
        outgrad = sigmoid_grad(output)
        d_output = Eo * outgrad

        Eh = d_output.dot(wout.T)
        hiddengrad = sigmoid_grad(h_act)
        d_hidden = Eh * hiddengrad

        wout += h_act.T.dot(d_output) * eta
        wh += X_normalized.T.dot(d_hidden) * eta

    return output

def main():
    st.write("22AIA - UNIQUE CODERS")
    st.title("Neural Network Regression using Gradient Descent")
    
    st.write("Input data:")
    X = st.text_area("Enter input data (separated by commas and new lines)", value="2, 9\n1, 5\n3, 6")
    X = np.array([list(map(float, row.split(','))) for row in X.split('\n')])
    
    st.write("Output data:")
    y = st.text_area("Enter output data (separated by new lines)", value="92\n86\n89")
    y = np.array([float(val) for val in y.split('\n')])

    st.write("Training the neural network...")
    output = neural_network_regression(X, y)

    st.write("Normalized Input:")
    st.write(X / np.amax(X, axis=0))
    st.write("Actual Output:")
    st.write(y / 100)
    st.write("Predicted Output:")
    st.write(output)

if __name__ == "__main__":
    main()
