import streamlit as st
import numpy as np

st.title("Neural Network using Backpropagation")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 - x)

def neural_network(X, y, epoch, eta, input_neurons, hidden_neurons, output_neurons):
    X_normalized = X / np.amax(X, axis=0)
    y_normalized = y / 100

    # Variable initialization
    wh = np.random.uniform(size=(input_neurons, hidden_neurons))  # 2x3
    bh = np.random.uniform(size=(1, hidden_neurons))  # 1x3
    wout = np.random.uniform(size=(hidden_neurons, output_neurons))  # 1x1
    bout = np.random.uniform(size=(1, output_neurons))

    for i in range(epoch):
        # Forward pass
        h_ip = np.dot(X_normalized, wh) + bh
        h_act = sigmoid(h_ip)
        o_ip = np.dot(h_act, wout) + bout
        output = sigmoid(o_ip)

        # Backpropagation
        Eo = y_normalized - output
        outgrad = sigmoid_grad(output)
        d_output = Eo * outgrad

        Eh = d_output.dot(wout.T)
        hiddengrad = sigmoid_grad(h_act)
        d_hidden = Eh * hiddengrad

        # Update weights
        wout += h_act.T.dot(d_output) * eta
        wh += X_normalized.T.dot(d_hidden) * eta

    return output

def main():
    st.sidebar.title("Neural Network Parameters")
    epoch = st.sidebar.slider("Epoch", min_value=100, max_value=5000, step=100, value=1000)
    eta = st.sidebar.slider("Learning Rate (eta)", min_value=0.01, max_value=1.0, step=0.01, value=0.2)
    input_neurons = st.sidebar.number_input("Number of Input Neurons", min_value=1, value=2)
    hidden_neurons = st.sidebar.number_input("Number of Hidden Neurons", min_value=1, value=3)
    output_neurons = st.sidebar.number_input("Number of Output Neurons", min_value=1, value=1)

    st.subheader("Input Data")
    X_input = []
    y_input = []

    for i in range(input_neurons):
        input_val = st.number_input(f"Input {i+1}", value=0.0)
        X_input.append(input_val)

    for i in range(output_neurons):
        output_val = st.number_input(f"Output {i+1}", value=0.0)
        y_input.append(output_val)

    X = np.array([X_input])
    y = np.array([y_input])

    predicted_output = neural_network(X, y, epoch, eta, input_neurons, hidden_neurons, output_neurons)

    st.subheader("Results")
    st.write("Input Data:")
    st.write(X)
    st.write("Actual Output:")
    st.write(y)
    st.write("Predicted Output:")
    st.write(predicted_output)

if __name__ == "__main__":
    main()
