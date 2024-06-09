import streamlit as st
import csv
import io
st.write("22AIA- UNIQUE CODERS")
st.title("Candidate Elimination Algorithm")

# Function to perform Candidate Elimination algorithm
def candidate_elimination(data):
    num_attributes = len(data[0])-1
    S = ['0'] * num_attributes
    G = ['?'] * num_attributes
    version_space = []
    
    st.subheader("Initial Hypotheses:")
    st.write("Most Specific Hypothesis (S0): ", S)
    st.write("Most General Hypothesis (G0): ", G)
    
    for j in range(0,num_attributes):
        S[j] = data[0][j]
    
    for i in range(0, len(data)):
        if data[i][num_attributes] == 'Yes':
            for j in range(0, num_attributes):
                if data[i][j] != S[j]:
                    S[j] = '?'
            for j in range(0, num_attributes):
                for k in range(1, len(version_space)):
                    if version_space[k][j] != '?' and version_space[k][j] != S[j]:
                        del version_space[k]
            st.write("----------------------------------------------------------------------------- ")
            st.write("For Training Example No :", i+1, "the hypothesis is S" + str(i+1), S)
            if len(version_space) == 0:
                st.write("For Training Example No :", i+1, "the hypothesis is G" + str(i+1), G)
            else:
                st.write("For Positive Training Example No :", i+1, "the hypothesis is G" + str(i+1), version_space)
                
        if data[i][num_attributes] == 'No':
            for j in range(0, num_attributes):
                if S[j] != data[i][j] and S[j] != '?':
                    G[j] = S[j]
                    version_space.append(G)
                    G = ['?'] * num_attributes
            st.write("----------------------------------------------------------------------------- ")
            st.write("For Training Example No :", i+1, "the hypothesis is S" + str(i+1), S)
            st.write("For Training Example No :", i+1, "the hypothesis is G" + str(i+1), version_space)

# User input for uploading CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file as string
    file_contents = uploaded_file.getvalue().decode("utf-8")
    # Convert string to file-like object
    stringio = io.StringIO(file_contents)
    # Load data
    data = list(csv.reader(stringio))

    st.subheader("Training Data Set:")
    for row in data:
        st.write(row)

    # Run Candidate Elimination algorithm
    st.subheader("Candidate Elimination Algorithm Hypotheses Version Space Computation:")
    candidate_elimination(data)
