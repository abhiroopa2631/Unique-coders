import streamlit as st
import csv

def candidate_elimination(file_path):
    a = []
    with open(file_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            a.append(row)
    
    num_attributes = len(a[0])-1
    S = ['0'] * num_attributes
    G = ['?'] * num_attributes
    output = []

    for j in range(0,num_attributes):
        S[j] = a[0][j]

    for i in range(0,len(a)):
        if a[i][num_attributes] == 'Yes':
            for j in range(0,num_attributes):
                if a[i][j] != S[j]:
                    S[j] = '?'
            for j in range(0,num_attributes):
                for k in range(1,len(output)):
                    if output[k][j] != '?' and output[k][j] != S[j]:
                        del output[k]

            output.append(S[:])
            
        if a[i][num_attributes] == 'No':
            for j in range(0,num_attributes):
                if S[j] != a[i][j] and S[j] != '?':
                    G[j] = S[j]
                    output.append(G[:])
                    G = ['?'] * num_attributes
    
    return output

def main():
    st.title("Candidate Elimination Algorithm")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        file_path = "temp_file.csv"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        output = candidate_elimination(file_path)
        
        st.write("Version Space:")
        for i, hypothesis in enumerate(output):
            st.write(f"Hypothesis {i}: {hypothesis}")

if __name__ == "__main__":
    main()
