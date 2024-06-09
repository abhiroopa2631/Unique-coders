import streamlit as st
import csv

def main():
    st.write("22AIA- UNIQUE CODERS")
    st.title("Candidate Elimination Algorithm Visualization")
    
    st.write("Upload your dataset in CSV format:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.write("The Given Training Data Set:")
        with uploaded_file:
            reader = csv.reader(uploaded_file)
            for row in reader:
                st.write(row)
        
        a = []
        num_attributes = 0
        
        with uploaded_file:
            reader = csv.reader(uploaded_file)
            for row in reader:
                a.append(row)
                num_attributes = len(row) - 1
        
        st.write("The initial value of hypothesis:")
        S = ['0'] * num_attributes
        G = ['?'] * num_attributes
        st.write("The most specific hypothesis S0: ", S)
        st.write("The most general hypothesis G0: ", G)
        
        for j in range(0, num_attributes):
            S[j] = a[0][j]
        
        st.write("Candidate Elimination algorithm Hypotheses Version Space Computation:")
        temp = []
        
        for i in range(0, len(a)):
            if a[i][num_attributes] == 'Yes':
                for j in range(0, num_attributes):
                    if a[i][j] != S[j]:
                        S[j] = '?'
                
                for j in range(0, num_attributes):
                    for k in range(1, len(temp)):
                        if temp[k][j] != '?' and temp[k][j] != S[j]:
                            del temp[k]
                
                st.write("-----------------------------------------------------------------------------")
                st.write("For Training Example No: ", i + 1, "the hypothesis is S" + str(i + 1), S)
                if len(temp) == 0:
                    st.write("For Training Example No: ", i + 1, "the hypothesis is G" + str(i + 1), G)
                else:
                    st.write("For Positive Training Example No: ", i + 1, "the hypothesis is G" + str(i + 1), temp)
            
            if a[i][num_attributes] == 'No':
                for j in range(0, num_attributes):
                    if S[j] != a[i][j] and S[j] != '?':
                        G[j] = S[j]
                temp.append(G)
                G = ['?'] * num_attributes
                
                st.write("-----------------------------------------------------------------------------")
                st.write("For Training Example No: ", i + 1, "the hypothesis is S" + str(i + 1), S)
                st.write("For Training Example No: ", i + 1, "the hypothesis is G" + str(i + 1), temp)

if __name__ == "__main__":
    main()
