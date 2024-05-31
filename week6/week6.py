import streamlit as st
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('Exposure', 'COVID'),
    ('COVID', 'Fever'),
    ('COVID', 'Cough'),
    ('COVID', 'Shortness_of_Breath'),
    ('COVID', 'Test_Result')
])

# Hypothetical data for parameter estimation
np.random.seed(0)
data = pd.DataFrame(data={
    'Exposure': np.random.choice(['Yes', 'No'], size=1000),
    'COVID': np.random.choice(['Yes', 'No'], size=1000),
    'Fever': np.random.choice(['Yes', 'No'], size=1000),
    'Cough': np.random.choice(['Yes', 'No'], size=1000),
    'Shortness_of_Breath': np.random.choice(['Yes', 'No'], size=1000),
    'Test_Result': np.random.choice(['Positive', 'Negative'], size=1000),
})

# Estimate the parameters of the model using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference
infer = VariableElimination(model)

# Streamlit app
st.title("COVID-19 Bayesian Network Diagnosis")

st.header("Input Symptoms and Risk Factors")

exposure = st.selectbox('Exposure to confirmed COVID-19 case?', ['Yes', 'No'])
fever = st.selectbox('Fever?', ['Yes', 'No'])
cough = st.selectbox('Cough?', ['Yes', 'No'])
shortness_of_breath = st.selectbox('Shortness of Breath?', ['Yes', 'No'])
test_result = st.selectbox('COVID-19 Test Result?', ['Positive', 'Negative'])

if st.button('Diagnose'):
    evidence = {
        'Exposure': exposure,
        'Fever': fever,
        'Cough': cough,
        'Shortness_of_Breath': shortness_of_breath,
        'Test_Result': test_result
    }

    result = infer.query(variables=['COVID'], evidence=evidence)
    prob_covid = result.values[1]

    st.subheader(f"Probability of COVID-19 Infection: {prob_covid * 100:.2f}%")
