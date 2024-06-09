import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def main():
    st.title('Text Classification with Naive Bayes')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        msg = pd.read_csv(uploaded_file, names=['message', 'label'])
        
        st.write("Total Instances of Dataset: ", msg.shape[0])

        # Check for missing values
        st.write("Missing values in dataset: \n", msg.isnull().sum())

        # Drop rows with missing values
        msg.dropna(inplace=True)
        st.write("Total Instances of Dataset after dropping missing values: ", msg.shape[0])

        # Map the 'pos' and 'neg' labels to 1 and 0 respectively
        msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

        # Verify mapping
        st.write("Unique values in labelnum: ", msg['labelnum'].unique())

        # Ensure there are no NaN values after mapping
        st.write("Missing values after mapping: \n", msg.isnull().sum())

        # Define features and labels
        X = msg.message
        y = msg.labelnum

        # Split the data into training and test sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        # Verify the splits
        st.write("Training labels distribution:\n", ytrain.value_counts())
        st.write("Test labels distribution:\n", ytest.value_counts())

        # Vectorize the text data
        count_v = CountVectorizer()
        Xtrain_dm = count_v.fit_transform(Xtrain)
        Xtest_dm = count_v.transform(Xtest)

        # Create a DataFrame from the training data matrix
        df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
        st.write(df.head())

        # Train the Multinomial Naive Bayes classifier
        clf = MultinomialNB()
        clf.fit(Xtrain_dm, ytrain)

        # Predict the labels for the test set
        pred = clf.predict(Xtest_dm)

        # Print the predictions for the test set
        st.write("Predictions for the test set:")
        for doc, p in zip(Xtest, pred):
            label = 'pos' if p == 1 else 'neg'
            st.write(f"{doc} -> {label}")

        # Verify there are no NaN values in ytest or pred
        st.write("Missing values in ytest: ", pd.isnull(ytest).sum())
        st.write("Missing values in pred: ", pd.isnull(pred).sum())

        # Calculate and print accuracy metrics
        st.write('Accuracy Metrics: \n')
        st.write('Accuracy: ', accuracy_score(ytest, pred))
        st.write('Recall: ', recall_score(ytest, pred))
        st.write('Precision: ', precision_score(ytest, pred))
        st.write('Confusion Matrix: \n', confusion_matrix(ytest, pred))

if __name__ == "__main__":
    main()
