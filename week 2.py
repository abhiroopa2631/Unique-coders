import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import math

def entropy_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instance = len(a_list) * 1.0
    probs = [x / num_instance for x in cnt.values()]
    return entropy(probs)

def entropy(probs):
    return sum([-prob * math.log(prob, 2) for prob in probs])

def info_gain(df, split, target):
    df_split = df.groupby(split)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target: [entropy_list, lambda x: len(x) / nobs]})
    df_agg_ent.columns = ['Entropy', 'PropObserved']
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent["PropObserved"])
    old_entropy = entropy_list(df[target])
    return old_entropy - new_entropy

def id3(df, target, attribute_name, default_class=None):
    cnt = Counter(x for x in df[target])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attribute_name):
        return default_class
    else:
        default_class = max(cnt.keys())
        gains = [info_gain(df, attr, target) for attr in attribute_name]
        index_max = gains.index(max(gains))
        best_attr = attribute_name[index_max]
        tree = {best_attr: {}}
        remaining_attr = [x for x in attribute_name if x != best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target, remaining_attr, default_class)
            tree[best_attr][attr_val] = subtree
        return tree

def classify(instance, tree, default=None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classify(instance, result)
        else:
            return result
    else:
        return default

def main():
    st. write("22AIA -UNIQUE CODERS")
    st.title("Decision Tree Classifier using ID3 Algorithm")
    
    st.write("Upload your dataset in CSV format:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        attribute_names = list(df.columns)
        target_attribute = 'PlayTennis'
        attribute_names.remove(target_attribute)

        tree = id3(df, target_attribute, attribute_names)
        st.write("\n\nThe Resultant Decision Tree is :\n", tree)

        training_data = df.iloc[:-4]  # all but last four instances
        test_data = df.iloc[-4:]  # just the last four instances
        
        train_tree = id3(training_data, target_attribute, attribute_names)
        st.write("\n\nThe Resultant Decision tree for training data is :\n", train_tree)

        test_data['predicted'] = test_data.apply(classify, axis=1, args=(train_tree, 'Yes'))
        accuracy = sum(test_data['PlayTennis'] == test_data['predicted']) / len(test_data.index)
        st.write("\n\nTraining the model for a few samples, and again predicting 'PlayTennis' for remaining attributes.")
        st.write("The Accuracy for new trained data is: {:.2f}%".format(accuracy * 100))

if __name__ == "__main__":
    main()
