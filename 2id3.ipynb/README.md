# Decision Tree Classifier for Tennis Dataset

This project implements a decision tree classifier using the ID3 algorithm for the Tennis dataset. The decision tree is built to predict whether to play tennis based on weather conditions.

## Files

- `decision_tree_tennis.py`: Python script containing the implementation of the decision tree classifier.
- `tennis.csv`: Dataset containing weather conditions and whether tennis was played.

## Usage

1. Ensure you have Python installed on your system.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the script `decision_tree_tennis.py` to build the decision tree classifier and evaluate its accuracy.

## Implementation Details

- `entropy_list(a_list)`: Function to calculate the entropy of a list.
- `entropy(probs)`: Function to calculate entropy from probabilities.
- `info_gain(df, split, target)`: Function to calculate information gain for a split.
- `id3(df, target, attribute_name)`: Function to build the decision tree using the ID3 algorithm.
- `classify(instance, tree, default)`: Function to classify a new instance using the decision tree.

## Dataset

The Tennis dataset contains the following columns:

1. Outlook: Weather outlook (Sunny, Overcast, Rainy)
2. Temperature: Temperature in Celsius (Hot, Mild, Cool)
3. Humidity: Humidity level (High, Normal)
4. Wind: Wind condition (Weak, Strong)
5. PlayTennis: Whether tennis was played (Yes, No)

## Results

The script builds a decision tree and evaluates its accuracy using a training-testing split.
