# Candidate Elimination Algorithm

This project implements the Candidate Elimination algorithm to learn hypotheses from training examples. The algorithm iteratively refines the hypothesis space based on positive and negative examples.

## Files

- `candidate_elimination.py`: Python script containing the implementation of the Candidate Elimination algorithm.
- `tennis.csv`: Dataset containing training examples for the algorithm.

## Usage

1. Ensure you have Python installed on your system.
2. Run the script `candidate_elimination.py` to execute the Candidate Elimination algorithm on the 'tennis.csv' dataset.

## Implementation Details

- `candidate_elimination.py`: Python script containing the implementation of the Candidate Elimination algorithm.
- `tennis.csv`: CSV file containing training examples with attributes and class labels.

## Dataset

The 'tennis.csv' dataset contains the following columns:

1. Outlook: Weather outlook (Sunny, Overcast, Rainy)
2. Temperature: Temperature (Hot, Mild, Cool)
3. Humidity: Humidity level (High, Normal)
4. Wind: Wind condition (Weak, Strong)
5. PlayTennis: Whether tennis was played (Yes, No)

## Results

The script executes the Candidate Elimination algorithm on the provided dataset and prints the hypotheses at each step of iteration based on positive and negative examples.

