# Pokémon Type Classification with PyTorch

This project aims to classify Pokémon into their primary types using a neural network implemented in PyTorch. By leveraging the "Complete Pokémon Dataset" from Kaggle, the model learns to predict a Pokémon's type based on various attributes.

## Introduction

Classifying Pokémon by their primary type can provide insights into their characteristics and behaviors. This project utilizes a neural network to perform this classification task, offering a practical application of machine learning techniques in the context of the Pokémon universe.

# Model Architecture

## Input Layer

The input layer accepts features such as `'hp'`, `'attack'`, `'defense'`, etc.

## Hidden Layers

Two fully connected layers with ReLU activation and dropout for regularization.

## Output Layer

Outputs probabilities for each Pokémon type.

# Training

The model is trained using the Adam optimizer and Cross-Entropy Loss over 20 epochs. Training data is split into training and testing sets with an 80-20 ratio.

# Evaluation

Model performance is evaluated on the test set, with accuracy as the primary metric. Results are printed after evaluation.

# Results

The model achieves an accuracy of approximately 85% on the test set, indicating a strong ability to classify Pokémon by their primary types based on the provided attributes.
