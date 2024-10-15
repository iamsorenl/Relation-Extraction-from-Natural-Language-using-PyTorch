Relation Extraction from Natural Language using PyTorch

This project implements a Multi-Layer Perceptron (MLP) for relation extraction from natural language sentences using PyTorch. The goal is to predict core relations between entities based on an utterance, as per a predefined set of relations. This project uses spaCy for text processing and static word embeddings to generate features for classification.

Table of Contents

    •    Project Overview
    •    Requirements
    •    Installation
    •    Dataset Structure
    •    Usage
    •    Training the Model
    •    Evaluation
    •    Troubleshooting

Project Overview

Relation extraction is an essential task in Natural Language Processing (NLP) that involves identifying relationships between entities in a sentence. In this project:

    •    CSV datasets are used for training and testing.
    •    The model is trained using stratified K-fold cross-validation to handle multi-label classification.
    •    The input sentences are processed using spaCy for feature extraction.
    •    A multi-layer perceptron (MLP) is trained to classify the core relations from utterances.
    •    Evaluation metrics include accuracy and F1-score.

Requirements

Ensure that the following dependencies are available in your environment. You can find the complete list in the requirements.txt file.

Main Dependencies

    •    Python >= 3.11
    •    PyTorch
    •    spaCy
    •    scikit-learn
    •    pandas
    •    numpy
    •    iterative-stratification

Python Packages

You can install the required packages using the command below (once you set up a virtual environment):

pip install -r requirements.txt

Installation

1. Clone the Repository

git clone https://github.com/your-username/relation-extraction-pytorch.git
cd relation-extraction-pytorch

2. Create a Virtual Environment

To avoid conflicts with other Python packages, it’s recommended to use a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the Dependencies

pip install -r requirements.txt

4. Download spaCy Model

The project uses the en_core_web_md spaCy model for generating word embeddings:

python -m spacy download en_core_web_md

Dataset Structure

Ensure the following CSV files are present in the project directory:

    •    hw1_train.csv: Training dataset containing utterances and core relations.
    •    hw1_test.csv: Test dataset with utterances where core relations need to be predicted.
    •    exampleSubmission.csv: An example of the output format for predictions.

Each CSV file must follow this structure:

Training Dataset (hw1_train.csv)

ID    UTTERANCE    CORE RELATIONS
1    who directed Titanic?    movie.directed_by
2    list Pixar animated films    movie.production_companies
3    find the female actress…    movie.starring.actor

Test Dataset (hw1_test.csv)

ID    UTTERANCE
1    who directed Titanic?
2    list Pixar animated films

Usage

Training and Testing the Model

To train the model and generate predictions, use the following command:

python run.py hw1_train.csv hw1_test.csv submission.csv

This will:

    •    Train the model using K-fold cross-validation.
    •    Process the test data to predict core relations.
    •    Output the predictions in the format specified by exampleSubmission.csv.

File Output

The results are saved to a file named submission.csv in the current directory.

Training the Model

The project uses a Multi-Layer Perceptron (MLP). Here’s how the training works:

    1.    Data Preprocessing: spaCy processes the input utterances to generate embeddings.
    2.    Cross-Validation: The training data is split using K-fold cross-validation to evaluate the model.
    3.    Model Training: The MLP is trained on the processed embeddings and labels.

Evaluation

After each K-fold iteration, the following evaluation metrics are calculated:

    •    Accuracy: Measures the overall performance in predicting the correct relations.
    •    F1-Score: Balances precision and recall for multi-label classification.

Troubleshooting

Common Issues

    1.    NumPy Version Errors: Ensure that NumPy is compatible with other dependencies:

pip install numpy


    2.    Virtual Environment Issues: If dependencies aren’t recognized, make sure the virtual environment is activated:

source venv/bin/activate


    3.    Dependency Conflicts: If you face dependency conflicts while installing the requirements, try relaxing version restrictions in requirements.txt.

Numpy Errors

If you encounter the error RuntimeError: Numpy is not available, ensure that numpy is installed correctly in the environment and matches the version requirements of other packages (like spacy and thinc).
