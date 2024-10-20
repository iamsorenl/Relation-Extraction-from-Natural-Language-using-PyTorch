# Relation Extraction from Natural Language using PyTorch

This project implements a **Multi-Layer Perceptron (MLP)** for relation extraction from natural language sentences using **PyTorch**. It predicts core relations between entities in a sentence based on predefined relations, using **spaCy** for text processing and static word embeddings to generate features for classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

Relation extraction is a vital NLP task involving the identification of relationships between entities within a sentence. This project:

- Uses CSV datasets for **training** and **testing**.
- Trains an **MLP** with **stratified K-fold cross-validation** to handle multi-label classification.
- Processes input sentences using **spaCy** for feature extraction.
- Utilizes **static embeddings** for relation classification.
- Evaluates model performance using **accuracy** and **F1-score** metrics.

---

## Requirements

Make sure you have the following dependencies available. For a full list, refer to `requirements.txt`.

**Main Dependencies:**
- Python >= 3.11
- PyTorch
- spaCy
- scikit-learn
- pandas
- numpy
- iterative-stratification

### Install Packages

Once you have a virtual environment ready, install the required packages using:

```bash
pip install -r requirements.txt

Installation

1. Clone the Repository

git clone https://github.com/your-username/relation-extraction-pytorch.git
cd relation-extraction-pytorch

2. Create a Virtual Environment

To avoid conflicts with other packages, it’s recommended to use a virtual environment:

python3 -m venv venv
# On MacOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

3. Install Dependencies

Install all required dependencies:

pip install -r requirements.txt

4. Download spaCy Model

The project uses spaCy’s en_core_web_md model for word embeddings:

python -m spacy download en_core_web_md

Dataset Structure

Ensure the following CSV files are present in the project directory:

	•	hw1_train.csv: Training dataset with utterances and core relations.
	•	hw1_test.csv: Test dataset with utterances for which core relations need to be predicted.
	•	exampleSubmission.csv: Example output format for predictions.

Each dataset follows these structures:

Training Dataset (hw1_train.csv):

ID	UTTERANCE	CORE RELATIONS
1	who directed Titanic?	movie.directed_by
2	list Pixar animated films	movie.production_companies

Test Dataset (hw1_test.csv):

ID	UTTERANCE
1	who directed Titanic?
2	list Pixar animated films

Usage

To train the model and generate predictions, use:

python run.py hw1_train.csv hw1_test.csv submission.csv

This command will:

	•	Train the MLP using K-fold cross-validation.
	•	Process the test data and predict core relations.
	•	Save predictions to submission.csv.

File Output

The results will be saved as submission.csv with the following structure:

ID	CORE RELATIONS
1	movie.directed_by
2	movie.directed_by, movie.genre

Training the Model

The MLP model workflow includes:

	1.	Preprocessing:
	•	spaCy processes each utterance and converts it into embeddings.
	2.	Cross-Validation:
	•	Stratified K-Fold cross-validation ensures an even distribution of labels.
	3.	Training:
	•	The MLP model is trained on the processed embeddings to predict relations.

Evaluation

After each cross-validation fold, the following metrics are calculated:

	•	Accuracy: Measures how many predictions match the ground truth.
	•	F1-Score: Balances precision and recall for multi-label classification.

Example:

Metric	Fold 1	Fold 2	Fold 3	Fold 4	Fold 5	Mean
Accuracy	0.86	0.87	0.85	0.89	0.87	0.87
F1-Score	0.91	0.93	0.89	0.94	0.92	0.92

Troubleshooting

Common Issues

	1.	NumPy Version Errors:
Ensure that NumPy is compatible with other dependencies:

pip install numpy


	2.	Virtual Environment Issues:
If packages are not recognized, ensure the virtual environment is activated:

source venv/bin/activate


	3.	Dependency Conflicts:
If you encounter conflicts, try relaxing version restrictions in requirements.txt.

Hyperparameter Tuning

During model development, the following hyperparameters were optimized:

	•	Learning Rate: Best value found was 0.01.
	•	Dropout Rate: A rate of 0.3 minimized overfitting.
	•	Hidden Neurons: Each hidden layer was set to 128 neurons for optimal performance.

Comparison with State-of-the-Art

While transformer-based models like BERT may offer higher accuracy, the MLP architecture used here provides a balance of simplicity and performance. It remains an effective solution for tasks where computational efficiency is critical.

Contributing

Contributions are welcome! If you encounter issues or have ideas for improvements, feel free to open a pull request or issue.

