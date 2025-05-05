# Product Recommendation Classifier

This project builds a machine learning pipeline to predict whether a customer would recommend a clothing product based on their review. It leverages numerical, categorical, and text features and applies NLP techniques with `spaCy`, text vectorization with `TfidfVectorizer`, and classification using a Random Forest model.

## Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.


### Dependencies

```
pandas
numpy
scikit-learn
spacy
en_core_web_sm  # spaCy model
jupyterlab  # optional, for notebook interface
```

### Installation

Clone the repository or download the project files
```bash
git clone https://github.com/udacity/dsnd-pipelines-project.git
cd dsnd-pipelines-project
```

Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install all required packages
```bash
pip install -r requirements.txt
```

Download the spaCy English model
```bash
python -m spacy download en_core_web_sm
```

(Optional) Launch JupyterLab
```bash
jupyter lab
```

## Testing

To test and evaluate the pipeline:

* Run the training script or notebook.
* Observe the output of the classification_report for metrics like accuracy, precision, recall, and f1-score.

### Break Down Tests

```
# Train/test split
# Fit the pipeline with preprocessing steps
# Generate predictions
# Evaluate results with classification_report
# Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
```
* Test 1: Verifies pipeline fit and transform steps.
* Test 2: Confirms prediction and evaluates model accuracy.
* Test 3: Tunes model performance using hyperparameter search.


## Project Instructions

* Create steps for processing text data to be used in a model pipeline
* Create steps for creating features from text data to be used in a model pipeline
* Create a model pipeline that combines your preprocessing, feature engineering, and model prediction steps
* Fine-tune your model pipeline to find optimal hyperparameters for your model
* Evaluate with the test data your final model pipeline after being trained with all the training data

## Built With

- **[scikit-learn](https://scikit-learn.org)** – Machine learning library  
- **[spaCy](https://spacy.io)** – NLP library for tokenization and lemmatization  
- **[pandas](https://pandas.pydata.org)** – Data analysis and manipulation  
- **[NumPy](https://numpy.org)** – Numerical computations  
- **[JupyterLab](https://jupyter.org)** – Interactive development interface

## License

[License](LICENSE.txt)
