# Credit Card Fraud Detection

This project builds and evaluates a fraud-detection model for credit card transactions using a tuned Random Forest pipeline. It is designed for highly imbalanced classification data, where fraudulent transactions are much rarer than legitimate ones.

The workflow loads a prepared dataset, prepares the feature matrix, evaluates the model with precision-recall and ROC-oriented metrics, and generates a final submission CSV for test data predictions.

## Project goal

Detect fraudulent credit card transactions with a model trained on historical labeled transaction data and save the predictions in a submission-ready format.

## Key files

- `main.py` — loads the dataset, builds the model pipeline, loads the tuned model, and generates the final prediction file.
- `ml_utils.py` — evaluation utilities for model scoring, confusion matrix plotting, and cross-validation.
- `requirements.txt` — Python dependencies used by the project.
- `tuned_random_forest_v1.joblib` — serialized trained model.
- `my_final_submission.csv` — generated submission file produced from the test set predictions.

## Model approach

The project uses a scikit-learn pipeline with:

- StandardScaler on the `Amount` feature
- passthrough for the remaining numeric features
- RandomForestClassifier configured for class imbalance and tuned for fraud detection

The final decision threshold is set to `0.7`, and predictions are written to a CSV file with the same schema as the sample submission.

## Dataset requirements

This repository expects a credit card fraud dataset with the following files available locally:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The current `main.py` script contains absolute local file paths to a Windows dataset directory. Before running it, update those file paths to match your environment.

## Setup

1. Clone the repository.
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it:

   On Windows (PowerShell):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On Windows (Command Prompt):

   ```cmd
   .\.venv\Scripts\activate.bat
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the project

From the project root, execute:

```bash
python main.py
```

This script:

- reads the training and test data
- loads the saved tuned model
- predicts fraud probabilities on the test set
- applies a threshold of `0.7`
- writes predictions to `my_final_submission.csv`

## Notes

- The project is built for local experimentation and model iteration.
- Because the dataset paths are hardcoded in `main.py`, you may need to modify them to suit your machine or project layout.
- The training logic and evaluation utilities are structured for model development, benchmarking, and generating a final prediction file for competition-style deployment.

## Result

The project produces a submission-ready output file that can be used to submit model predictions for the test dataset.
