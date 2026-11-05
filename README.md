# SMS Spam Classifier

An end-to-end Python project that classifies an SMS message as **spam** or
**ham** (not spam). The Streamlit app uses a saved scikit-learn pipeline
through the shared prediction API in `src/predict.py`.

## Features

- Streamlit interface for classifying a message and viewing its estimated
  spam probability.
- TF-IDF text features and a class-weighted Logistic Regression classifier.
- Saved decision threshold and model bundle at
  `models/final/final_model.joblib`.
- Data preparation, leakage-safe splitting, model evaluation, and automated
  tests.

## Dataset

The project uses the original SMS Spam Collection from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
The original dataset and generated raw, processed, training, and test files
are excluded from Git; they are not needed to run the prediction app.

To reproduce data preparation, download the UCI archive and place its
`SMSSpamCollection` file at `data/raw/SMSSpamCollection`. From the project
root, run:

```powershell
python .\src\inspect_data.py
python .\src\prepare_data.py
python .\src\split_data.py
```

## Model evaluation

The final model was evaluated once on a held-out test set. The reported spam
precision was **0.9225**, spam recall **0.9084**, and spam F1-score **0.9154**
(1,035 test messages). The saved decision threshold is **0.5434265976102519**;
it was selected using training-only out-of-fold predictions, not the held-out
test labels. See `reports/final_model_evaluation.md` for the evaluation
details and limitations.

These results describe one held-out sample from this dataset; they do not
guarantee performance on future SMS messages.

## Setup and run locally (Windows PowerShell)

From the project root, create and activate the tested Python 3.13 environment,
then install the pinned dependencies:

```powershell
py -3.13 -m venv .venv-py313
.\.venv-py313\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

Paste a message into the text area and select **Classify Message**. The app
shows the prediction and estimated spam probability. This probability is a
model estimate, not a guarantee.

The reusable prediction API can also be called directly:

```python
from src.predict import predict_message

result = predict_message("Congratulations! Claim your free prize now.")
print(result.prediction)
print(result.spam_probability)
print(result.threshold)
```

Non-string, empty, and whitespace-only inputs are rejected. The model bundle
is loaded and cached by the prediction module; app startup does not train a
model.

## Tests

Run the test suite from the project root:

```powershell
python -m pytest
```

The data-pipeline integration test expects the generated processed dataset.
Run the data-preparation commands above before running the full suite in a
fresh clone. Prediction and app tests use the saved final model bundle.

## Project structure

```text
app.py                         Streamlit entrypoint
models/final/final_model.joblib  Runtime model, threshold, and metadata
src/                           Prediction and data/model pipeline code
tests/                         Unit and app tests
reports/                       Inspection and evaluation reports
requirements.txt               Pinned Python dependencies
```

## Deployment preparation: Streamlit Community Cloud

The deployment entrypoint is the root-level `app.py`. Keep it alongside
`requirements.txt`, the `src/` package, and
`models/final/final_model.joblib`. The model path is resolved relative to the
prediction module; datasets are not required at runtime.

To deploy when ready:

1. Push the reviewed project files to a GitHub repository.
2. In Streamlit Community Cloud, create an app and select the repository and
   branch.
3. Set the app file path to `app.py` and select Python 3.13 in the advanced
   settings.
4. Deploy. No public deployment URL has been created.

Do not commit secrets, local environment files, raw or processed datasets, or
experimental model artifacts. The final runtime model bundle is explicitly
allowed by `.gitignore`. See the official [Community Cloud deployment
guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [dependency
guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Limitations and responsible use

The dataset is historical, and spam content can change over time. False
positives may hide legitimate messages, while false negatives may leave spam
unflagged. Use the classifier as an informational aid, not as the sole basis
for consequential decisions, and do not treat its probability as certainty.
