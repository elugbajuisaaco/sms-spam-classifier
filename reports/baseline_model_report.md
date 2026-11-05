# SMS Spam Classifier — Baseline Model Report

## Model architecture

The baseline is a scikit-learn `Pipeline` containing a `TfidfVectorizer`
followed by `MultinomialNB` with its default smoothing parameter (`alpha=1.0`).
Keeping vectorization and classification in one pipeline ensures the vocabulary
and inverse document frequencies are learned only from the model-training subset.

### TF-IDF configuration

- `lowercase=True`
- `strip_accents="unicode"`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`

## Validation strategy

Only `data/processed/train.csv` was used. It was split with an 80/20
stratified `train_test_split` using `random_state=42`. The pipeline was fit
only on the model-training subset; the validation subset was used only for
prediction and evaluation. `data/processed/test.csv` was not loaded or used.

- Model-training subset: **3,308** records
- Validation subset: **828** records
- Positive evaluation class: **spam**

## Validation metrics

| Metric (spam positive class) | Score |
|---|---:|
| Accuracy | 0.9626 |
| Precision | 1.0000 |
| Recall | 0.7048 |
| F1-score | 0.8268 |

### Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual \ Predicted | ham | spam |
|---|---:|---:|
| ham | 723 | 0 |
| spam | 31 | 74 |

### Classification report

```text
              precision    recall  f1-score   support

         ham     0.9589    1.0000    0.9790       723
        spam     1.0000    0.7048    0.8268       105

    accuracy                         0.9626       828
   macro avg     0.9794    0.8524    0.9029       828
weighted avg     0.9641    0.9626    0.9597       828
```

## Interpretation

- **True positive (TP):** spam correctly predicted as spam (74).
- **True negative (TN):** ham correctly predicted as ham (723).
- **False positive (FP):** ham incorrectly predicted as spam (0); legitimate messages may be flagged.
- **False negative (FN):** spam incorrectly predicted as ham (31); these spam messages may reach the recipient as legitimate mail.

False negatives deserve particular attention because they represent spam that
the classifier misses. Recall measures the fraction of actual spam found;
precision measures how often a spam prediction is correct. The F1-score
summarizes their balance. The measured scores describe this validation split
only, not the untouched final test set or future messages.

## Limitations

- This is a simple bag-of-words baseline; it does not understand message
  meaning, context, or sender behavior.
- The dataset has substantially more ham than spam, so accuracy alone can
  obscure weak performance on spam.
- A single random validation split can give a noisy estimate.
- Real-world messages may differ from this historical dataset, and false
  positives can prevent legitimate messages from being delivered.

## Recommendations

- Compare spam precision, recall, and F1-score, with special consideration of
  false negatives and the cost of false positives.
- Use cross-validation on training data for a more stable comparison.
- Review misclassified validation examples and assess whether additional
  message-level signals are appropriate.
- Keep the final test set untouched until model selection is complete.
