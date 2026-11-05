# SMS Spam Collection: Dataset Inspection

## Source

The original dataset was obtained from the [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The downloaded archive and unmodified `SMSSpamCollection` file are preserved in `data/raw/`. The source file has no header and contains tab-separated label and message fields.

## Dimensions and columns

The dataset contains **5,574 rows and 2 columns**.

| Column | Description |
|---|---|
| `label` | Message category: `ham` (legitimate) or `spam`, as described by UCI. |
| `message` | Original SMS message text. |

These column names are assigned during inspection; they are not a header in the original file.

The parser uses `sep="\t"`, `header=None`, the two explicit column names, UTF-8 decoding, and `quoting=csv.QUOTE_NONE`. The SMS body is free text and can contain literal quote characters. Pandas' default CSV quote handling treated those characters as field enclosures, changing message values and yielding 5,572 rather than 5,574 records. Disabling quote interpretation preserves all 5,574 tab-separated source records and message text.

## Class distribution

| Label | Count | Percentage |
|---|---:|---:|
| ham | 4,827 | 86.60% |
| spam | 747 | 13.40% |
| **Total** | **5,574** | **100.00%** |

## Data quality

- **Missing values:** 0 in `label`; 0 in `message`.
- **Duplicate message text:** 403 repeated occurrences beyond the first, across 281 message-text groups.
- **Exact duplicate rows:** 403 repeated label-and-message records beyond the first.
- **Conflicting labels:** 0 message texts appear under both `ham` and `spam`.
- **Message character length:** mean 80.48; standard deviation 59.85; minimum 2; 25th percentile 36; median 62; 75th percentile 122; 90th percentile 156; 95th percentile 160; maximum 910 characters.

## Prepared dataset

The reproducible preparation script removes only exact duplicate rows, retaining each row's first occurrence and original order. It does not normalize or tokenize message text. The processed file has **5,171 rows and 2 columns**, saved as `data/processed/sms_spam_cleaned.csv`.

| Label | Count after exact deduplication |
|---|---:|
| ham | 4,518 |
| spam | 653 |
| **Total** | **5,171** |

The original file remains unchanged in `data/raw/`; deduplication is applied only to the processed copy.

## Remaining data quality concerns

- The classes are imbalanced: ham messages outnumber spam messages by about 6.5 to 1 before exact deduplication and 6.9 to 1 afterward. Evaluation should account for this rather than relying only on accuracy.
- Repeated text appears in 281 groups (403 extra occurrences). Exact duplicates are removed from the processed copy, but a later train/test split should still consider repeated message text to avoid leakage and inflated evaluation results. No conflicting-label message texts were observed.
- Message lengths vary considerably, with some messages much longer than the median. No messages have been shortened or otherwise altered.
- No missing values were observed in the two parsed fields. Punctuation, capitalization, and message content have not been normalized.

## Raw-data preservation

The raw dataset is retained to preserve the authoritative source, allow parser and preparation results to be reproduced, and ensure preprocessing choices can be revisited without losing original records. The processed output is derived data and can be recreated:

```powershell
.\.venv-py313\Scripts\python.exe .\src\inspect_data.py
.\.venv-py313\Scripts\python.exe .\src\prepare_data.py
```
