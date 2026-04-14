# Census-Income Classification and Segmentation

A retail client wants to identify customers earning more than \$50K for targeted marketing.
This repo trains a classifier to score customers by their likelihood of being high earners,
and a segmentation model to group them into behaviorally distinct profiles for product and
channel decisions.

The data is the Census-Income (KDD) extract from the 1994 and 1995 U.S. Current Population
Surveys: roughly 200,000 records, 40 demographic and employment variables, a sampling weight,
and a binary income label. The positive class is about 6%, so accuracy is not a useful metric.
Evaluation uses PR-AUC, recall at fixed mailer budgets, and Brier score.

## Headline result

The final model is a calibrated XGBoost classifier evaluated against census sampling weights.
On the held-out test set it reaches:

- PR-AUC of 0.655
- Recall of 75.1% within the top 10% of scored customers (vs. 70.7% for a logistic regression baseline)
- Brier score of 0.035 after isotonic calibration

At a fixed 10% mailer budget, the model captures roughly 600 more high-income customers than
the baseline at the same spend.

The full report is in [`report/report.pdf`](report/report.pdf).

## Repo layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── census-bureau.data
│   └── census-bureau.columns
├── notebooks/
│   └── main.ipynb
├── report/
│   ├── report.pdf
│   └── figures/
└── docs/
    └── MLProject_new.pdf
```

## Setup

Tested on Python 3.11.

```bash
git clone https://github.com/usamaahmedsh/JPMorgan-CaseStudy
cd JPMorgan-CaseStudy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The data files (`census-bureau.data` and `census-bureau.columns`) are included in `data/`.

## Running the analysis

```bash
cd notebooks
jupyter notebook main.ipynb
```

Run cells top to bottom. The notebook covers both tasks:

1. EDA, feature engineering, and the classification model (logistic regression baseline,
   XGBoost base, tuned, calibrated, and weighted-evaluation variants).
2. Segmentation using TruncatedSVD and a Gaussian Mixture Model, with stability checks
   via Adjusted Rand Index.

## Reproducibility

All randomness is seeded with `SEED = 42` at the top of the notebook. Re-running the
notebook should produce the numbers in Table 1 of the report to within bootstrap noise.
GMM cluster labels are arbitrary across runs, so the cluster numbering in the segmentation
section may differ from the report; the cluster identities (described by their feature
profiles) will be the same.

## What is not in this repo

- Saved model artifacts. Retrain from the notebook.
- Cached SHAP values. Recomputed on each notebook run.

## References

The full reference list is in the report. The methods used are XGBoost (Chen and Guestrin,
2016), isotonic calibration (Niculescu-Mizil and Caruana, 2005), SHAP (Lundberg and Lee, 2017),
TruncatedSVD via randomized SVD (Halko et al., 2011), Gaussian Mixture Models (Reynolds, 2009),
and the Adjusted Rand Index (Hubert and Arabie, 1985).

## Author

Usama Ahmed, April 2026.
