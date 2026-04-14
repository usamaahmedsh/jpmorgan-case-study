# Census-Income Classification and Segmentation

Take-home project for a retail client who wants to identify customers earning more than
\$50K for targeted marketing. The repo contains two models: a classifier that scores
customers by their likelihood of being high earners, and a segmentation model that groups
customers into behaviorally distinct profiles.

## Deliverables

The four deliverables from the project brief map to this repo as follows:

1. **Code for training and evaluating the classification model.** Part 1 of
   [`notebooks/main.ipynb`](notebooks/main.ipynb).
2. **Code for generating the segmentation model.** Part 2 of the same notebook.
3. **This README**, with setup and execution instructions.
4. **Project report.** [`report/report.pdf`](report/report.pdf), 10 pages, covering data
   exploration, preprocessing, feature engineering, model architecture, training, evaluation,
   findings, business judgment, limitations, and references.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── main.ipynb
└── report/
    ├── report.pdf
    └── figures/
```

## Data

The project expects two files provided by the client:

- `census-bureau.data` — 40 demographic and employment variables, a sampling weight, and a
  binary income label, comma-delimited, no header.
- `census-bureau.columns` — column names aligned to the indices in the data file.

Place both files in a `data/` directory at the repo root before running the notebook.

## Setup

Tested on Python 3.11.

```bash
git clone https://github.com/usamaahmedsh/jpmorgan-case-study
cd jpmorgan-case-study
python -m venv venv
source venv/bin/activate           # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the code

```bash
cd notebooks
jupyter notebook main.ipynb
```

Run cells top to bottom. The notebook loads the data from `../data/`, so run it from
inside the `notebooks/` directory.

The notebook is organized in two parts that match the two tasks:

- Part 1, Classification: EDA, feature engineering, logistic regression baseline, XGBoost
  (base, tuned, calibrated, and weighted-evaluation variants), SHAP feature attribution.
- Part 2, Segmentation: TruncatedSVD dimensionality reduction, Gaussian Mixture Model
  clustering, stability check via Adjusted Rand Index, segment profiles, and marketing
  recommendations.

End-to-end runtime is roughly 25 to 35 minutes on a laptop. The two slow stages are the
`RandomizedSearchCV` hyperparameter sweep in Part 1 and the SHAP value computation on the
test set.

## Reproducibility

All randomness is seeded with `SEED = 42` at the top of the notebook. A fresh run should
produce the metrics in Table 1 of the report to within bootstrap noise. GMM cluster labels
are arbitrary across runs, so the cluster numbering in Part 2 may differ from the report,
but the cluster identities, defined by their feature profiles, will be the same.

## Author

Usama Ahmed, April 2026.
