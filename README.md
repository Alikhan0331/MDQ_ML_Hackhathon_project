# Mastercard Hidden Business Detection

A machine-learning project for identifying **potential hidden businesses** operating through consumer card accounts. The pipeline learns behavioral differences between known business-card activity (B2B) and consumer-card activity (B2C), then assigns a risk score and tier to cards in the consumer pool.

> **Important:** This project is intended for analytical and compliance-review workflows. A high score indicates a card whose behavior resembles the business examples in the training data; it is not proof of wrongdoing and should be reviewed alongside appropriate operational controls and domain expertise.

## Overview

The project is implemented as a reproducible Jupyter Notebook pipeline. It:

- Loads business-card transactions, consumer-card transactions, and merchant reference data from Parquet files.
- Normalizes possible schema variations in the raw transaction files.
- Enriches transactions with merchant reference attributes such as MCC and country.
- Creates transaction-level indicators for online, tokenized, recurring, night-time, weekend, premium-card, large-value, small-value, and suspicious-MCC activity.
- Aggregates transactions into one behavioral feature record per card.
- Trains and tunes multiple binary-classification candidates.
- Uses leakage-aware, out-of-fold evaluation with stratified cross-validation.
- Combines business-distance and anomaly-detection signals into a composite score.
- Applies a secondary model to ambiguous cards in a configurable grey zone.
- Exports risk tiers, trained models, metrics, diagnostic charts, and high-risk MCC details.

## Technologies

| Category | Technologies | Role in the project |
|---|---|---|
| Language | Python 3 | Main implementation language for the data and ML pipeline |
| Notebook environment | Jupyter Notebook, IPython | Interactive execution, inline plots, and presentation of results |
| High-performance data processing | Polars | Fast reading, joining, transformation, grouping, and aggregation of large Parquet transaction datasets |
| Tabular analysis | pandas | Feature matrices, tabular exports, model-facing data preparation, and result manipulation |
| Numerical computing | NumPy | Vectorized numerical operations, distance calculations, percentiles, score blending, and array handling |
| Machine learning framework | scikit-learn | Pipelines, preprocessing, imputing, scaling, cross-validation, randomized search, metrics, and baseline models |
| Gradient boosting | LightGBM | Fast tree-based classifier for non-linear hidden-business patterns and grey-zone refinement |
| Gradient boosting | CatBoost | Alternative tree-based classifier evaluated alongside LightGBM and logistic regression |
| Anomaly detection | Isolation Forest (scikit-learn) | Detects consumer cards that behave unusually relative to the consumer baseline |
| Visualization | Matplotlib, Seaborn | Confusion matrices, ROC and precision-recall curves, score distributions, feature importance, and risk diagnostics |
| Model persistence | joblib | Saves the selected base model and grey-zone refiner as reusable `.pkl` artifacts |
| Data formats | Apache Parquet, CSV, JSON | Parquet for raw and processed data; CSV for predictions and MCC details; JSON for metrics and configuration summaries |
| File and configuration handling | Python `pathlib`, `json` | Cross-platform paths, output directories, and human-readable metrics exports |

## Machine-Learning Architecture

The supervised classification task is:

\[
\text{Business card} = 1, \qquad \text{Consumer card} = 0
\]

### Candidate models

Three model candidates are constructed and compared:

- **Logistic Regression** provides an interpretable baseline. It is paired with `SimpleImputer` and `RobustScaler` because linear models are sensitive to missing values and feature scale.
- **LightGBM (`LGBMClassifier`)** captures non-linear behavioral relationships efficiently and supports class-imbalance weighting through `scale_pos_weight`.
- **CatBoost (`CatBoostClassifier`)** is evaluated as another robust gradient-boosted-tree approach.

Each candidate is wrapped in a scikit-learn `Pipeline`, which ensures preprocessing is consistently applied during training, tuning, and validation.

### Hyperparameter optimization

`RandomizedSearchCV` searches model-specific hyperparameter spaces using ROC-AUC as the optimization metric. Stratified folds preserve the business/consumer class balance during tuning.

### Leakage-aware evaluation

The pipeline evaluates complete composite models using shuffled `StratifiedKFold` out-of-fold validation. For every validation fold, the following components are fitted strictly on that fold's training data:

- The base supervised classifier.
- Robust feature scaling and the business/consumer centroids used by the distance model.
- The consumer-only Isolation Forest.
- The grey-zone refiner.

This design prevents validation examples from influencing transformations, anomaly scores, centroids, or secondary-model training.

### Composite scoring

The first-stage ensemble blends two diagnostic signals:

\[
\text{ensemble score} = 0.65 \times \text{business-distance score} + 0.35 \times \text{Isolation Forest score}
\]

- The **business-distance score** measures whether a card is closer to the business behavioral centroid than the consumer centroid after robust scaling.
- The **Isolation Forest score** measures how anomalous a card appears against the consumer-card population.

For cards with ensemble scores in the grey zone, the pipeline trains a smaller, regularized LightGBM model on confident consumer and business examples. The final score for ambiguous cards is blended as:

\[
\text{final score} = 0.45 \times \text{ensemble score} + 0.55 \times \text{grey-zone probability}
\]

## Feature Engineering

Raw transactions are aggregated to the card level so the models learn behavioral patterns rather than card identifiers. The pipeline generates features across the following categories:

| Feature group | Examples | Why it matters |
|---|---|---|
| Transaction volume | `n_txns`, `burst_cv` | Commercial activity may show higher volume or concentrated bursts of activity |
| Amount behavior | `amt_mean`, `amt_median`, `amt_std`, `amt_cv`, `amt_q95`, `max_same_amt_count`, `log_amt_std` | Repeated or unusually consistent payment amounts can indicate structured spending patterns |
| Merchant concentration | `n_merchants`, `merchant_diversity`, `same_merchant_ratio`, `txns_per_merchant` | Business activity can recur with suppliers or be concentrated around fewer counterparties |
| MCC behavior | `mcc_diversity`, `merchants_per_mcc`, `susp_mcc_ratio` | A narrow or commercially oriented MCC profile can differ from general consumer spending |
| Timing and rhythm | `hour_mean`, `commercial_hours_ratio`, `night_ratio`, `weekend_ratio`, `online_night_ratio` | Regular weekday business-hour behavior can be a useful B2B signal |
| Payment channel | `online_ratio`, `token_ratio`, `recur_ratio` | Online, tokenized, and recurring transactions provide additional context about card use |
| Geography and segment | `foreign_ratio`, `premium_ratio` | Cross-border usage and card-tier patterns supplement the behavioral profile |

The notebook also uses cyclical hour encoding with sine and cosine transformations so that nearby hours across midnight are represented correctly.

## Project Structure

```text
.
├── Mastercard_Hidden_Business_Detection_program.ipynb
├── business_cards_MDQ.parquet
├── consumer_cards_MDQ.parquet
├── merchants_reference.parquet
├── data/
│   └── processed/
│       ├── biz_card_features.parquet
│       └── cons_card_features.parquet
└── outputs/
    ├── figures/
    │   ├── 01_confusion_matrix.png
    │   ├── 02_oof_score_dist.png
    │   ├── 03_roc_curves.png
    │   ├── 04_pr_curves.png
    │   ├── 05_feature_importance.png
    │   ├── 06_grey_zone_analysis.png
    │   ├── 07_consumer_score_breakdown.png
    │   └── 08_top50_suspicious.png
    ├── metrics/
    │   └── metrics.json
    ├── models/
    │   ├── best_base_model.pkl
    │   └── final_grey_refiner.pkl
    └── predictions/
        ├── final_submission.csv
        └── high_risk_mcc_details.csv
```

The raw Parquet files must be placed in the same directory from which the notebook is executed.

## Installation

### Prerequisites

- Python 3.10 or later is recommended.
- Jupyter Notebook, JupyterLab, VS Code, or PyCharm with Jupyter support.
- Sufficient memory and CPU resources for multi-million-row transaction datasets.

### Create an environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Install dependencies

```bash
pip install "polars>=1.0.0" \
            "pandas>=2.2.0" \
            "numpy>=1.26.0,<2.0.0" \
            "scikit-learn>=1.5.0" \
            "lightgbm>=4.3.0" \
            "catboost>=1.2.5" \
            "matplotlib>=3.8.0" \
            "seaborn>=0.13.0" \
            "joblib>=1.4.0" \
            jupyter
```

Optionally, place the dependencies in a `requirements.txt` file:

```text
polars>=1.0.0
pandas>=2.2.0
numpy>=1.26.0,<2.0.0
scikit-learn>=1.5.0
lightgbm>=4.3.0
catboost>=1.2.5
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.4.0
jupyter
```

Then install them with:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

1. Place these three required files alongside the notebook:

```text
business_cards_MDQ.parquet
consumer_cards_MDQ.parquet
merchants_reference.parquet
```

2. Open `Mastercard_Hidden_Business_Detection_program.ipynb`.

3. Run the notebook cells in order, including the cell that calls:

```python
run_pipeline()
```

4. Review the generated artifacts in `data/processed/` and `outputs/`.

The pipeline uses a fixed `RANDOM_STATE = 42` to support reproducible model training, cross-validation splits, hyperparameter search, and anomaly detection.

## Input Data

The project expects three Parquet datasets:

| File | Expected purpose |
|---|---|
| `business_cards_MDQ.parquet` | Known business-card transactions used as positive examples |
| `consumer_cards_MDQ.parquet` | Consumer-card transactions used as negative examples and later scored for hidden-business risk |
| `merchants_reference.parquet` | Merchant lookup data used to enrich transactions, especially MCC and country information |

The preprocessing functions can reconcile several common alternatives for raw fields, including card ID (`card_number`, `card_id`, `pan`), amount, MCC, merchant ID, channel, tokenization, recurrence, country, card tier, and transaction timestamp.

## Outputs

### Predictions

`outputs/predictions/final_submission.csv` contains:

| Column | Description |
|---|---|
| `card_number` | Consumer-card identifier |
| `score` | Final hidden-business risk score from 0 to 1 |
| `risk_tier` | `LOW`, `MEDIUM`, or `HIGH` risk classification |

Risk tiers are assigned with the default thresholds:

| Tier | Default rule | Suggested handling |
|---|---|---|
| `LOW` | Score < 0.30 | Likely consumer behavior; no additional model-driven review by default |
| `MEDIUM` | 0.30 ≤ score < 0.70 | Ambiguous cases requiring review or soft compliance actions |
| `HIGH` | Score ≥ 0.70 | Strong business-like signal; prioritize for investigator review |

`outputs/predictions/high_risk_mcc_details.csv` provides MCC-level transaction summaries for consumer cards with scores of at least 0.70.

### Models and metrics

- `outputs/models/best_base_model.pkl`: selected tuned base classifier.
- `outputs/models/final_grey_refiner.pkl`: secondary LightGBM refiner, saved when suitable confident-sample data exists.
- `outputs/metrics/metrics.json`: best-model selection, per-model evaluation metrics, tuning results, feature list, dataset sizes, ensemble weights, and grey-zone configuration.

### Visual diagnostics

The notebook writes eight charts, including:

- Out-of-fold confusion matrix.
- Business-versus-consumer score distribution.
- ROC curve.
- Precision-recall curve.
- Feature-importance chart.
- Grey-zone before/after score analysis.
- Consumer score-component distributions.
- Ranked chart of the highest-scored consumer cards.

## Evaluation Metrics

The pipeline reports standard binary-classification metrics:

- **ROC-AUC**: ranking quality across classification thresholds; used as the primary model-selection metric.
- **Average Precision**: summarizes precision-recall performance and is particularly informative in imbalanced classification problems.
- **F1 score**: harmonic mean of precision and recall at the selected threshold.
- **Precision**: fraction of cards predicted as business-like that are actually business cards in labeled evaluation data.
- **Recall**: fraction of labeled business cards successfully detected.
- **Accuracy**: overall fraction of correct classifications.

Model comparison prioritizes ROC-AUC, using F1 as a tie-breaker.

## Configuration

Key behavior can be changed in the notebook configuration cell:

```python
RANDOM_STATE = 42

TUNE_CV_SPLITS = 3
EVAL_CV_SPLITS = 5

GREY_ZONE_LOW = 0.30
GREY_ZONE_HIGH = 0.70
CONF_BIZ_THRESH = 0.80

SMALL_AMOUNT = 2_000.0
LARGE_AMOUNT = 50_000.0
```

Before applying the model in a different environment, recalibrate score thresholds and review operational policies using representative data, expected investigator capacity, and the relative cost of false positives and false negatives.

## Limitations

- Large, regular, or category-concentrated consumer spending can resemble business behavior. For example, a legitimate customer financing a renovation, wedding, or other major event may receive an elevated score.
- The pipeline depends on the quality of merchant classification and MCC assignment. Incorrect merchant metadata can distort features and risk scores.
- Performance metrics on internal labeled data do not guarantee the same performance on new populations, time periods, countries, or product segments.
- The score is a prioritization signal rather than an automated adverse decision. Human review, monitoring, governance, and fairness assessment remain necessary.

## License and Data

No license is specified in the provided project materials. Add an appropriate license before distributing the code.

Transaction data and card identifiers are sensitive. Do not commit raw data, model outputs containing identifiable card numbers, or credentials to version control. Use access controls, encryption, masking or tokenization, retention policies, and applicable privacy/compliance procedures when operating this project.
