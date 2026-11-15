# Data-Science


# Formula 1 Top-10 Finish Prediction

Predicting whether a driver will finish in the top 10 of a Formula 1 race using only information available **before** the race starts.



## 1. Problem Statement

**Question:** Can pre-race information (grid position, driver form, constructor form, circuit) predict whether a driver finishes in the top 10?

**Goal:** `<A Formula 1 result is a compact record of speed, reliability, strategy, and occasionally a safety car with impeccable comic timing>`

**Why it matters:** `<A top-10 finish is used as a consistent operational target. Historical Formula 1 points systems have changed, so it should not be interpreted as an exact historical points-scoring definition in every season.>`

**Target variable:** `points_finish = 1` if the final finishing position is 10 or better, otherwise `0`. Note: historical points systems differ by era, so this is a consistent proxy, not the exact points rule of every season.

---

## 2. Dataset

- **Source:** Formula 1 World Championship (1950 - 2026) : `<https://results-e3dfd2.tiiny.site>`
- **Tables used:** `results`, `races`, `drivers`, `constructors`, `circuits`, `status`
- **Excluded on purpose:** lap times and pit stops, because they are only known during or after the race and would cause data leakage.


## 3. Project Structure

```
f1-top10-prediction/
├── README.md
├── requirements.txt
├── data/                          # raw CSV files (not committed if large)
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   └── 02_modelling_evaluation.ipynb
├── src/
│   ├── data_prep.py
│   ├── features.py
│   └── models.py
└── reports/
    └── figures/
```

---

## 4. Method

### 4.1 Data preparation
- Parsed dates and removed duplicate rows.
- Joined tables with many-to-one merges to avoid row duplication.
- Treated `grid = 0` as a pit-lane start (set to missing, not pole position).


### 4.2 Exploratory analysis
Key findings

### 4.3 Feature
| Feature | Description |
|---|---|
| `grid_clean` | 
| `grid_percentile_in_race` | 
| `driver_prior_top10_rate` | Driver's top-10 rate over **earlier** races only |
| `constructor_prior_top10_rate` | Constructor's top-10 rate over **earlier** races only |
| `driver_age`, `year`, `round`, `race_month`, `alt`, `circuit_country` | Context features |


### 4.4 Avoiding data leakage
- Data is sorted chronologically before building history features.
- History features use `shift(1)` so a race never sees its own result.
- Constructor history is aggregated per race so team-mates cannot leak into each other.
- Train, validation and test sets are split **by date**, never randomly.

### 4.5 Models
| Model | Role | Why chosen |
|---|---|---|
| `<Logistic Regression / majority class>` | Baseline | `<reason>` |
| Random Forest | Approach 1 | `<reason>` |
| `<Gradient Boosting>` | Approach 2 | `<reason>` |

Tuning: `<cross-validation method, e.g. time-aware CV, and parameters searched>`

---

## 5. Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | `<>` | `<>` | `<>` | `<>` | `<>` |
| Random Forest | `<>` | `<>` | `<>` | `<>` | `<>` |
| `<Model 3>` | `<>` | `<>` | `<>` | `<>` | `<>` |

### Interpretation
`<Which model is best and why? Why did you choose these metrics given class imbalance?>`

### Error analysis
`<Where does the model fail? e.g. by grid band, era, constructor, or retirements. What do the errors tell you?>`

### Feature importance
`<Which features matter most, and does that make sense?>`

---

## 6. Limitations and Next Steps

**Limitations**
- Top-10 is not identical to the points rule in every era.
- No weather, tyre strategy, penalties, safety cars or car upgrades.
- Mechanical retirements are largely unpredictable from pre-race data.
- `<add your own>`

**Next steps**
- Build season-specific points targets.
- Add qualifying and practice data with strict timing controls.
- Calibrate predicted probabilities.
- Model retirement risk separately.

---


`<ME24B1018, ME24B1024>` | `<Course / Institution>` | `<Date>`ce-
