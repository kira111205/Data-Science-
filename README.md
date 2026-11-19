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

** Code **
import pandas as pd

# Load CSV files
races = pd.read_csv('races.csv')
results = pd.read_csv('results.csv')
drivers = pd.read_csv('drivers.csv')
constructors = pd.read_csv('constructors.csv')
circuits = pd.read_csv('circuits.csv')
status = pd.read_csv('status.csv')

# Store all tables in a dictionary
tables = {
    'races': races,
    'results': results,
    'drivers': drivers,
    'constructors': constructors,
    'circuits': circuits,
    'status': status
}

# Create a data audit
audit = pd.DataFrame([
    {
        'table': name,
        'rows': frame.shape[0],
        'columns': frame.shape[1],
        'duplicate_rows': int(frame.duplicated().sum()),
        'missing_cells': int(frame.isna().sum().sum())
    }
    for name, frame in tables.items()
]).sort_values('rows', ascending=False)

display(audit)

# Preview race results
print('Preview of race results:')
display(results.head())

# Display data types
print('Results data types:')
display(results.dtypes.to_frame(name='dtype'))




# Data loading and Audit 

races = load_csv('races.csv')
results = load_csv('results.csv')
drivers = load_csv('drivers.csv')
constructors = load_csv('constructors.csv')
circuits = load_csv('circuits.csv')
status = load_csv('status.csv')

tables = {
    'races': races,
    'results': results,
    'drivers': drivers,
    'constructors': constructors,
    'circuits': circuits,
    'status': status
}

audit = pd.DataFrame([
    {
        'table': name,
        'rows': frame.shape[0],
        'columns': frame.shape[1],
        'duplicate_rows': int(frame.duplicated().sum()),
        'missing_cells': int(frame.isna().sum().sum())
    }
    for name, frame in tables.items()
]).sort_values('rows', ascending=False)
display(audit)

print('Preview of race results:')
display(results.head())
print('Results data types:')
display(results.dtypes.to_frame(name='dtype'))


## Missing values 
missing = results.isna().mean().mul(100).sort_values(ascending=False)
missing = missing.rename('missing_percent').reset_index()
missing = missing.rename(columns={'index': 'column'})
display(missing)

missing_to_plot = missing[missing['missing_percent'] > 0]
if not missing_to_plot.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=missing_to_plot, x='missing_percent', y='column', color='#6C5CE7')
    plt.title('Missingness in Race Results')
    plt.xlabel('Missing values (%)')
    plt.ylabel('Column')
    plt.tight_layout()
    plt.show()
else:
    print('No missing values were found in results.csv.')


## DAta Cleaing 

for frame_name, frame in tables.items():
    date_columns = [
        col for col in frame.columns
        if 'date' in col.lower() or col.lower() == 'dob'
    ]
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors='coerce')
        print(f'{frame_name}.{column}: {frame[column].notna().sum():,} valid datetime values')

# Remove exact raw duplicates before enriching the fact table.
results = results.drop_duplicates().copy()

drivers_dim = drivers[['driverId', 'forename', 'surname', 'nationality', 'dob']].copy()
drivers_dim['driver_name'] = (
    drivers_dim['forename'].fillna('') + ' ' + drivers_dim['surname'].fillna('')
).str.strip()

constructors_dim = constructors[['constructorId', 'name', 'nationality']].rename(
    columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'}
)
races_dim = races[['raceId', 'year', 'round', 'circuitId', 'name', 'date']].rename(
    columns={'name': 'race_name', 'date': 'race_date'}
)
circuits_dim = circuits[['circuitId', 'name', 'location', 'country', 'lat', 'lng', 'alt']].rename(
    columns={'name': 'circuit_name'}
)

analysis_df = (
    results
    .merge(races_dim, on='raceId', how='left', validate='many_to_one')
    .merge(drivers_dim[['driverId', 'driver_name', 'nationality', 'dob']], on='driverId', how='left', validate='many_to_one')
    .merge(constructors_dim, on='constructorId', how='left', validate='many_to_one')
    .merge(circuits_dim, on='circuitId', how='left', validate='many_to_one')
    .merge(status[['statusId', 'status']], on='statusId', how='left', validate='many_to_one')
)

# Grid value zero represents a pit-lane or unrecorded start, never pole position.
analysis_df['grid_clean'] = analysis_df['grid'].replace(0, np.nan)
analysis_df['finish_position'] = analysis_df['positionOrder'].replace(0, np.nan)
analysis_df['points_finish'] = (analysis_df['positionOrder'] <= 10).astype(int)
analysis_df['finished_classified'] = analysis_df['position'].notna().astype(int)
analysis_df['race_month'] = analysis_df['race_date'].dt.month
analysis_df['driver_age'] = (analysis_df['race_date'] - analysis_df['dob']).dt.days / 365.25

print(f'Analytical table dimensions: {analysis_df.shape[0]:,} rows x {analysis_df.shape[1]} columns')
display(analysis_df[[
    'year', 'race_name', 'driver_name', 'constructor_name', 'grid_clean',
    'positionOrder', 'points', 'status'
]].head())

# Driver and Constructor Perspectives

driver_summary = (
    analysis_df.groupby('driver_name', as_index=False)
    .agg(
        starts=('resultId', 'size'),
        total_points=('points', 'sum'),
        wins=('positionOrder', lambda values: (values == 1).sum()),
        podiums=('positionOrder', lambda values: (values <= 3).sum()),
        top10_rate=('points_finish', 'mean')
    )
    .query('starts >= 20')
    .sort_values('total_points', ascending=False)
)

constructor_summary = (
    analysis_df.groupby('constructor_name', as_index=False)
    .agg(
        starts=('resultId', 'size'),
        total_points=('points', 'sum'),
        wins=('positionOrder', lambda values: (values == 1).sum()),
        top10_rate=('points_finish', 'mean')
    )
    .query('starts >= 20')
    .sort_values('total_points', ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
sns.barplot(data=driver_summary.head(15), y='driver_name', x='total_points', color='#E17055', ax=axes[0])
axes[0].set_title('Top Drivers by Career Points')
axes[0].set_xlabel('Career points')
axes[0].set_ylabel('Driver')

sns.barplot(data=constructor_summary.head(15), y='constructor_name', x='total_points', color='#6C5CE7', ax=axes[1])
axes[1].set_title('Top Constructors by Career Points')
axes[1].set_xlabel('Career points')
axes[1].set_ylabel('Constructor')
plt.tight_layout()
plt.show()

top_constructors = constructor_summary.head(8)['constructor_name'].tolist()
annual_constructor = (
    analysis_df[analysis_df['constructor_name'].isin(top_constructors)]
    .groupby(['year', 'constructor_name'], as_index=False)['points'].sum()
)
plt.figure(figsize=(14, 6))
sns.lineplot(data=annual_constructor, x='year', y='points', hue='constructor_name', linewidth=2)
plt.title('Season Points Trajectories for Leading Constructors')
plt.xlabel('Season')
plt.ylabel('Points')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
status_order = analysis_df['status'].value_counts().head(10).index
sns.countplot(data=analysis_df, y='status', order=status_order, color='#00CEC9')
plt.title('Ten Most Common Result Statuses')
plt.xlabel('Result count')
plt.ylabel('Status')
plt.tight_layout()
plt.show()










`<ME24B1010, ME24B1018, ME24B1024>` | `<IDSE/ IIITDM>` | `<Date>`ce-
