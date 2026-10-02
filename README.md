# Seattle Weather Classification

A machine learning project that uses historical Seattle weather data to classify daily weather conditions with a **Random Forest Classifier**.

## Overview

This project loads the Seattle weather dataset, performs basic data preparation and feature engineering, encodes the weather labels, trains a Random Forest classification model, and evaluates its performance.

The dataset contains **1,461 daily observations** with the following original variables:

- `date`
- `precipitation`
- `temp_max`
- `temp_min`
- `wind`
- `weather`

The weather classes are:

- `drizzle`
- `fog`
- `rain`
- `snow`
- `sun`

The supplied workflow reports an overall test accuracy of approximately **88.36%**.

## Dataset

The project reads the dataset from:

```text
seattle-weather.csv
```

The data covers dates from **2012-01-01 through 2015-12-31** and contains **1,461 rows and 6 original columns**.

The original dataset columns are:

| Column | Description |
| --- | --- |
| `date` | Date of the observation |
| `precipitation` | Recorded precipitation |
| `temp_max` | Maximum temperature |
| `temp_min` | Minimum temperature |
| `wind` | Wind measurement |
| `weather` | Weather classification label |

## Data Preparation

The project performs the following preprocessing steps:

1. Loads the CSV data with pandas.
2. Creates a copy of the dataset.
3. Checks for missing values.
4. Converts `date` to a pandas datetime value.
5. Extracts date-based features:
   - `month`
   - `year`
   - `day`
   - `day_of_yr`
6. Creates two lag features from `temp_max`:
   - `temp_lag1` — previous day's maximum temperature
   - `temp_lag2` — maximum temperature from two days earlier
7. Removes rows containing missing values created by the lag features.

The supplied workflow reports that after removing the initial rows affected by the lag features, the dataset contains **1,459 rows**.

The original dataset contains no missing values in its six source columns; the missing values removed during preprocessing are introduced by the lag-feature calculations.

## Features

The model uses the following input features:

- `precipitation`
- `temp_min`
- `wind`
- `month`
- `year`
- `day`
- `day_of_yr`
- `temp_lag1`
- `temp_lag2`
- `temp_max`

The prediction target is:

```text
weather
```

### Target Encoding

The target labels are encoded using `LabelEncoder`.

The classes reported by the supplied workflow are:

```python
['drizzle', 'fog', 'rain', 'snow', 'sun']
```

The encoded labels therefore correspond to:

| Encoded Value | Weather Class |
| ---: | --- |
| `0` | `drizzle` |
| `1` | `fog` |
| `2` | `rain` |
| `3` | `snow` |
| `4` | `sun` |

## Model

The project uses scikit-learn's `RandomForestClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
```

The data is split into training and test sets using:

```python
train_test_split(
    x,
    y_encoded,
    test_size=0.2,
    random_state=42
)
```

This results in a **test set containing 292 observations**.

## Evaluation

### Accuracy

The reported test accuracy is:

**88.36%**

```text
Accuracy: 0.8835616438356164
```

### Classification Report

The supplied project output reports the following classification metrics:

| Weather | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `drizzle` | 1.00 | 0.14 | 0.25 | 7 |
| `fog` | 0.85 | 0.42 | 0.56 | 26 |
| `rain` | 0.97 | 0.93 | 0.95 | 123 |
| `snow` | 1.00 | 0.67 | 0.80 | 6 |
| `sun` | 0.81 | 0.98 | 0.89 | 130 |
| **Accuracy** | — | — | **0.88** | **292** |
| **Macro avg** | **0.93** | **0.63** | **0.69** | **292** |
| **Weighted avg** | **0.89** | **0.88** | **0.87** | **292** |

### Confusion Matrix

The reported confusion matrix is:

```text
[[  1   0   0   0   6]
 [  0  11   0   0  15]
 [  0   0 115   0   8]
 [  0   0   2   4   0]
 [  0   2   1   0 127]]
```

The matrix follows the label order:

```text
drizzle
fog
rain
snow
sun
```

In this evaluation, the matrix shows that the largest number of test observations belong to the `sun` and `rain` classes, while several `drizzle` and `fog` observations are classified as `sun`.

## Requirements

The project uses Python and the following main libraries:

- pandas
- scikit-learn

Install the required packages with:

```bash
pip install pandas scikit-learn
```

## Usage

Place `seattle-weather.csv` in the project directory and run the Python notebook or script containing the workflow.

### Core Workflow

```python
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load data
df = pd.read_csv("seattle-weather.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Date features
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day"] = df["date"].dt.day
df["day_of_yr"] = df["date"].dt.dayofyear

# Lag features
df["temp_lag1"] = df["temp_max"].shift(1)
df["temp_lag2"] = df["temp_max"].shift(2)

# Remove rows with lag-related missing values
df = df.dropna()

# Features and target
x = df[
    [
        "precipitation",
        "temp_min",
        "wind",
        "month",
        "year",
        "day",
        "day_of_yr",
        "temp_lag1",
        "temp_lag2",
        "temp_max"
    ]
]

y = df["weather"]

# Encode target
LE = LabelEncoder()
y_encoded = LE.fit_transform(y)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Predict
predictions = model.predict(x_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, predictions))

print(
    classification_report(
        y_test,
        predictions,
        target_names=LE.classes_
    )
)
```

## Project Structure

A simple project structure can be:

```text
project/
├── README.md
├── seattle-weather.csv
└── weather_prediction.py
```

If the work is kept in a notebook:

```text
project/
├── README.md
├── seattle-weather.csv
└── weather_prediction.ipynb
```

## Results Summary

The Random Forest model achieved **88.36% accuracy** on the reported test set of **292 observations**.

The supplied classification report shows:

- `rain` — precision **0.97**, recall **0.93**, F1-score **0.95**
- `sun` — precision **0.81**, recall **0.98**, F1-score **0.89**
- `snow` — precision **1.00**, recall **0.67**, F1-score **0.80**
- `fog` — precision **0.85**, recall **0.42**, F1-score **0.56**
- `drizzle` — precision **1.00**, recall **0.14**, F1-score **0.25**

The results above are reproduced from the supplied project output.

## Workflow Summary

The overall machine learning workflow is:

```text
Seattle Weather CSV
        |
        v
Load Dataset with pandas
        |
        v
Check Missing Values
        |
        v
Convert Date to Datetime
        |
        v
Create Date Features
(month, year, day, day_of_yr)
        |
        v
Create Temperature Lag Features
(temp_lag1, temp_lag2)
        |
        v
Remove Rows with Lag-related NaN Values
        |
        v
Select Features and Target
        |
        v
Encode Weather Labels
        |
        v
Train/Test Split
        |
        v
Random Forest Classifier
        |
        v
Predictions
        |
        v
Model Evaluation
(Accuracy, Classification Report,
Confusion Matrix)
```

## Notes

- The two lag features are created from `temp_max` using pandas `shift()`.
- Because `temp_lag1` and `temp_lag2` require previous observations, the first two rows contain missing lag values and are removed by `dropna()`.
- The supplied workflow uses `random_state=42` for both the train/test split and Random Forest model.
- The reported results are specific to the supplied dataset, preprocessing steps, feature selection, and train/test split.
- The project is intended as a machine learning classification workflow for educational and experimental purposes.

## Conclusion

This project demonstrates an end-to-end weather classification workflow using historical Seattle weather data. It covers data loading, validation, date-based feature engineering, lag-feature creation, target encoding, model training, prediction, and evaluation with accuracy, classification metrics, and a confusion matrix.

The supplied Random Forest workflow reports a test accuracy of **88.36%**.
