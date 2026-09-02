# House Price Prediction with Linear Regression

A simple **Multiple Linear Regression** model that predicts house sale prices
using three predictors: **square footage**, **number of bedrooms**, and
**number of bathrooms**. Built on Kaggle's
["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
dataset.

## Overview

| | |
|---|---|
| **Task** | Regression |
| **Target variable** | `SalePrice` |
| **Predictors** | Square footage, Bedrooms, Bathrooms |
| **Model** | `sklearn.linear_model.LinearRegression` |
| **Dataset** | Kaggle House Prices (`train.csv`, `test.csv`) |

## Features Used

| Feature | Source column(s) |
|---|---|
| `SquareFootage` | `GrLivArea` (above-ground living area) |
| `Bedrooms` | `BedroomAbvGr` |
| `TotalBathrooms` | `FullBath` + 0.5 × `HalfBath` |

## Project Structure

```
├── train.csv                          # Kaggle training data (with SalePrice)
├── test.csv                           # Kaggle hold-out data (no SalePrice)
├── House_Price_Regression.ipynb       # Full notebook: EDA, training, evaluation
├── submission.csv                     # Generated predictions for Kaggle
└── README.md
```

## Workflow

1. **Feature engineering** — combine `FullBath`/`HalfBath` into `TotalBathrooms`; rename columns for clarity.
2. **EDA** — correlation heatmap, pairplot, and scatter/bar plots against `SalePrice`.
3. **Outlier removal** — drop 2 known large-sqft/low-price outliers.
4. **Train/test split** — 80/20 split for validation.
5. **Model training** — fit `LinearRegression` on the 3 selected features.
6. **Evaluation** — MAE, MSE, RMSE, R² on train and test sets, plus 5-fold cross-validation.
7. **Diagnostics** — residual plot and predicted-vs-actual plot.
8. **Final prediction** — retrain on full `train.csv`, predict on Kaggle's `test.csv`, clip negative outputs, and export `submission.csv`.

## Results

| Metric | Train | Test |
|---|---|---|
| MAE | ~35,270 | ~37,257 |
| RMSE | ~48,382 | ~50,024 |
| R² | ~0.64 | ~0.55 |

Average 5-fold CV RMSE: **~48,511**

## Key Insight

`SquareFootage` is the strongest predictor of price. `Bedrooms` has a
**negative** coefficient when controlling for square footage — at a fixed
size, more bedrooms usually means smaller individual rooms, which doesn't
add value on its own.

## How to Run

1. Place `train.csv` and `test.csv` in the project root.
2. Open `House_Price_Regression.ipynb` in Jupyter, JupyterLab, VS Code, or Google Colab.
3. Run all cells top to bottom.
4. `submission.csv` will be generated in the project root.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
