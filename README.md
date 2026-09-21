# Concrete Compressive Strength Prediction

**Comparing regression models and exploring which inputs matter.**

A team coursework project for the 2023–24 Computational Challenge. The notebooks investigate how eight inputs describing concrete composition and age relate to compressive strength, measured in MPa.

**Team:** Douglas Penning, Roman Ogorodnov, Manish Saharan and Ryan Hutchings.

[Start with the main analysis](Code.1.ipynb) · [View the dataset](Concrete_Data_Yeh_final.csv)

## Project overview

The workflow combines data cleaning, an 80/20 train–test split, regression comparison, hyperparameter tuning and interactive exploration of feature removal.

| Stage | Approach |
| --- | --- |
| Preprocessing | Duplicate removal, mean imputation and standard scaling; preprocessing fitted to the training split |
| Model comparison | Linear regression, decision tree, random forest, ridge, lasso, elastic net, support vector regression and k-nearest neighbours |
| Evaluation | RMSE, MAE and R² |
| Tuning | Randomised search followed by grid search for the random forest |
| Interpretation | Removing input combinations and visualising the change in prediction errors |
| Communication | Comparison plots, interactive widgets, learning curves and exported spreadsheets |

**Reported coursework finding:** the team selected random forest and identified age as the most influential input in its feature-removal analysis. These are findings from this dataset and workflow, not independently validated performance claims.

## Explore the work

| File | Contents |
| --- | --- |
| [Code.1.ipynb](Code.1.ipynb) | Main analysis, explanations and interactive visualisations |
| [Code_hyperparameters.ipynb](Code_hyperparameters.ipynb) | Supporting hyperparameter experiments |
| [Scaler_Imputer.ipynb](Scaler_Imputer.ipynb) | Supporting preprocessing experiments |
| [Concrete_Data_Yeh_final.csv](Concrete_Data_Yeh_final.csv) | Input dataset |
| [regression_performance.xlsx](regression_performance.xlsx) | Saved model-comparison output |

## Local setup

The main notebook records Python 3.9.12. Its imports and spreadsheet exports require the following packages:

```bash
python -m pip install jupyterlab pandas numpy scipy scikit-learn matplotlib seaborn ipywidgets openpyxl
python -m jupyterlab
```

Open `Code.1.ipynb` from the repository folder so it can find `Concrete_Data_Yeh_final.csv`. Restart the kernel and run the cells in order. Execution includes repeated model fitting and writes spreadsheet outputs into the working folder. Interactive controls require a live notebook session.

Dependencies are not version-pinned, so compatibility with the latest packages has not been verified.

## Evaluation context

This is educational, collaborative work. Model and feature selection reuse the evaluation split, so the reported scores should not be treated as an untouched final benchmark. A stronger follow-up would use cross-validation for selection, fit preprocessing within each fold, and reserve a separate test set for final evaluation.

Random-forest fits are not consistently seeded, so results can vary between runs. See the notebook for the team's original reasoning and references.
