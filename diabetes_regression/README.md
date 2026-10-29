# Diabetes Regression — Instructor Example (CSCI 6201, Fall 2026)

**Instructor demonstration, not a student/team submission.** This dataset is for software education, not for clinical diagnosis or decision-making.

**Dataset:** [scikit-learn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html), 442 samples, 10 numeric (preprocessed) features, continuous progression target. **Task:** Regression.

## Research questions
1. Can baseline features predict the numeric progression target?
2. Is the standardized BMI feature related to the target?
3. What level of error does a simple Linear Regression baseline have?

## Run in order (from repository root)
```powershell
python -m pip install -r requirements.txt
python diabetes_regression/01_load_diabetes.py
python diabetes_regression/02_explore_diabetes.py
python diabetes_regression/03_visualize_diabetes.py
python diabetes_regression/04_regress_diabetes.py
python diabetes_regression/05_test_diabetes.py
python diabetes_regression/06_search_sort_diabetes.py
python diabetes_regression/07_review_diabetes.py
```

## Lab 2 progress
- Loaded the dataset, inspected shapes and feature names, checked missing values, and summarized the numerical target.
- Created `figures/01_target_distribution.png`, showing that the target values vary substantially across observations.
- Created `figures/02_bmi_vs_target.png`, suggesting an association between the **standardized** BMI feature and the target; this plot does not establish causation.
- Trained a Linear Regression model with an 80/20 train/test split; see `results/preliminary_results.txt` for test MAE and R².
- Used simple functions, assertions, and unit tests.

## Week 10 progress: searching, sorting, and reviewing predictions
- `06_search_sort_diabetes.py` finds target values above a threshold and uses `sorted()`, `min()`, and `max()` to inspect the dataset. Change the threshold to practice filtering.
- `07_review_diabetes.py` reuses the Week 9 Linear Regression settings and shows five actual versus predicted targets, their absolute errors, and test MAE. It does not overwrite previous results.
- **Final project connection:** searching and sorting help explore target values; the error examples make model evaluation easier to explain. After Quiz 2, examine residuals and create an error-analysis plot.

## Next Steps for Final Project
- **Working:** exploration, two plots, regression baseline, and tests.
- **Needs improvement:** more systematic investigation of errors and input-feature effects.
- **Planned:** interpret residuals, compare against a simple baseline, and improve documentation.
- **Instructor feedback requested:** Are the chosen visualizations sufficient to motivate the modeling question?

## GitHub collaboration note
Real student teams should document individual contributions and commits. This repository is an instructor-authored reference example.

## AI-use reminder
Follow written assignment-level permissions, verify all suggested code, and do not submit unexplained or unauthorized AI-generated work.
