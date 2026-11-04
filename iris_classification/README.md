# Iris Classification — Instructor Example (CSCI 6201, Fall 2026)

**Instructor demonstration, not a student/team submission.**

**Dataset:** [scikit-learn Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), 150 samples, four numeric features, three species. **Task:** Classification.

## Research questions
1. Can the four flower measurements predict species?
2. How does petal length differ among species?
3. Do petal length and width appear to separate the classes?

## Run in order (from repository root)
```powershell
python -m pip install -r requirements.txt
python iris_classification/01_load_iris.py
python iris_classification/02_explore_iris.py
python iris_classification/03_visualize_iris.py
python iris_classification/04_classify_iris.py
python iris_classification/05_test_iris.py
python iris_classification/06_search_sort_iris.py
python iris_classification/07_review_iris.py
python iris_classification/08_evaluate_iris.py
python iris_classification/09_report_iris.py
```

## Lab 2 progress
- Loaded and inspected the dataset; checked feature shapes, simple summaries, and missing values.
- Created `figures/01_petal_length_by_species.png`, which compares distributions of petal lengths across species. The Setosa group generally has shorter petals than the other groups.
- Created `figures/02_petal_length_vs_width.png`, which shows how the three species occupy different regions in petal measurement space, with some overlap between Versicolor and Virginica.
- Trained a Decision Tree using a reproducible stratified 80/20 split; see `results/preliminary_results.txt` for the actual test-set accuracy and confusion matrix.
- Wrote simple helper functions and used `assert` and `unittest` to check them.

## Week 10 progress: searching, sorting, and reviewing predictions
- `06_search_sort_iris.py` searches for petal lengths above a threshold and uses `sorted()`, `min()`, and `max()` to explore the measurements. Change the threshold to practice filtering.
- `07_review_iris.py` reuses the Week 9 Decision Tree settings and displays five actual versus predicted flower species, plus test accuracy. It does not overwrite previous results.
- **Final project connection:** searching and sorting help inspect data; checking predictions makes the evaluation easier to explain. After Quiz 2, explore misclassified flowers and add an error-analysis plot.

## Week 11 progress: evaluation, CSV files, and project reports
- `08_evaluate_iris.py`: reruns the Week 9 Decision Tree settings, saves each held-out prediction to `results/08_iris_predictions.csv`, and creates `figures/08_iris_correct_vs_incorrect.png`.
- `09_report_iris.py`: **reads that CSV** and builds `results/09_iris_project_report.txt` with accuracy, species-level counts, incorrect predictions, limitations, and next steps.
- **Run 08 before 09.** Open the CSV, PNG, and TXT outputs and compare them with your printed terminal results.
- **Class exercise:** trace the `correct` counter and explain why test accuracy is not a guarantee for future predictions.
- **Next steps:** compare a second classifier or different train/test split; describe any changed results.

## Next Steps for Final Project
- **Working:** data loading, plots, baseline model, and basic tests.
- **Needs improvement:** check stability across different train/test splits and interpret model errors.
- **Planned:** add an error-analysis figure, improve explanations, and compare a second classifier.
- **Instructor feedback requested:** Are the research questions focused enough, and does our evaluation avoid overclaiming?

## GitHub collaboration note
In a student team of two or three, each member should contribute meaningfully and commit their own work. These files were made as an instructor example and therefore do not show multiple student contributors.

## AI-use reminder
Follow the specific assignment's AI rules; understand and verify any AI-assisted code or writing, and disclose assistance when required. No AI use during closed assessments.
