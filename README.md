# CSCI 6201 — Fall 2026 example projects

Instructor demonstrations for [CSCI 6201](https://faculty.cs.gwu.edu/xiaodongqu/cs6201_f26/).

- [`iris_classification/`](iris_classification/README.md): primary classroom example (classification).
- [`diabetes_regression/`](diabetes_regression/README.md): parallel self-study example (regression; not for clinical use).

## Windows / GitHub Desktop setup
Open this repository on your computer, then open a terminal in its top-level folder.

```powershell
python -m pip install -r requirements.txt
python iris_classification/08_evaluate_iris.py
python iris_classification/09_report_iris.py
python diabetes_regression/08_evaluate_diabetes.py
python diabetes_regression/09_report_diabetes.py
```

Files 01–07 in each folder cover earlier weeks; see each project's README for the full run order.
For Week 11, **run file 08 before file 09**: file 08 saves predictions to CSV and a chart to PNG; file 09 **reads the CSV** and saves a text progress report. Inspect the generated `results/` and `figures/` folders.

Learning resources (original links):
- Think Python, 3rd edition: https://allendowney.github.io/ThinkPython/
- CS50 Introduction to Programming with Python: https://cs50.harvard.edu/python/

Students should understand and test their own projects; follow the course's AI-use and collaboration requirements.
