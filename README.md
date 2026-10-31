# Online Shoppers Purchasing Intention — Classification

**M.Tech AIML | Machine Learning Assignment 2**

## 1. Problem statement

The objective is to predict whether an online shopping session results in a purchase (`Revenue=True`) or does not result in a purchase (`Revenue=False`) using session-level browsing behaviour and visitor information. This is a binary classification problem.

## 2. Dataset description

- **Dataset:** Online Shoppers Purchasing Intention
- **Source:** UCI Machine Learning Repository
- **Original records:** 12,330
- **Input features:** 17
- **Target:** `Revenue`
- **Target classes:** `True` (purchase) and `False` (no purchase)
- **Class distribution:** 10,422 no-purchase sessions (84.53%) and 1,908 purchase sessions (15.47%).
- **Missing values:** None in the supplied CSV.

The dataset contains browsing counts and durations, bounce/exit rates, page values, special-day proximity, month, operating system, browser, region, traffic type, visitor type and weekend information.

## 3. Repository contents

```text
online-shoppers-classification/
├── app.py
├── train_models.py
├── requirements.txt
├── .python-version
├── README.md
├── .gitignore
├── data/
│   └── online_shoppers_intention.csv
├── test_data.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metadata.json
└── results/
    ├── model_comparison.csv
    └── classification_reports.json
```

## 4. Models used

The five required classifiers are:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

### Preprocessing and experimental design

- A stratified 80:20 train-test split is used with `random_state=2026`.
- Numerical counts and measurements are median-imputed and standardised.
- Categorical fields, including integer-coded identifiers and Boolean `Weekend`, are most-frequent-imputed and one-hot encoded.
- Each classifier is stored in a scikit-learn `Pipeline` with its own preprocessing steps. This prevents train/test leakage and ensures the app applies exactly the same transformations.
- The same held-out test records are used to evaluate all five models.
- Models are trained by `train_models.py`; the Streamlit app only loads the saved pipelines and evaluates uploaded labelled test data.

### Evaluation metrics

Accuracy is the proportion of correct predictions. ROC-AUC measures the ability to rank purchase sessions above non-purchase sessions and uses the predicted probability of `Revenue=True`. Precision, recall and F1 are reported using **macro averaging**, so both classes contribute equally despite class imbalance. MCC summarises the confusion matrix in a single measure ranging from -1 to +1.

### Model comparison

All metrics below were calculated on the same stratified 20% held-out test split (2,466 records). Precision, Recall and F1 are macro-averaged; AUC is binary ROC-AUC for the positive (`Revenue=True`) class.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8771 | 0.8799 | 0.7982 | 0.6643 | 0.7018 | 0.4428 |
| Decision Tree | 0.8788 | 0.8311 | 0.7702 | 0.7508 | **0.7599** | **0.5207** |
| KNN | 0.8690 | 0.7982 | 0.7651 | 0.6585 | 0.6904 | 0.4100 |
| Gaussian Naive Bayes | 0.2810 | 0.7431 | 0.5784 | 0.5682 | 0.2806 | 0.1463 |
| Support Vector Machine | **0.8865** | 0.8746 | **0.8049** | 0.7137 | 0.7466 | 0.5105 |

### Observations on model performance

- **Logistic Regression:** Achieved the highest ROC-AUC (0.8799), indicating the strongest ranking ability for distinguishing purchase from non-purchase sessions among these experiments. Its macro recall (0.6643) is lower than the Decision Tree and SVM, so its threshold-based class predictions are less balanced.
- **Decision Tree:** Achieved the highest macro F1 (0.7599) and MCC (0.5207). It provides the strongest balance of precision and recall in this experiment, making it a strong choice when identifying both classes matters.
- **KNN:** Produced reasonable accuracy (0.8690), but its AUC (0.7982) and macro F1 (0.6904) were lower than the Logistic Regression, Decision Tree and SVM results.
- **Gaussian Naive Bayes:** Performed poorly with accuracy of 0.2810 and macro F1 of 0.2806. Its conditional-independence assumption may be unsuitable for the relationships among these browsing features. This result should be discussed rather than hidden.
- **Support Vector Machine:** Achieved the highest accuracy (0.8865) and macro precision (0.8049). Its macro F1 (0.7466) and MCC (0.5105) were slightly below the Decision Tree.

**Overall winner: Decision Tree Classifier**, when macro F1 and MCC are prioritised because the dataset is imbalanced and both purchase and non-purchase classes matter. The SVM has the highest accuracy, while Logistic Regression has the highest AUC. Therefore, the winner depends on the evaluation objective; the Decision Tree is selected here for the best balance of class-wise performance.

## 5. GitHub repository link

Add the URL of your own public GitHub repository here after creating it.

## 6. Streamlit application

The app supports:
- CSV upload for labelled test data
- Classifier selection dropdown
- Accuracy, ROC-AUC, macro precision, macro recall, macro F1 and MCC
- Confusion matrix
- Classification report
- Downloadable CSV containing predictions and purchase probabilities

The uploaded CSV must contain all 17 feature columns and the `Revenue` target. The app does not retrain models.

## 7. Run locally

Use Python 3.13 (the `.python-version` file specifies this for deployment).

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Upload `test_data.csv` in the app to verify the evaluation workflow. Run the training script before launching the app if the saved model files are not present.

## 8. Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud and sign in with GitHub.
3. Create a new app and select this repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Deploy. Streamlit installs dependencies from `requirements.txt`.
6. Open the deployed app and upload `test_data.csv` to verify that metrics and plots render.

Keep `app.py`, `requirements.txt`, `models/`, and `models/metadata.json` in the repository root structure shown above. The app uses paths relative to its own file, so it is compatible with cloud deployment.

## 9. References

- UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/


