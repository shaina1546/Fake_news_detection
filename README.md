# Fake News Detection System Using Machine Learning

An end-to-end, reproducible Natural Language Processing (NLP) web application built with **Python 3.12**, **scikit-learn**, and **Streamlit**. The system classifies news headlines and articles based on linguistic and stylistic patterns learned from labeled datasets.

---

## 1. Problem Statement
The exponential spread of online misinformation, sensationalism, and manufactured news stories impacts public discourse and institutional trust. While human editorial oversight and investigative reporting remain the gold standards of truth verification, automated Natural Language Processing (NLP) tools can aid readers by analyzing textual style, sensationalized diction, and linguistic markers characteristic of unverified reporting.

---

## 2. Project Objectives
- **Demonstrate Core NLP & ML Concepts:** Feature extraction with TF-IDF and linear classification with Logistic Regression.
- **Maintain Rigorous Evaluation Standards:** Prevent data leakage through unified scikit-learn pipelines and stratified holdout evaluation.
- **Prioritize Ethical Transparency:** Clearly distinguish statistical style-based predictions from factual verification of real-world truth.
- **Provide a Robust, Clean Web Interface:** Deliver a Streamlit application that initializes gracefully even before training artifacts exist.
- **Maintain Production-Grade Software Quality:** 100% automated test coverage with `pytest`, clean directory structures, and git hygiene.

---

## 3. Technology Stack
- **Language:** Python 3.12 (also tested on 3.14)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & NLP:** `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`)
- **Model Serialization:** `joblib`
- **Visualizations:** `matplotlib`
- **Web Application Framework:** `streamlit`
- **Testing Framework:** `pytest`
- **Version Control & Hosting:** Git, GitHub
- **Deployment Platform:** Streamlit Community Cloud

---

## 4. Project Structure
```text
fake_news_detection/
│
├── app.py                     # Streamlit web application entry point
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive documentation and guides
├── PROJECT_PROGRESS.md        # Stage-by-stage implementation checklist
├── .gitignore                 # Exclusions for virtual envs, datasets, and caches
├── .python-version            # Specifies Python 3.12 for cloud platforms
│
├── data/                      # Dataset directory (excluded from git)
│   ├── README.md              # Instructions for downloading Kaggle dataset
│   └── .gitkeep               # Preserves empty folder in repository
│
├── models/                    # Serialized machine learning models
│   └── .gitkeep
│
├── reports/                   # Performance reports and confusion matrix plots
│   └── .gitkeep
│
├── src/                       # Core modular application logic
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, validation, leakage detection
│   ├── text_preprocessor.py   # Text normalization and whitespace cleaning
│   ├── train_model.py         # Pipeline fitting, evaluation, and artifact export
│   ├── predict.py             # Inference module, validation, and disclaimers
│   └── evaluation.py          # Metric calculations and confusion matrix plots
│
└── tests/                     # Automated unit and integration tests
    ├── test_app_startup.py    # Tests Streamlit UI startup without models
    ├── test_data_loader.py    # Tests loading, validation, and leakage detection
    ├── test_evaluation.py     # Tests metric computation and report exports
    ├── test_prediction.py     # Tests inference validation and probability handling
    └── test_preprocessing.py  # Tests text cleaning and boundary edge cases
```

---

## 5. Dataset Source and Setup

### Source
This system is configured to train on the **Fake and Real News Dataset** on Kaggle:
- **Kaggle URL:** [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Files:** `Fake.csv` (~23,481 articles) and `True.csv` (~21,417 articles from Reuters).

### Setup Instructions
1. Download `archive.zip` from Kaggle.
2. Extract the files into the `data/` folder:
   ```text
   fake_news_detection/data/Fake.csv
   fake_news_detection/data/True.csv
   ```
3. *Single CSV Alternative:* The loader also accepts any single CSV (e.g. `news.csv`) containing `text` (or `title` + `text`) and a `label` column (0 for Real, 1 for Fake).

> **Why Datasets Are Excluded from Git:** Raw datasets exceed 100 MB and violate GitHub repository limits. The `.gitignore` file automatically excludes `data/*.csv`.

---

## 6. Installation & Environment Setup (Windows Command Prompt)

Follow these beginner-friendly steps in **Command Prompt (`cmd`)** or **PowerShell**:

### Step 1: Open Terminal in Project Folder
```cmd
cd path\to\fake_news_detection
```

### Step 2: Verify Python Version
```cmd
py -3.12 --version
```
*(If you do not have the py launcher, use `python --version`)*

### Step 3: Create a Virtual Environment
A virtual environment isolates this project's packages from other Python projects:
```cmd
py -3.12 -m venv .venv
```

### Step 4: Activate the Virtual Environment
- **Command Prompt (`cmd`):**
  ```cmd
  .venv\Scripts\activate
  ```
- **PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
*(When active, you will see `(.venv)` at the beginning of your prompt)*

### Step 5: Upgrade pip and Install Dependencies
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Model Training Instructions

Once your dataset is placed in `data/`, run the reproducible training module:
```cmd
python -m src.train_model
```

### What happens during training:
1. Loads and validates `Fake.csv` and `True.csv`.
2. Checks for missing values and drops corrupt entries.
3. Detects exact duplicates and cross-class duplicate leakage.
4. Splits data into **80% Training** and **20% Holdout Testing** using stratified sampling.
5. Fits a scikit-learn pipeline:
   - `TfidfVectorizer` (unigrams + bigrams, English stop words, max 10,000 features).
   - `LogisticRegression` (L2 penalty, balanced class weights, fixed random seed `42`).
6. Evaluates exclusively on the holdout test set.
7. Saves the trained pipeline to `models/fake_news_model.joblib`.
8. Saves evaluation metrics to `reports/metrics_summary.json` and `reports/confusion_matrix.png`.

---

## 8. Running Automated Tests

Run the complete test suite using `pytest`:
```cmd
python -m pytest tests/ -v
```
All tests use small synthetic in-memory fixtures and do not depend on the large Kaggle dataset.

---

## 9. Launching the Streamlit Web Application

Start the local web application:
```cmd
streamlit run app.py
```
Your browser will open automatically at:
```
http://localhost:8501
```

### Application Sections:
- **🏠 Home:** Overview, how the system works, and ethical guidance.
- **🔍 News Analysis:** Interactive text area with preloaded real and fake sample buttons, real-time prediction, and class probability estimation.
- **📊 Model Information:** Real evaluation metrics (Accuracy, Precision, Recall, F1), confusion matrix visualization, and raw JSON report inspector.
- **ℹ️ About Project:** Problem description, technology stack, limitations, and future work.

---

## 10. Model Evaluation Methodology

- **Holdout Test Set:** 20% stratified test split evaluated strictly after training.
- **Target Metrics:**
  - **Accuracy:** Proportion of all correct classifications.
  - **Precision (Fake):** Percentage of flagged articles that were genuinely fake.
  - **Recall (Fake):** Percentage of total fake articles detected.
  - **F1-Score (Fake):** Harmonic mean of Precision and Recall.
- **Confusion Matrix:** Measures True Negatives (Real as Real), False Positives (Real flagged as Fake), False Negatives (Fake missed as Real), and True Positives (Fake flagged as Fake).

---

## 11. Ethical Considerations & Limitations

1. **Style ≠ Truth:** Machine learning models evaluate lexical similarity to training patterns. They cannot independently verify historical reality or consult external evidence.
2. **Temporal & Topic Bias:** Articles discussing topics outside the dataset distribution (e.g., breaking news, sports, entertainment) may exhibit lower classification reliability.
3. **Satire & Parody:** Satirical publications (e.g., *The Onion*) use exaggerated writing styles for humor, which style-based models often flag as fake.
4. **Adversarial Phrasing:** Misinformation intentionally written in dry, formal journalistic phrasing can deceive linear classifiers.
5. **No Permanent Logging:** For privacy, the application does not store or log user-submitted texts.

---

## 12. Git & GitHub Setup Guide

Follow these commands to initialize and stage your project for GitHub:

### Step 1: Check Git Status
```cmd
git status
```
If Git is not yet initialized in this folder:
```cmd
git init
```

### Step 2: Add Files to Staging
```cmd
git add .
```
Verify that `.venv/` and `data/*.csv` are ignored:
```cmd
git status
```

### Step 3: Create Initial Commit
```cmd
git commit -m "feat: complete fake news detection system with scikit-learn and streamlit"
```

### Step 4: Rename Branch to `main`
```cmd
git branch -M main
```

### Step 5: Connect to GitHub
1. Open your web browser and go to [https://github.com/new](https://github.com/new).
2. Create a new repository (e.g. `fake-news-detection`). Leave "Initialize with README" **unchecked**.
3. Copy your repository URL from GitHub (e.g., `https://github.com/YOUR_USERNAME/fake-news-detection.git`).
4. Link your local repository:
   ```cmd
   git remote add origin https://github.com/YOUR_USERNAME/fake-news-detection.git
   ```

### Step 6: Push Code to GitHub
```cmd
git push -u origin main
```

---

## 13. Streamlit Community Cloud Deployment Guide

Streamlit Community Cloud lets you deploy and share your app for free:

### Prerequisites:
- Your project is committed and pushed to a public or private GitHub repository.

### Step-by-Step Deployment:
1. Navigate to [https://share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.
2. Click **New app**.
3. Select your GitHub repository: `YOUR_USERNAME/fake-news-detection`.
4. Set **Branch** to `main`.
5. Set **Main file path** to `app.py`.
6. Click **Deploy!**.

### Handling the Model Artifact in Cloud Deployment:
- **Crash-Free Startup:** Even if no model is committed, `app.py` boots cleanly and displays an informative banner guiding visitors.
- **Enabling Live Predictions on Cloud:**
  - After training locally (`python -m src.train_model`), the resulting `models/fake_news_model.joblib` is ~3 MB (compact due to `max_features=10000`).
  - If you wish to enable instant predictions on your deployed app, you can commit this small model file:
    ```cmd
    git add -f models/fake_news_model.joblib models/model_metadata.json reports/metrics_summary.json reports/confusion_matrix.png
    git commit -m "chore: add trained model artifact and evaluation reports for cloud deployment"
    git push origin main
    ```
  - Streamlit Community Cloud will automatically detect the push and reboot with live predictions enabled!

---

## 14. Troubleshooting Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: Directory '.../data' must contain...` | Dataset not downloaded | Place `Fake.csv` and `True.csv` from Kaggle into `data/`. |
| `Trained model artifact not found` in Streamlit | Model not trained yet | Run `python -m src.train_model` in terminal, then refresh the app. |
| `ModuleNotFoundError: No module named 'src'` | Running script outside root | Run commands from the project root using `-m`, e.g. `python -m src.train_model`. |
| `streamlit is not recognized` | Virtual environment not active | Activate virtual environment: `.venv\Scripts\activate`. |

---

## 15. License
This educational project is distributed under the MIT License. Datasets used are subject to their respective original licenses on Kaggle.
