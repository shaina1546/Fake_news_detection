# Dataset Instructions: Fake and Real News Dataset

## 1. Dataset Source
This application is designed to train on the popular **Fake and Real News Dataset** available on Kaggle:
- **Source Link:** [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Author:** Clément Bisaillon
- **Contents:**
  - `Fake.csv`: Articles labeled as fake news (contains columns: `title`, `text`, `subject`, `date`).
  - `True.csv`: Articles verified as real news from Reuters (contains columns: `title`, `text`, `subject`, `date`).

## 2. Setup Instructions
To train the machine learning model on your local machine:
1. Navigate to the dataset page on Kaggle: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Click **Download** (requires a free Kaggle account).
3. Unzip the downloaded file (`archive.zip`).
4. Place `Fake.csv` and `True.csv` directly into this `data/` directory:
   ```
   fake_news_detection/
   └── data/
       ├── Fake.csv
       ├── True.csv
       ├── README.md
       └── .gitkeep
   ```

*Alternative:* The project also supports a unified single CSV file (e.g. `news.csv`) containing `text` (or `title` + `text`) and a `label` column (e.g., `0` / `1` or `'fake'` / `'real'`).

## 3. Why Datasets Are Excluded from Git (`.gitignore`)
- **Repository Size Limits:** GitHub enforces strict file size limits (100MB per file) and warns on files larger than 50MB. `Fake.csv` and `True.csv` together exceed 100MB uncompressed.
- **Git Performance:** Committing large binary or text files slows down cloning, pushing, and pulling for all collaborators.
- **Reproducibility & Licensing:** Best practices dictate keeping raw data external or downloaded via verifiable links, while preserving code and schemas in version control.
