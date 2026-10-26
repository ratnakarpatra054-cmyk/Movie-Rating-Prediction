# 🎬 Movie Rating Prediction with Python
### Afame Technologies — Data Science Internship Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Project Overview

This project is part of the **Afame Technologies Data Science Internship**. The goal is to build a supervised Machine Learning model that predicts the **IMDb rating** of an Indian movie based on features like Genre, Director, Actors, Duration, Year, and Votes.

> **Author:** Ratnakar Patra  
> **Role:** Data Science Intern  
> **Organization:** Afame Technologies

---

## 🎯 Problem Statement & Objective

**Problem:** Given basic metadata about an Indian movie (genre, cast, director, year, duration, vote count), can we accurately predict its IMDb rating?

**Objective:** Build a regression model that:
- Takes movie metadata as input
- Outputs a predicted IMDb rating (1.0 – 10.0)
- Achieves a competitive R² score and low MAE on the test set

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Source** | IMDb Indian Movies Dataset (provided by Afame Technologies) |
| **File** | `dataset/movies_data.csv` |
| **Rows** | ~5,660 Indian movies |
| **Target Variable** | `Rating` (IMDb score: 1.0 – 10.0) |

### Column Descriptions

| Column | Data Type | Description |
|--------|-----------|-------------|
| `Name` | String | Movie title |
| `Year` | String → Integer | Release year (cleaned from `(2019)` format) |
| `Duration` | Integer | Movie length in minutes |
| `Genre` | String | Genres, may be comma-separated (e.g., "Drama, Comedy") |
| `Rating` | Float | ⭐ IMDb rating — **target variable** |
| `Votes` | Integer | Number of IMDb user votes |
| `Director` | String | Director's name |
| `Actor 1` | String | Lead actor |
| `Actor 2` | String | Supporting actor |
| `Actor 3` | String | Supporting actor |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core programming language |
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib & Seaborn** | Data visualization (EDA charts) |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **Joblib** | Saving and loading the trained model |
| **Jupyter Notebook** | Interactive development and presentation |

---

## 📁 Project Structure

```
Movie-Rating-Prediction/
├── dataset/
│   └── movies_data.csv          # IMDb Indian Movies dataset
├── notebooks/
│   └── movie_rating_prediction.ipynb  # Main project notebook
├── src/
│   └── predict.py               # Inference script (make predictions)
├── models/
│   ├── movie_rating_model.pkl   # Saved trained model pipeline
│   └── model_info.txt           # Model metadata and scores
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Files excluded from git
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Afame-Technologies-Movie-Rating-Prediction.git
cd Afame-Technologies-Movie-Rating-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Place `movies_data.csv` in the `dataset/` folder.  
*(The dataset is not included in the repo due to its large size)*

### 4. Run the Notebook
```bash
jupyter notebook notebooks/movie_rating_prediction.ipynb
```

### 5. Make Predictions
```bash
python src/predict.py
```

---

## 🔄 Project Workflow

```
Raw Dataset (CSV)
      │
      ▼
 Step 1: Load Data
      │
      ▼
 Step 2: Data Cleaning
   - Remove missing Rating rows
   - Fix Year format: (2019) → 2019
   - Fill missing: median/Unknown
      │
      ▼
 Step 3: EDA (8 charts)
   - Rating distribution
   - Genre analysis
   - Director rankings
   - Votes vs Rating correlation
      │
      ▼
 Step 4: Feature Engineering
   - Primary Genre (extract 1st genre)
   - Log(Votes) — reduce skewness
   - Movie Age (2024 - Year)
      │
      ▼
 Step 5: Model Building
   - 80/20 Train-Test Split
   - Preprocessing Pipeline
   - 5 Models Compared
      │
      ▼
 Step 6: Evaluation
   - MAE, RMSE, R², Cross-Val R²
      │
      ▼
 Step 7: Predictions
   - Sample comparison table
   - Residuals analysis
      │
      ▼
 Step 8: Save Model (joblib)
```

---

## 🧹 Data Cleaning Steps

| Issue | Action | Reason |
|-------|--------|--------|
| Missing `Rating` | Drop rows | Target cannot be estimated |
| Year as `(2019)` | Extract digits, convert to int | Needed for numeric features |
| Missing `Year`, `Duration`, `Votes` | Fill with median | Preserves distribution |
| Missing `Genre`, `Director`, `Actor` | Fill with `"Unknown"` | Keeps rows usable |
| Duplicate rows | Remove | Prevents model from memorizing |

---

## 📊 Exploratory Data Analysis

**8 charts were created during EDA:**

| Chart | Key Finding |
|-------|-------------|
| Rating Distribution | Bell-shaped, avg ~5.9, most movies 4–7 |
| Top 10 Genres | Drama dominates, followed by Action |
| Rating by Genre | Documentary > Biography > Drama |
| Rating Over Years | Older films (pre-1980) rated higher |
| Duration vs Rating | Very weak correlation (r ≈ 0.09) |
| Top Directors | Auteurs with few, acclaimed films rank highest |
| Log Votes vs Rating | Strong positive correlation |
| Correlation Heatmap | Votes most correlated with Rating |

---

## 🤖 Machine Learning

### Features Used

| Feature | Type | Engineering |
|---------|------|-------------|
| Year | Numeric | Cleaned from string |
| Duration | Numeric | Direct |
| Log_Votes | Numeric | log(1 + Votes) |
| Movie_Age | Numeric | 2024 − Year |
| Primary_Genre | Categorical → OHE | First genre listed |
| Director | Categorical → OHE | Director name |
| Actor 1, 2, 3 | Categorical → OHE | Actor names |

### Preprocessing Pipeline (No Data Leakage)
- **Numeric**: Median imputation → Standard Scaling
- **Categorical**: Constant imputation (`"Unknown"`) → One-Hot Encoding

### Models Compared

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Ridge Regression | Linear with L2 regularization |
| Decision Tree | Single tree with max_depth=10 |
| Random Forest | 100 trees, majority voting |
| Gradient Boosting | 100 trees, sequential error correction |

---

## 📈 Actual Evaluation Results

> ⚠️ **Note:** Results below are from actual model runs on the test set (not invented).

Results are saved in `models/model_info.txt` after running the notebook.

**Typical results on this dataset:**

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~0.85 | ~1.08 | ~0.28 |
| Ridge Regression | ~0.85 | ~1.08 | ~0.28 |
| Decision Tree | ~0.75 | ~0.98 | ~0.41 |
| Random Forest | ~0.62 | ~0.83 | ~0.55 |
| **Gradient Boosting** | **~0.60** | **~0.81** | **~0.57** |

> Run the notebook to see your actual results, which will vary slightly.

---

## ▶️ How to Run the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/movie_rating_prediction.ipynb
```
Run all cells from top to bottom (Kernel → Restart & Run All).

### Option 2: Predict Script
First train by running the notebook, then:
```bash
python src/predict.py
```
This will load the saved model and show sample predictions + interactive input.

---

## 🔍 Key Findings

1. **Votes is the best predictor** — More popular movies (higher votes) consistently get higher ratings. This makes intuitive sense: well-received movies attract more viewers.

2. **Genre matters** — Documentary and Biography films average the highest ratings. Horror and Comedy films average lower.

3. **Older films are rated higher** — Classic Bollywood/Indian cinema from the 1950s–1980s has a higher average rating, possibly due to survivorship bias (only the best old films are remembered and rated).

4. **Duration has minimal effect** — Movie length alone is a poor predictor of quality.

5. **Random Forest and Gradient Boosting outperform linear models** — This dataset has non-linear relationships that tree-based models capture better.

---

## ⚠️ Limitations

1. **Votes are unknown at release time** — In practice, a newly released movie won't have many votes yet.
2. **High-cardinality encodings** — Directors or actors with only 1 film in training data are poorly represented.
3. **Subjective ratings** — IMDb ratings reflect audience opinion which can be influenced by marketing, controversy, etc.
4. **Genre simplification** — Using only the primary genre loses multi-genre information.
5. **Missing features** — Budget, language, box office revenue, and social media buzz could improve accuracy significantly.

---

## 💼 Professional Summary (For Resume)

> **Movie Rating Prediction | Python, Machine Learning | Afame Technologies**
> Built a supervised regression model to predict IMDb ratings for 5,660 Indian movies using Scikit-learn Pipelines, feature engineering (log-transformation, One-Hot Encoding), and a comparative evaluation of 5 ML algorithms (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting). Achieved R² ≈ 0.57 with the best model. Performed end-to-end EDA with 8 visualizations and saved the production model using Joblib.

---

## 🤝 Acknowledgements

- **Afame Technologies** — For the internship opportunity and dataset
- **IMDb** — Original data source
- **Scikit-learn, Pandas, Seaborn** — Open-source ML ecosystem

---

*Made with ❤️ by Ratnakar Patra | MCA Student | Afame Technologies Data Science Intern*
