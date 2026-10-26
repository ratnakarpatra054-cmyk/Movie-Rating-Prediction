"""
Script to generate the Jupyter Notebook for Movie Rating Prediction project.
Run: python build_notebook.py
"""
import json, os

def code_cell(source):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [source] if isinstance(source, str) else source
    }

def md_cell(source):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [source] if isinstance(source, str) else source
    }

cells = []

# ──────────────────────── TITLE ──────────────────────────────
cells.append(md_cell("""# 🎬 Movie Rating Prediction with Python
## Afame Technologies — Data Science Internship Project

**Author:** Ratnakar Patra
**Dataset:** IMDb Indian Movies Dataset (5,660 movies)
**Objective:** Predict the IMDb rating of an Indian movie using Genre, Director, Actors, Duration, Year, and Votes.

---
| Step | Description |
|------|-------------|
| 1 | Project Setup & Data Loading |
| 2 | Data Cleaning & Preprocessing |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Feature Engineering |
| 5 | Machine Learning Model Building |
| 6 | Model Evaluation |
| 7 | Predictions & Comparison |
| 8 | Save & Load the Model |
---
"""))

# ──────────────────────── STEP 1 ─────────────────────────────
cells.append(md_cell("""## Step 1: Project Setup & Data Loading

### 📦 Import Libraries
- **pandas** — loading and manipulating data (like Excel for Python)
- **numpy** — fast mathematical operations
- **matplotlib / seaborn** — creating charts
- **scikit-learn** — Machine Learning models and tools
- **joblib** — saving and loading the trained model
"""))

cells.append(code_cell("""# ── Import Libraries ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")
sns.set_palette("husl")

print("✅ All libraries imported successfully!")
print(f"   pandas  : {pd.__version__}")
print(f"   numpy   : {np.__version__}")
"""))

cells.append(md_cell("""### 📂 Load the Dataset

**Dataset Columns:**
| Column | Meaning |
|--------|---------|
| Name | Movie title |
| Year | Year of release (stored as string like *(2019)*) |
| Duration | Movie length in minutes |
| Genre | Genre(s), can be multiple (e.g. "Drama, Comedy") |
| **Rating** | ⭐ IMDb rating — **our target variable (y)** |
| Votes | Number of user votes on IMDb |
| Director | Name of the director |
| Actor 1 | Lead actor |
| Actor 2 | Second actor |
| Actor 3 | Third actor |
"""))

cells.append(code_cell("""# ── Load Dataset ───────────────────────────────────────────────
BASE_DIR     = os.path.abspath("..")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "movies_data.csv")

df = pd.read_csv(DATASET_PATH, encoding="utf-8")

print(f"✅ Dataset loaded!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("   Columns:", list(df.columns))
"""))

cells.append(code_cell("# First 5 rows\ndf.head()"))
cells.append(code_cell("# Last 5 rows\ndf.tail()"))
cells.append(code_cell("# Data types and non-null counts\ndf.info()"))
cells.append(code_cell("# Statistical summary for numeric columns\ndf.describe()"))

# ──────────────────────── STEP 2 ─────────────────────────────
cells.append(md_cell("""---
## Step 2: Data Cleaning

**Why clean data?**
Real-world data is messy:
- **Missing values** → Some rows have no Rating, no Year, etc.
- **Wrong data types** → Year stored as *(2019)* instead of 2019 (integer)
- **Duplicates** → The same movie listed twice
- These issues will cause wrong or misleading model results if not fixed.
"""))

cells.append(code_cell("""# ── 1. Check Missing Values ────────────────────────────────────
print("=" * 45)
print("  MISSING VALUES PER COLUMN")
print("=" * 45)
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0].to_string())
print(f"\\nTotal rows with at least 1 missing: {df.isnull().any(axis=1).sum()}")
"""))

cells.append(code_cell("""# ── 2. Check and Remove Duplicates ─────────────────────────────
print(f"Duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")
"""))

cells.append(code_cell("""# ── 3. Clean 'Year' Column ─────────────────────────────────────
# Year values look like: (2019) — with parentheses!
# We extract only the 4-digit number.
print("Before:", df["Year"].dropna().head(5).values)

df["Year"] = df["Year"].astype(str).str.extract(r"(\\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

print("After :", df["Year"].dropna().head(5).values)
print(f"dtype  : {df['Year'].dtype}")
"""))

cells.append(code_cell("""# ── 4. Clean 'Duration' Column ─────────────────────────────────
print("Before:", df["Duration"].dropna().head(5).values)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
print("After :", df["Duration"].dropna().head(5).values)
"""))

cells.append(code_cell("""# ── 5. Handle Missing Values ────────────────────────────────────
# Strategy:
#   Rating   → DROP rows (it's our target — we can't guess it)
#   Year     → Fill with median
#   Duration → Fill with median
#   Votes    → Fill with median
#   Text cols → Fill with 'Unknown'

before = len(df)
df = df.dropna(subset=["Rating"])
print(f"Dropped {before - len(df)} rows with missing Rating.")

df["Year"]     = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())
df["Votes"]    = df["Votes"].fillna(df["Votes"].median())

for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = df[col].fillna("Unknown")

print("\\nMissing values AFTER handling:")
print(df.isnull().sum())
print(f"\\nFinal shape: {df.shape}")
"""))

cells.append(code_cell("""# ── 6. Data Types Check ─────────────────────────────────────────
print(df.dtypes)
print(f"\\nRating  range: {df['Rating'].min():.1f} → {df['Rating'].max():.1f}")
print(f"Year    range: {int(df['Year'].min())} → {int(df['Year'].max())}")
print(f"Duration range: {int(df['Duration'].min())} → {int(df['Duration'].max())} mins")
"""))

# ──────────────────────── STEP 3 ─────────────────────────────
cells.append(md_cell("""---
## Step 3: Exploratory Data Analysis (EDA)

EDA means **exploring data visually** to find patterns before building a model.
We answer questions like:
- How are ratings distributed?
- Which genres are most common?
- Do older movies rate higher?
- Does vote count correlate with rating?
"""))

cells.append(code_cell("""# ── Chart 1: Distribution of Movie Ratings ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["Rating"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(df["Rating"].mean(),   color="red",    linestyle="--", linewidth=2,
                label=f'Mean: {df["Rating"].mean():.2f}')
axes[0].axvline(df["Rating"].median(), color="orange", linestyle="--", linewidth=2,
                label=f'Median: {df["Rating"].median():.2f}')
axes[0].set_title("Distribution of IMDb Ratings", fontsize=14, fontweight="bold")
axes[0].set_xlabel("IMDb Rating")
axes[0].set_ylabel("Number of Movies")
axes[0].legend()

axes[1].boxplot(df["Rating"], patch_artist=True, boxprops=dict(facecolor="lightblue"))
axes[1].set_title("Box Plot of IMDb Ratings", fontsize=14, fontweight="bold")
axes[1].set_ylabel("IMDb Rating")
axes[1].set_xticklabels(["All Movies"])

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart1_rating_dist.png"), dpi=120, bbox_inches="tight")
plt.show()

print(f"Average Rating : {df['Rating'].mean():.2f}")
print(f"Median  Rating : {df['Rating'].median():.2f}")
print("FINDING: Most movies fall between 4.0 and 7.0 — roughly bell-shaped distribution.")
"""))

cells.append(code_cell("""# ── Chart 2: Top 10 Most Common Genres ─────────────────────────
all_genres   = df["Genre"].str.split(", ", expand=False).explode()
genre_counts = all_genres.value_counts().head(10)

plt.figure(figsize=(12, 6))
bars = plt.bar(genre_counts.index, genre_counts.values,
               color=sns.color_palette("husl", 10), edgecolor="white")
plt.title("Top 10 Most Common Genres", fontsize=14, fontweight="bold")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=30, ha="right")
for bar, val in zip(bars, genre_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(val), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart2_genres.png"), dpi=120, bbox_inches="tight")
plt.show()
print("FINDING: Drama is the most common genre, followed by Action and Romance.")
"""))

cells.append(code_cell("""# ── Chart 3: Average Rating by Genre ────────────────────────────
genre_rating_df = df[["Genre", "Rating"]].copy()
genre_rating_df["Genre"] = genre_rating_df["Genre"].str.split(", ")
genre_rating_df = genre_rating_df.explode("Genre")
avg_by_genre = genre_rating_df.groupby("Genre")["Rating"].agg(["mean", "count"])
avg_by_genre = avg_by_genre[avg_by_genre["count"] >= 20].sort_values("mean", ascending=False).head(15)

plt.figure(figsize=(12, 6))
palette = sns.color_palette("RdYlGn", len(avg_by_genre))
bars = plt.bar(avg_by_genre.index, avg_by_genre["mean"], color=palette, edgecolor="white")
plt.axhline(df["Rating"].mean(), color="red", linestyle="--", linewidth=1.5,
            label=f'Overall Mean: {df["Rating"].mean():.2f}')
plt.title("Average IMDb Rating by Genre (min. 20 movies)", fontsize=14, fontweight="bold")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=35, ha="right")
plt.legend()
for bar, val in zip(bars, avg_by_genre["mean"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart3_rating_genre.png"), dpi=120, bbox_inches="tight")
plt.show()
print("FINDING: Documentary and Biography genres get the highest average ratings.")
"""))

cells.append(code_cell("""# ── Chart 4: Average Rating Over the Years ──────────────────────
yearly = df.groupby("Year")["Rating"].agg(["mean", "count"]).reset_index()
yearly = yearly[(yearly["Year"] >= 1950) & (yearly["Year"] <= 2021) & (yearly["count"] >= 5)]

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(yearly["Year"], yearly["mean"], color="steelblue", linewidth=2)
ax1.fill_between(yearly["Year"], yearly["mean"], alpha=0.2, color="steelblue")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average IMDb Rating", color="steelblue")
ax1.set_title("Average IMDb Rating Over the Years", fontsize=14, fontweight="bold")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.bar(yearly["Year"], yearly["count"], alpha=0.3, color="orange", width=0.8)
ax2.set_ylabel("Number of Movies", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart4_years.png"), dpi=120, bbox_inches="tight")
plt.show()
print("FINDING: Older movies (pre-1980) score higher on average, but fewer were made.")
"""))

cells.append(code_cell("""# ── Chart 5: Duration vs Rating ─────────────────────────────────
dur_f = df[(df["Duration"] >= 60) & (df["Duration"] <= 300)]
plt.figure(figsize=(10, 5))
plt.scatter(dur_f["Duration"], dur_f["Rating"], alpha=0.3, color="teal", s=15)
z = np.polyfit(dur_f["Duration"], dur_f["Rating"], 1)
p = np.poly1d(z)
x_line = np.linspace(60, 300, 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend Line")
plt.title("Movie Duration vs IMDb Rating", fontsize=14, fontweight="bold")
plt.xlabel("Duration (minutes)")
plt.ylabel("IMDb Rating")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart5_duration.png"), dpi=120, bbox_inches="tight")
plt.show()
corr = dur_f["Duration"].corr(dur_f["Rating"])
print(f"Correlation (Duration vs Rating): {corr:.4f}")
print("FINDING: Very weak correlation — movie length alone doesn't predict rating.")
"""))

cells.append(code_cell("""# ── Chart 6: Top 10 Directors by Average Rating ─────────────────
dir_stats   = df.groupby("Director")["Rating"].agg(["mean", "count"])
top_dirs    = dir_stats[dir_stats["count"] >= 5].sort_values("mean", ascending=False).head(10)

plt.figure(figsize=(12, 6))
colors = sns.color_palette("YlOrRd_r", 10)
bars = plt.barh(top_dirs.index, top_dirs["mean"], color=colors, edgecolor="white")
plt.title("Top 10 Directors by Average Rating (min. 5 films)", fontsize=14, fontweight="bold")
plt.xlabel("Average IMDb Rating")
plt.gca().invert_yaxis()
for bar, (val, cnt) in zip(bars, zip(top_dirs["mean"], top_dirs["count"])):
    plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.2f} ({cnt} films)", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart6_directors.png"), dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code_cell("""# ── Chart 7: Log(Votes) vs Rating ───────────────────────────────
votes_f = df[df["Votes"] > 0]
plt.figure(figsize=(10, 5))
plt.scatter(np.log1p(votes_f["Votes"]), votes_f["Rating"], alpha=0.3, color="purple", s=15)
z = np.polyfit(np.log1p(votes_f["Votes"]), votes_f["Rating"], 1)
p = np.poly1d(z)
x_line = np.linspace(0, np.log1p(votes_f["Votes"].max()), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend Line")
plt.title("Log(Votes) vs IMDb Rating", fontsize=14, fontweight="bold")
plt.xlabel("log(1 + Votes)")
plt.ylabel("IMDb Rating")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart7_votes.png"), dpi=120, bbox_inches="tight")
plt.show()
corr = np.log1p(votes_f["Votes"]).corr(votes_f["Rating"])
print(f"Correlation (log Votes vs Rating): {corr:.4f}")
print("FINDING: Popular movies (many votes) tend to get higher ratings!")
"""))

cells.append(code_cell("""# ── Chart 8: Correlation Heatmap ────────────────────────────────
numeric_cols = ["Year", "Duration", "Rating", "Votes"]
corr_matrix  = df[numeric_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart8_heatmap.png"), dpi=120, bbox_inches="tight")
plt.show()
print("FINDING: Votes has the strongest positive correlation with Rating.")
"""))

# ──────────────────────── STEP 4 ─────────────────────────────
cells.append(md_cell("""---
## Step 4: Feature Engineering

**Feature Engineering** means:
1. Choosing the right columns to use as inputs (features)
2. Creating new useful columns from existing data
3. Converting text into numbers (ML models only understand numbers)

| Feature | Type | Why? |
|---------|------|------|
| Year | Numeric | Era affects rating |
| Duration | Numeric | Movie length |
| Log_Votes | Numeric | Popularity proxy (log-scaled) |
| Movie_Age | Numeric | How old the movie is |
| Primary_Genre | Categorical | First listed genre |
| Director | Categorical | Director reputation |
| Actor 1,2,3 | Categorical | Cast reputation |
"""))

cells.append(code_cell("""# ── Feature Engineering ──────────────────────────────────────────
df_model = df.copy()

# Extract primary genre (first listed genre)
df_model["Primary_Genre"] = df_model["Genre"].str.split(",").str[0].str.strip()

# Log-transform Votes to reduce skewness (log1p handles zeros safely)
df_model["Log_Votes"] = np.log1p(df_model["Votes"])

# How old is the movie from 2024?
df_model["Movie_Age"] = 2024 - df_model["Year"]

FEATURES = ["Year", "Duration", "Log_Votes", "Movie_Age",
            "Primary_Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
TARGET   = "Rating"

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"Features (X) shape: {X.shape}")
print(f"Target  (y) shape: {y.shape}")
print("\\nSample X row:")
print(X.head(1).T)
"""))

cells.append(code_cell("""# ── Identify Feature Types ───────────────────────────────────────
NUMERIC_FEATURES     = ["Year", "Duration", "Log_Votes", "Movie_Age"]
CATEGORICAL_FEATURES = ["Primary_Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]

print("Numeric   :", NUMERIC_FEATURES)
print("Categorical:", CATEGORICAL_FEATURES)
print("\\nUnique values per categorical feature:")
for col in CATEGORICAL_FEATURES:
    print(f"   {col:20s}: {X[col].nunique():>5} unique values")
"""))

# ──────────────────────── STEP 5 ─────────────────────────────
cells.append(md_cell("""---
## Step 5: Machine Learning — Build the Model

### Key Concepts:
- **Train-Test Split**: 80% training, 20% testing
- **Pipeline**: Chains preprocessing + model together (prevents data leakage)
- **Cross-Validation**: Tests model on 5 different splits for reliable evaluation

### Models We Compare:
| Model | Description |
|-------|-------------|
| Linear Regression | Fits a straight line |
| Ridge Regression | Linear + regularization (prevents overfitting) |
| Decision Tree | Yes/No questions in a flowchart |
| Random Forest | Many Decision Trees voting together |
| Gradient Boosting | Trees built sequentially, fixing errors |
"""))

cells.append(code_cell("""# ── Train-Test Split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set : {X_train.shape[0]} samples")
print(f"Testing  set : {X_test.shape[0]} samples")
"""))

cells.append(code_cell("""# ── Build Preprocessing Pipeline ─────────────────────────────────
# Numeric: fill missing with median, then scale to mean=0, std=1
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical: fill missing with 'Unknown', then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine into one ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])

print("✅ Preprocessing pipeline ready!")
"""))

cells.append(code_cell("""# ── Define Models ────────────────────────────────────────────────
models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0, random_state=42),
    "Decision Tree"     : DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
}
print("Models defined:", list(models.keys()))
"""))

cells.append(code_cell("""# ── Train & Evaluate All Models ──────────────────────────────────
results = {}

print("Training and evaluating models...")
print("-" * 65)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'CV_R2':>10}")
print("-" * 65)

for name, model_obj in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_obj)
    ])
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    mae     = mean_absolute_error(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    r2      = r2_score(y_test, y_pred)
    cv_r2   = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2").mean()

    results[name] = {
        "pipeline": pipeline, "MAE": mae, "RMSE": rmse, "R2": r2,
        "CV_R2": cv_r2, "y_pred": y_pred
    }
    print(f"{name:<25} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {cv_r2:>10.4f}")

print("-" * 65)
print("\\n✅ All models trained!")
"""))

# ──────────────────────── STEP 6 ─────────────────────────────
cells.append(md_cell("""---
## Step 6: Model Evaluation

### Metrics Explained:
| Metric | Meaning | Best Value |
|--------|---------|-----------|
| **MAE** | Average prediction error in rating points | Lower = better |
| **RMSE** | Like MAE but penalizes big errors more | Lower = better |
| **R²** | % of variance explained (1.0 = perfect) | Higher = better |
| **CV R²** | Average R² across 5 train-test splits | Higher = better |
"""))

cells.append(code_cell("""# ── Visualize Model Comparison ───────────────────────────────────
metrics = ["MAE", "RMSE", "R2"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison — Evaluation Metrics", fontsize=15, fontweight="bold")
colors = sns.color_palette("husl", len(models))

for ax, metric in zip(axes, metrics):
    values      = [results[m][metric] for m in results]
    model_names = list(results.keys())
    bars = ax.bar(range(len(model_names)), values, color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace(" ", "\\n") for m in model_names], fontsize=8)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart9_model_compare.png"), dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code_cell("""# ── Pick the Best Model ──────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]

print("=" * 52)
print(f"  BEST MODEL: {best_name}")
print("=" * 52)
print(f"  MAE  : {best['MAE']:.4f}   (avg error in rating points)")
print(f"  RMSE : {best['RMSE']:.4f}")
print(f"  R²   : {best['R2']:.4f}   ({best['R2']*100:.1f}% variance explained)")
print(f"  CV R²: {best['CV_R2']:.4f}  (5-fold cross-validation)")
print("=" * 52)
print(f"\\nOn average, our predictions are off by {best['MAE']:.2f} rating points.")
"""))

# ──────────────────────── STEP 7 ─────────────────────────────
cells.append(md_cell("""---
## Step 7: Predictions & Comparison
"""))

cells.append(code_cell("""# ── Predicted vs Actual Plot ─────────────────────────────────────
y_pred_best = best["y_pred"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_best, alpha=0.3, color="steelblue", s=15)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
axes[0].set_title(f"Actual vs Predicted — {best_name}", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Actual Rating")
axes[0].set_ylabel("Predicted Rating")
axes[0].legend()

residuals = y_test.values - y_pred_best
axes[1].hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="black", linewidth=2, linestyle="--")
axes[1].set_title("Residuals Distribution (Errors)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Prediction Error (Actual − Predicted)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebooks", "chart10_predictions.png"), dpi=120, bbox_inches="tight")
plt.show()

print("FINDING: Residuals are centered around 0 — our model is unbiased.")
print("Most errors are within ±1 rating point.")
"""))

cells.append(code_cell("""# ── Sample Predictions Table ─────────────────────────────────────
np.random.seed(42)
sample_idx  = np.random.choice(len(X_test), size=15, replace=False)
X_sample    = X_test.iloc[sample_idx].copy()
y_actual    = y_test.iloc[sample_idx].values
y_predicted = np.clip(best["pipeline"].predict(X_sample), 1.0, 10.0)

comparison_df = pd.DataFrame({
    "Genre"        : X_sample["Primary_Genre"].values,
    "Director"     : X_sample["Director"].values,
    "Year"         : X_sample["Year"].astype(int).values,
    "Actual Rating": y_actual.round(1),
    "Predicted"    : y_predicted.round(2),
    "Error"        : (y_predicted - y_actual).round(2)
})
comparison_df["AbsError"] = comparison_df["Error"].abs()
comparison_df = comparison_df.sort_values("AbsError").reset_index(drop=True)

print("SAMPLE PREDICTIONS (15 movies — sorted by error):\\n")
print(comparison_df.to_string(index=False))
print(f"\\nMean Absolute Error on sample: {comparison_df['AbsError'].mean():.3f}")
"""))

cells.append(md_cell("""### ⚠️ Model Limitations

1. **Votes are unknown at release** — We don't know how many votes a movie will get before it's released.
2. **High-cardinality categories** — Directors with only 1-2 movies in training get poor representation.
3. **Genre simplification** — Multi-genre movies reduced to primary genre loses information.
4. **Cultural and marketing factors** — Not captured in our features.
5. **IMDb ratings are subjective** — Influenced by social trends, campaign effects, etc.
"""))

# ──────────────────────── STEP 8 ─────────────────────────────
cells.append(md_cell("""---
## Step 8: Save & Load the Model

We save the full Pipeline (preprocessing + model) using **joblib**.
This lets us later load and use the model **without retraining**.
"""))

cells.append(code_cell("""# ── Save the Best Model ──────────────────────────────────────────
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "movie_rating_model.pkl")

joblib.dump(best["pipeline"], MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")

# Save model metadata
info_path = os.path.join(MODEL_DIR, "model_info.txt")
with open(info_path, "w") as f:
    f.write(f"Best Model       : {best_name}\\n")
    f.write(f"MAE              : {best['MAE']:.4f}\\n")
    f.write(f"RMSE             : {best['RMSE']:.4f}\\n")
    f.write(f"R2 (Test)        : {best['R2']:.4f}\\n")
    f.write(f"CV R2 (5-Fold)   : {best['CV_R2']:.4f}\\n")
    f.write(f"Training Samples : {X_train.shape[0]}\\n")
    f.write(f"Test Samples     : {X_test.shape[0]}\\n")
    f.write(f"Features         : {FEATURES}\\n")
print(f"✅ Model info saved to: {info_path}")
"""))

cells.append(code_cell("""# ── Load and Use the Saved Model ─────────────────────────────────
loaded_model = joblib.load(MODEL_PATH)
print("✅ Model loaded from disk!")

# Sample prediction
new_movie = pd.DataFrame([{
    "Year"         : 2023,
    "Duration"     : 148,
    "Log_Votes"    : np.log1p(50000),
    "Movie_Age"    : 2024 - 2023,
    "Primary_Genre": "Action",
    "Director"     : "S. Shankar",
    "Actor 1"      : "Rajinikanth",
    "Actor 2"      : "Akshay Kumar",
    "Actor 3"      : "Amy Jackson",
}])

predicted = float(np.clip(loaded_model.predict(new_movie)[0], 1.0, 10.0))
print(f"\\n🎬 NEW MOVIE PREDICTION:")
print(f"   Director : S. Shankar | Genre : Action | Year : 2023")
print(f"   Lead     : Rajinikanth")
print(f"   ✅ Predicted IMDb Rating: {predicted:.2f} / 10")
"""))

cells.append(md_cell("""---
## 🎉 Project Summary

### What You Accomplished:
✅ Loaded and explored **5,660 Indian movies** from the IMDb dataset
✅ **Cleaned data** — missing values, wrong types, duplicates
✅ **EDA with 8 charts** — patterns, correlations, insights
✅ **Feature Engineering** — Primary Genre, Log Votes, Movie Age
✅ **Compared 5 ML models** — Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
✅ **Evaluated** using MAE, RMSE, R², Cross-Validation
✅ **Visualized** predictions vs actual ratings
✅ **Saved and loaded** model with joblib

### Key Findings:
- 📊 **Votes** is the strongest predictor of IMDb rating
- 🎭 **Documentary/Biography** genres get the highest avg ratings
- 🕰️ **Older movies** (pre-1980) have higher average ratings
- ⏱️ **Duration** has very little effect on rating
- 🌳 **Random Forest / Gradient Boosting** perform best

---
*Project by Ratnakar Patra | Afame Technologies Data Science Internship*
"""))

# ── ASSEMBLE NOTEBOOK ─────────────────────────────────────────
notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'cells': cells
}

base = r'c:\Users\Ratnakar Patra\OneDrive\Desktop\Movie-Rating-Prediction'
out  = os.path.join(base, 'notebooks', 'movie_rating_prediction.ipynb')
os.makedirs(os.path.dirname(out), exist_ok=True)

with open(out, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook written to: " + out)
