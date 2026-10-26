"""
train_and_save.py
-----------------
Trains the Movie Rating Prediction model and saves it to models/.
Optimized version: uses Target Encoding for high-cardinality features
to avoid memory/speed issues with One-Hot Encoding on Director/Actors.

Run:  python train_and_save.py
"""

import os, sys, warnings, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection   import train_test_split
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.linear_model      import LinearRegression, Ridge
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree              import DecisionTreeRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "movies_data.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
NB_DIR       = os.path.join(BASE_DIR, "notebooks")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(NB_DIR,    exist_ok=True)

print("=" * 55)
print("  Movie Rating Prediction -- Training Script")
print("  Afame Technologies | Data Science Internship")
print("=" * 55)

# ── STEP 1: Load ───────────────────────────────────────────────
print("\n[Step 1] Loading dataset...")
df = pd.read_csv(DATASET_PATH, encoding="utf-8")
print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

# ── STEP 2: Clean ──────────────────────────────────────────────
print("\n[Step 2] Cleaning data...")

df = df.drop_duplicates()

# Fix Year: (2019) -> 2019
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Fix Duration
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# Fix Votes
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Drop rows where Rating (target) is missing
before = len(df)
df = df.dropna(subset=["Rating"])
print(f"  Dropped {before - len(df)} rows with missing Rating.")

# Fill numeric with median
df["Year"]     = df["Year"].fillna(df["Year"].median())
df["Duration"] = df["Duration"].fillna(df["Duration"].median())
df["Votes"]    = df["Votes"].fillna(df["Votes"].median())

# Fill text with Unknown
for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = df[col].fillna("Unknown")

print(f"  Final shape after cleaning: {df.shape}")
print(f"  Rating range: {df['Rating'].min():.1f} to {df['Rating'].max():.1f}")
print(f"  Year   range: {int(df['Year'].min())} to {int(df['Year'].max())}")

# ── STEP 3: EDA Charts ─────────────────────────────────────────
print("\n[Step 3] Generating EDA charts...")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# Chart 1: Rating Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["Rating"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(df["Rating"].mean(),   color="red",    linestyle="--", linewidth=2,
                label=f'Mean: {df["Rating"].mean():.2f}')
axes[0].axvline(df["Rating"].median(), color="orange", linestyle="--", linewidth=2,
                label=f'Median: {df["Rating"].median():.2f}')
axes[0].set_title("Distribution of IMDb Ratings", fontsize=13, fontweight="bold")
axes[0].set_xlabel("IMDb Rating"); axes[0].set_ylabel("Count"); axes[0].legend()
axes[1].boxplot(df["Rating"], patch_artist=True, boxprops=dict(facecolor="lightblue"))
axes[1].set_title("Box Plot of IMDb Ratings", fontsize=13, fontweight="bold")
axes[1].set_ylabel("IMDb Rating"); axes[1].set_xticklabels(["All Movies"])
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart1_rating_dist.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [1/8] Rating distribution chart saved.")

# Chart 2: Top Genres
all_genres   = df["Genre"].str.split(", ", expand=False).explode()
genre_counts = all_genres.value_counts().head(10)
plt.figure(figsize=(12, 5))
bars = plt.bar(genre_counts.index, genre_counts.values,
               color=sns.color_palette("husl", 10), edgecolor="white")
plt.title("Top 10 Most Common Genres", fontsize=13, fontweight="bold")
plt.xlabel("Genre"); plt.ylabel("Count"); plt.xticks(rotation=30, ha="right")
for bar, val in zip(bars, genre_counts.values):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
             str(val), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart2_genres.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [2/8] Genre distribution chart saved.")

# Chart 3: Rating by Genre
gr_df = df[["Genre","Rating"]].copy()
gr_df["Genre"] = gr_df["Genre"].str.split(", ")
gr_df = gr_df.explode("Genre")
avg_gr = gr_df.groupby("Genre")["Rating"].agg(["mean","count"])
avg_gr = avg_gr[avg_gr["count"] >= 20].sort_values("mean", ascending=False).head(15)
plt.figure(figsize=(12, 5))
palette = sns.color_palette("RdYlGn", len(avg_gr))
bars = plt.bar(avg_gr.index, avg_gr["mean"], color=palette, edgecolor="white")
plt.axhline(df["Rating"].mean(), color="red", linestyle="--", linewidth=1.5,
            label=f'Overall Mean: {df["Rating"].mean():.2f}')
plt.title("Average Rating by Genre (min 20 movies)", fontsize=13, fontweight="bold")
plt.xlabel("Genre"); plt.ylabel("Avg Rating"); plt.xticks(rotation=35, ha="right"); plt.legend()
for bar, val in zip(bars, avg_gr["mean"]):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{val:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart3_rating_genre.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [3/8] Rating by genre chart saved.")

# Chart 4: Rating over years
yearly = df.groupby("Year")["Rating"].agg(["mean","count"]).reset_index()
yearly = yearly[(yearly["Year"]>=1950)&(yearly["Year"]<=2021)&(yearly["count"]>=5)]
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(yearly["Year"], yearly["mean"], color="steelblue", linewidth=2)
ax1.fill_between(yearly["Year"], yearly["mean"], alpha=0.2, color="steelblue")
ax1.set_xlabel("Year"); ax1.set_ylabel("Avg Rating", color="steelblue")
ax1.set_title("Average IMDb Rating Over the Years", fontsize=13, fontweight="bold")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2 = ax1.twinx()
ax2.bar(yearly["Year"], yearly["count"], alpha=0.3, color="orange", width=0.8)
ax2.set_ylabel("Number of Movies", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart4_years.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [4/8] Rating over years chart saved.")

# Chart 5: Duration vs Rating
dur_f = df[(df["Duration"]>=60)&(df["Duration"]<=300)]
plt.figure(figsize=(10,5))
plt.scatter(dur_f["Duration"], dur_f["Rating"], alpha=0.3, color="teal", s=15)
z = np.polyfit(dur_f["Duration"], dur_f["Rating"], 1)
p = np.poly1d(z); xl = np.linspace(60,300,100)
plt.plot(xl, p(xl), "r--", linewidth=2, label=f"Trend (r={dur_f['Duration'].corr(dur_f['Rating']):.3f})")
plt.title("Movie Duration vs IMDb Rating", fontsize=13, fontweight="bold")
plt.xlabel("Duration (minutes)"); plt.ylabel("IMDb Rating"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart5_duration.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [5/8] Duration vs rating chart saved.")

# Chart 6: Top directors
dir_s   = df.groupby("Director")["Rating"].agg(["mean","count"])
top_dir = dir_s[dir_s["count"]>=5].sort_values("mean", ascending=False).head(10)
plt.figure(figsize=(12,5))
bars = plt.barh(top_dir.index, top_dir["mean"],
                color=sns.color_palette("YlOrRd_r",10), edgecolor="white")
plt.title("Top 10 Directors by Avg Rating (min 5 films)", fontsize=13, fontweight="bold")
plt.xlabel("Average IMDb Rating"); plt.gca().invert_yaxis()
for bar, (val, cnt) in zip(bars, zip(top_dir["mean"], top_dir["count"])):
    plt.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
             f"{val:.2f} ({cnt} films)", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart6_directors.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [6/8] Top directors chart saved.")

# Chart 7: Votes vs Rating
vf = df[df["Votes"]>0]
plt.figure(figsize=(10,5))
plt.scatter(np.log1p(vf["Votes"]), vf["Rating"], alpha=0.3, color="purple", s=15)
z = np.polyfit(np.log1p(vf["Votes"]), vf["Rating"], 1)
p = np.poly1d(z); xl = np.linspace(0, np.log1p(vf["Votes"].max()), 100)
plt.plot(xl, p(xl), "r--", linewidth=2,
         label=f"Trend (r={np.log1p(vf['Votes']).corr(vf['Rating']):.3f})")
plt.title("log(Votes) vs IMDb Rating", fontsize=13, fontweight="bold")
plt.xlabel("log(1 + Votes)"); plt.ylabel("IMDb Rating"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart7_votes.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [7/8] Votes vs rating chart saved.")

# Chart 8: Correlation heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df[["Year","Duration","Rating","Votes"]].corr(),
            annot=True, fmt=".3f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart8_heatmap.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  [8/8] Correlation heatmap saved.")
print("  All 8 EDA charts saved to notebooks/")

# ── STEP 4: Feature Engineering ────────────────────────────────
print("\n[Step 4] Feature engineering...")

df_model = df.copy()

# Primary genre (first listed)
df_model["Primary_Genre"] = df_model["Genre"].str.split(",").str[0].str.strip()

# Log-transform Votes (reduces skewness)
df_model["Log_Votes"] = np.log1p(df_model["Votes"])

# Movie age
df_model["Movie_Age"] = 2024 - df_model["Year"]

# ── Target Encoding for high-cardinality columns ───────────────
# Instead of One-Hot (which creates thousands of columns for Director/Actors),
# we replace each name with the MEAN RATING of all movies with that name.
# This is called Target Encoding — fast, compact, and very effective.
print("  Applying Target Encoding for Director, Actor 1, Actor 2, Actor 3...")
global_mean = df_model["Rating"].mean()

for col in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
    # Compute per-category mean rating
    mean_map = df_model.groupby(col)["Rating"].mean()
    new_col  = col.replace(" ", "_") + "_enc"
    df_model[new_col] = df_model[col].map(mean_map).fillna(global_mean)
    print(f"    {col:12s} -> {new_col}  ({df_model[col].nunique()} unique values encoded)")

# Primary_Genre: small enough for OneHotEncoding (only ~20 unique values)
# We keep it as a label-encoded numeric for simplicity
genre_mean_map = df_model.groupby("Primary_Genre")["Rating"].mean()
df_model["Genre_enc"] = df_model["Primary_Genre"].map(genre_mean_map).fillna(global_mean)

FEATURES = [
    "Year", "Duration", "Log_Votes", "Movie_Age",
    "Genre_enc",
    "Director_enc", "Actor_1_enc", "Actor_2_enc", "Actor_3_enc"
]
TARGET = "Rating"

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"  Features: {FEATURES}")
print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
print(f"  No missing values in X: {X.isnull().sum().sum() == 0}")

# ── STEP 5: Model Building ──────────────────────────────────────
print("\n[Step 5] Building and comparing models...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# Simple pipeline: imputer + scaler (no OHE needed since all features are numeric now)
simple_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

models = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0),
    "Decision Tree"     : DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest"     : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=4, random_state=42),
}

results = {}
print(f"\n  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8}   Time")
print("  " + "-" * 58)

for name, model_obj in models.items():
    t0 = time.time()
    pipeline = Pipeline(steps=[
        ("preprocessor", simple_preprocessor),
        ("model", model_obj)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 10.0)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    elapsed = time.time() - t0

    results[name] = {
        "pipeline": pipeline, "MAE": mae, "RMSE": rmse,
        "R2": r2, "y_pred": y_pred
    }
    print(f"  {name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}   {elapsed:.1f}s")

print("  " + "-" * 58)

# ── STEP 6: Evaluation Charts ──────────────────────────────────
print("\n[Step 6] Generating evaluation charts...")

metrics = ["MAE", "RMSE", "R2"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison -- Evaluation Metrics", fontsize=14, fontweight="bold")
colors = sns.color_palette("husl", len(models))
for ax, metric in zip(axes, metrics):
    vals = [results[m][metric] for m in results]
    bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart9_model_compare.png"), dpi=100, bbox_inches="tight")
plt.close()

best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]
y_pred_best = best["y_pred"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred_best, alpha=0.3, color="steelblue", s=15)
mn = min(y_test.min(), y_pred_best.min())
mx = max(y_test.max(), y_pred_best.max())
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Prediction")
axes[0].set_title(f"Actual vs Predicted -- {best_name}", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Actual Rating"); axes[0].set_ylabel("Predicted Rating"); axes[0].legend()
residuals = y_test.values - y_pred_best
axes[1].hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="black", linewidth=2, linestyle="--")
axes[1].set_title("Residuals Distribution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Error (Actual - Predicted)"); axes[1].set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(NB_DIR, "chart10_predictions.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Evaluation charts saved.")

# ── STEP 7: Sample Predictions Table ───────────────────────────
print("\n[Step 7] Sample predictions...")
np.random.seed(42)
idx      = np.random.choice(len(X_test), size=10, replace=False)
X_samp   = X_test.iloc[idx]
y_actual = y_test.iloc[idx].values
y_pred_s = np.clip(best["pipeline"].predict(X_samp), 1.0, 10.0)
errors   = np.abs(y_pred_s - y_actual)

print(f"  {'Actual':>8} {'Predicted':>10} {'Abs Error':>10}")
print("  " + "-" * 30)
for a, p, e in zip(y_actual, y_pred_s, errors):
    print(f"  {a:>8.1f} {p:>10.2f} {e:>10.3f}")
print(f"  {'':->30}")
print(f"  Mean Abs Error on sample: {errors.mean():.3f}")

# ── STEP 8: Save Model ──────────────────────────────────────────
print("\n[Step 8] Saving best model and metadata...")
MODEL_PATH = os.path.join(MODEL_DIR, "movie_rating_model.pkl")
INFO_PATH  = os.path.join(MODEL_DIR, "model_info.txt")
META_PATH  = os.path.join(MODEL_DIR, "encoding_maps.pkl")

# Save the pipeline
joblib.dump(best["pipeline"], MODEL_PATH)

# Save encoding maps so predict.py can use them for new data
encoding_maps = {
    "global_mean"   : global_mean,
    "genre_mean"    : genre_mean_map.to_dict(),
    "director_mean" : df_model.groupby("Director")["Rating"].mean().to_dict(),
    "actor1_mean"   : df_model.groupby("Actor 1")["Rating"].mean().to_dict(),
    "actor2_mean"   : df_model.groupby("Actor 2")["Rating"].mean().to_dict(),
    "actor3_mean"   : df_model.groupby("Actor 3")["Rating"].mean().to_dict(),
    "features"      : FEATURES,
}
joblib.dump(encoding_maps, META_PATH)

# Save human-readable model info
with open(INFO_PATH, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("  MOVIE RATING PREDICTION MODEL INFO\n")
    f.write("  Afame Technologies | Ratnakar Patra\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Best Model       : {best_name}\n\n")
    f.write("All Model Results:\n")
    f.write(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8}\n")
    f.write("  " + "-" * 50 + "\n")
    for name, res in results.items():
        marker = "  <-- BEST" if name == best_name else ""
        f.write(f"  {name:<22} {res['MAE']:>8.4f} {res['RMSE']:>8.4f} {res['R2']:>8.4f}{marker}\n")
    f.write("\n")
    f.write(f"Dataset Rows     : {len(df)}\n")
    f.write(f"Training Samples : {X_train.shape[0]}\n")
    f.write(f"Test Samples     : {X_test.shape[0]}\n")
    f.write(f"Features Used    : {FEATURES}\n")
    f.write(f"Encoding Method  : Target Encoding (mean rating per category)\n")

print(f"  Model saved  : {MODEL_PATH}")
print(f"  Info  saved  : {INFO_PATH}")
print(f"  Maps  saved  : {META_PATH}")

# ── Final Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  BEST MODEL : {best_name}")
print("=" * 55)
print(f"  MAE  : {best['MAE']:.4f}  (avg error in rating points)")
print(f"  RMSE : {best['RMSE']:.4f}")
print(f"  R2   : {best['R2']:.4f}  ({best['R2']*100:.1f}% variance explained)")
print("=" * 55)
print()
print("  All steps complete!")
print("  > Open the notebook   : notebooks/movie_rating_prediction.ipynb")
print("  > Make predictions    : python src/predict.py")
print("  > Model saved at      : models/movie_rating_model.pkl")
