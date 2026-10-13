"""
Student Performance Prediction - Model Evaluation
Author : Bill Odida
Adm No : 672542
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)

SEED = 42
res = {}

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("student_model_evaluation_dataset.csv")
res["shape"] = list(df.shape)
res["class_counts"] = df["Passed"].value_counts().sort_index().to_dict()
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------- 2. Missing values
missing = df.isna().sum()
res["missing_before"] = missing[missing > 0].to_dict()
print("\nMissing values:\n", missing)

# ---------------------------------------------------------------- 3. Separate X and y
X = df.drop(columns=["Student_ID", "Passed"])   # ID is an identifier, not a predictor
y = df["Passed"]
features = list(X.columns)
print("\nX shape:", X.shape, "| y shape:", y.shape)

# ---------------------------------------------------------------- 4. Train / validation / test (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
res["split"] = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
print("\nTrain/Val/Test:", len(X_train), len(X_val), len(X_test))

# Handle missing values: median imputation FIT ON TRAINING DATA ONLY (prevents data leakage).
# Imputer + scaler + model live in one Pipeline so the same rule is applied everywhere.
def make_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])

imp = SimpleImputer(strategy="median").fit(X_train)
res["train_medians"] = dict(zip(features, [round(float(v), 2) for v in imp.statistics_]))

# ---------------------------------------------------------------- 5. Train
model = make_model().fit(X_train, y_train)
res["coefficients"] = dict(zip(features, [round(float(c), 3) for c in model["clf"].coef_[0]]))

def evaluate(name, Xs, ys):
    pred = model.predict(Xs)
    proba = model.predict_proba(Xs)[:, 1]
    cm = confusion_matrix(ys, pred)
    out = {"confusion_matrix": cm.tolist(),
           "accuracy": accuracy_score(ys, pred),
           "precision": precision_score(ys, pred),
           "recall": recall_score(ys, pred),
           "f1": f1_score(ys, pred),
           "auc": roc_auc_score(ys, proba)}
    print(f"\n{name}: ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out, pred, proba

# ---------------------------------------------------------------- 6-8. Metrics
res["train"], _, _ = evaluate("TRAIN", X_train, y_train)
res["val"], pred_val, proba_val = evaluate("VALIDATION", X_val, y_val)
res["test"], pred_test, proba_test = evaluate("TEST", X_test, y_test)

# ---------------------------------------------------------------- 9. 5-fold CV (on train+val, pipeline refit per fold)
X_dev = pd.concat([X_train, X_val]); y_dev = pd.concat([y_train, y_val])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv = cross_validate(make_model(), X_dev, y_dev, cv=skf,
                    scoring=["accuracy", "precision", "recall", "roc_auc"])
res["cv"] = {m: [round(float(v), 4) for v in cv[f"test_{m}"]]
             for m in ["accuracy", "precision", "recall", "roc_auc"]}
res["cv_summary"] = {m: [round(float(np.mean(cv[f"test_{m}"])), 4),
                         round(float(np.std(cv[f"test_{m}"])), 4)]
                     for m in ["accuracy", "precision", "recall", "roc_auc"]}
print("\n5-fold CV:", res["cv_summary"])

# ---------------------------------------------------------------- 10. Figures
fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test),
                       display_labels=["Fail (0)", "Pass (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - Test Set")
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5, 4.2))
for nm, yy, pp in [("Validation", y_val, proba_val), ("Test", y_test, proba_test)]:
    fpr, tpr, _ = roc_curve(yy, pp)
    ax.plot(fpr, tpr, lw=2, label=f"{nm} (AUC = {roc_auc_score(yy, pp):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(loc="lower right"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_roc.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.hist(proba_test[y_test == 0], bins=10, alpha=.7, label="Actual Fail", color="#d9534f")
ax.hist(proba_test[y_test == 1], bins=10, alpha=.7, label="Actual Pass", color="#2e7d32")
ax.set_xlabel("Predicted probability of passing"); ax.set_ylabel("Students")
ax.set_title("Predicted Probabilities - Test Set"); ax.legend()
plt.tight_layout(); plt.savefig("fig_proba.png", dpi=200); plt.close()

json.dump(res, open("results.json", "w"), indent=2, default=float)
print("\nDone.")