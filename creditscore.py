# credit_scoring_from_scratch.py
# Objective: Predict an individual's creditworthiness using past financial data.
# Approach: Logistic Regression, Decision Tree, Random Forest.
# Key Features: Feature engineering, Precision/Recall/F1/ROC-AUC reporting.
# NOTE: This script simulates a dataset so you can run immediately. Replace the
# data-generation block with your real dataset load (pd.read_csv) when ready.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import pickle
import os

def simulate_dataset(n=5000, random_state=42):
    """Simulate a realistic-ish credit dataset."""
    np.random.seed(random_state)
    income = np.random.normal(50000, 20000, n).clip(5000, 400000)            # annual income
    debts = np.random.exponential(10000, n).clip(0, 200000)                  # total debts
    num_loans = np.random.poisson(2, n)                                      # number of open loans
    credit_util = np.random.beta(2, 5, n)                                    # 0-1 utilization
    on_time_pct = np.random.beta(5, 1.5, n)                                  # percent payments on time (0-1)
    prev_defaults = np.random.binomial(1, 0.05, n)                           # previous default flag
    age = np.random.normal(40, 12, n).clip(18, 90)                           # age
    employment_years = np.random.exponential(5, n).clip(0, 50)               # years employed
    loan_amount = np.random.normal(15000, 8000, n).clip(500, 200000)         # current loan applied
    avg_monthly_balance = np.random.normal(2000, 1500, n).clip(-5000, 200000)

    # Create a latent score and convert to probability (logistic-ish)
    score = (
        0.00003 * income -
        0.00005 * debts -
        0.8 * prev_defaults +
        2.5 * on_time_pct -
        1.8 * credit_util +
        0.02 * employment_years -
        0.00001 * loan_amount +
        0.0001 * avg_monthly_balance +
        np.random.normal(0, 0.5, n)
    )

    prob = 1 / (1 + np.exp(-score))
    # threshold at 55th percentile to simulate slight class imbalance
    good_credit = (prob > np.quantile(prob, 0.55)).astype(int)

    df = pd.DataFrame({
        "income": income,
        "debts": debts,
        "num_loans": num_loans,
        "credit_util": credit_util,
        "on_time_pct": on_time_pct,
        "prev_defaults": prev_defaults,
        "age": age,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "avg_monthly_balance": avg_monthly_balance,
        "good_credit": good_credit
    })

    return df

def feature_engineer(df):
    """Add derived features used often in credit scoring."""
    df = df.copy()
    # debt-to-income ratio
    df["debt_to_income"] = df["debts"] / (df["income"] + 1)
    # current loan relative to income
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    # high utilization flag
    df["high_util_flag"] = (df["credit_util"] > 0.8).astype(int)
    # young borrower flag
    df["young_flag"] = (df["age"] < 25).astype(int)
    # stable employment flag
    df["stable_employment"] = (df["employment_years"] >= 2).astype(int)
    return df

def train_and_evaluate(df, features, target="good_credit", random_state=42):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Scale numeric features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    dt = DecisionTreeClassifier(max_depth=8, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)

    # Fit
    lr.fit(X_train_scaled, y_train)
    dt.fit(X_train, y_train)      # tree-based models don't need scaling
    rf.fit(X_train, y_train)

    models = {
        "LogisticRegression": (lr, X_test_scaled),
        "DecisionTree": (dt, X_test),
        "RandomForest": (rf, X_test)
    }

    results = []
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, (model, Xeval) in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xeval)[:, 1]
        else:
            # decision_function fallback
            probs = model.decision_function(Xeval)
            probs = (probs - probs.min()) / (probs.max() - probs.min())
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        results.append({
            "model": name,
            "model_obj": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "confusion_matrix": cm,
            "report": report
        })

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curves - Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Summary DataFrame
    metrics_df = pd.DataFrame([{
        "model": r["model"], "accuracy": r["accuracy"], "precision": r["precision"],
        "recall": r["recall"], "f1": r["f1"], "roc_auc": r["roc_auc"]
    } for r in results]).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # Print detailed reports
    print("\n=== Model metrics summary ===")
    print(metrics_df.to_string(index=False))

    for r in results:
        print("\n" + "-"*60)
        print(f"Model: {r['model']}")
        print("Confusion matrix (rows: true 0/1, cols: pred 0/1):")
        print(r["confusion_matrix"])
        print("\nClassification report:")
        print(pd.DataFrame(r["report"]).transpose())

    # Feature importances for tree models
    for r in results:
        model_obj = r["model_obj"]
        if hasattr(model_obj, "feature_importances_"):
            imp = model_obj.feature_importances_
            imp_df = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
            print(f"\nFeature importances for {r['model']}:\n", imp_df.to_string(index=False))

    return results, scaler

def save_best_model(results, scaler, features, out_dir="./saved_models"):
    # Pick best by ROC-AUC
    best = max(results, key=lambda x: x["roc_auc"])
    best_name = best["model"]
    best_model = best["model_obj"]

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"best_credit_model_{best_name}.pkl")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    meta = {"features": features}
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "meta": meta}, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved best model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    return model_path, scaler_path

if __name__ == "__main__":
    # 1) Simulate / load data
    df = simulate_dataset(n=5000)  # change n or replace simulate_dataset with pd.read_csv(...) for real data

    # 2) Feature engineering (do not change objectives)
    df = feature_engineer(df)

    # 3) Define features list (these can be changed/extended)
    features = [
        "income", "debts", "num_loans", "credit_util", "on_time_pct", "prev_defaults",
        "age", "employment_years", "loan_amount", "avg_monthly_balance",
        "debt_to_income", "loan_to_income", "high_util_flag", "young_flag", "stable_employment"
    ]

    # 4) Train and evaluate
    results, scaler = train_and_evaluate(df, features, target="good_credit")

    # 5) Save best model and scaler
    save_best_model(results, scaler, features, out_dir="./saved_models")

    # 6) (optional) Save the simulated dataset so you can inspect
    df.to_csv("simulated_credit_dataset.csv", index=False)
    print("\nSimulated dataset saved to simulated_credit_dataset.csv")
    
    
    import joblib

# Save trained models to disk
joblib.dump(log_reg, "logistic_model.pkl")
joblib.dump(dt, "decision_tree_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")

print("✅ Models saved successfully as .pkl files!")

