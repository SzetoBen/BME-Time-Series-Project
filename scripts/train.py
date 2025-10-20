#!/usr/bin/env python3
"""
Parking Ticket Appeal Outcome Prediction (Simplified)
-----------------------------------------------------
Trains a classification model (Logistic Regression, Random Forest, or XGBoost)
using a pre-cleaned dataset ready for modeling.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)


def get_model(model_name):
    """Return model object based on user input."""
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        raise ValueError("Invalid model. Choose from: logistic, random_forest, xgboost.")


def main(args):
    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Load pre-cleaned dataset
    df = pd.read_csv(args.data)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # Split features and target
    X = df.drop(columns=["AppealStatus"])
    y = df["AppealStatus"]

    # Initialize model
    model = get_model(args.model)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    avg_cv_f1 = cv_scores.mean()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, digits=3)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save text outputs
    output_text_file = os.path.join("outputs", f"results_{args.model}.txt")
    with open(output_text_file, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Average 5-Fold CV F1-Score: {avg_cv_f1:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"ROC-AUC Score: {roc_auc:.3f}\n\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt="%d")

    print(f"\nResults saved to: {output_text_file}")

    # Save ROC curve plot
    RocCurveDisplay.from_predictions(y_test, y_pred_prob)
    plt.title(f"ROC Curve – {args.model.upper()}")
    roc_image_path = os.path.join("outputs", f"roc_{args.model}.png")
    plt.savefig(roc_image_path, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to: {roc_image_path}")

    print("\n Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification model to predict parking ticket appeal success."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the cleaned CSV dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logistic", "random_forest", "xgboost"],
        help="Which model to train: logistic, random_forest, or xgboost.",
    )

    args = parser.parse_args()
    main(args)
