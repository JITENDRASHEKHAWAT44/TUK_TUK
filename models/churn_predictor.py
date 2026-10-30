"""
Tenant Churn Prediction Model
Predicts which tenants are likely to not renew their lease.
Uses regularized classification with balanced weights for high recall.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os


FEATURE_COLUMNS = [
    "avg_delay_days",
    "max_delay_days",
    "late_payment_rate",
    "payment_rate",
    "delay_trend",
    "partial_payment_rate",
    "total_complaints",
    "unresolved_complaints",
    "emergency_complaints",
    "avg_complaint_urgency",
    "days_until_lease_end",
    "lease_duration_months",
    "tenure_months",
    "monthly_rent",
    "age",
]

TARGET = "will_churn"


class ChurnPredictionModel:
    """Wrapper that bundles scaler and classifier for inference."""
    def __init__(self, scaler, classifier, feature_names):
        self.scaler = scaler
        self.classifier = classifier
        self.feature_names = feature_names
        coef = np.abs(classifier.coef_[0])
        total = np.sum(coef) if np.sum(coef) > 0 else 1.0
        self.feature_importances_ = coef / total

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)


def train_churn_model(features_df, model_dir="models"):
    """Train the tenant churn prediction model."""
    os.makedirs(model_dir, exist_ok=True)

    df = features_df.copy()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = LogisticRegression(
        C=0.8,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    base_model.fit(X_train_scaled, y_train)

    model = ChurnPredictionModel(scaler, base_model, FEATURE_COLUMNS)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    model_path = os.path.join(model_dir, "churn_predictor.joblib")
    joblib.dump(model, model_path)

    results = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "auc_roc": auc,
        "confusion_matrix": cm,
        "feature_importance": importance,
        "model_path": model_path,
    }

    print(f"Churn Predictor — Accuracy: {results['accuracy']:.3f}, AUC: {auc:.3f}")
    return model, results


def predict_churn(model, tenant_features):
    """Predict churn probability for tenant(s)."""
    if isinstance(tenant_features, dict):
        tenant_features = pd.DataFrame([tenant_features])

    X = tenant_features[FEATURE_COLUMNS]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    results = tenant_features[["tenant_id", "tenant_name"]].copy()
    results["churn_probability"] = probabilities
    results["will_churn"] = predictions
    results["risk_level"] = pd.cut(
        probabilities,
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    # Action directives
    actions = []
    for _, row in results.iterrows():
        tid = row["tenant_id"]
        t_data = tenant_features[tenant_features["tenant_id"] == tid].iloc[0]
        action_list = []

        if t_data["unresolved_complaints"] > 0:
            action_list.append(f"Resolve {int(t_data['unresolved_complaints'])} pending complaint(s) immediately")
        if t_data["delay_trend"] > 2:
            action_list.append("Payment delays increasing — initiate proactive contact")
        if t_data["days_until_lease_end"] < 60:
            action_list.append("Lease ending in <60 days — trigger early renewal offer")
        if row["churn_probability"] > 0.65:
            action_list.append("Offer 5% renewal discount or complimentary maintenance")
        if not action_list:
            action_list.append("Tenant in good standing — standard scheduled check-in")

        actions.append(" | ".join(action_list))

    results["recommended_actions"] = actions
    return results


if __name__ == "__main__":
    features = pd.read_csv("data/tenant_features.csv")
    model, results = train_churn_model(features)
    print("\nFeature Importance:")
    for feat, imp in results["feature_importance"].items():
        print(f"  {feat}: {imp:.4f}")
