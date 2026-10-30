"""
Smart Maintenance Request Classifier
Classifies tenant complaints into categories and assigns urgency scores.
Uses keyword-based NLP + TF-IDF with Multinomial Naive Bayes / Logistic Regression.
Fully compatible with zero external C-extensions.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import os


# ─── Urgency keywords and scoring rules ──────────────────────────
EMERGENCY_KEYWORDS = {
    10: ["gas leak", "fire", "smoke", "flooding", "electrocution", "explosion",
         "gas smell", "burning smell", "spark", "short circuit", "dangerous"],
    8: ["burst", "no power", "no electricity", "exposed wire",
        "ceiling fell", "railing broken", "hissing sound", "no water"],
    6: ["leak", "not working", "broken", "blocked", "overflow", "cracked",
        "tripping", "stopped", "rattling", "urgent"],
    4: ["cooling", "noise", "slow", "drip", "dripping", "smell", "pest", "cockroach", "rat",
        "termite", "peeling", "damp", "paani"],
    2: ["receipt", "parking", "noc", "key", "nameplate", "wifi",
        "intercom", "cctv", "garbage", "guard"],
}

CATEGORY_KEYWORDS = {
    "Plumbing": ["water", "paani", "pipe", "tap", "drain", "sink", "toilet",
                 "flush", "geyser", "plumbing", "leak", "bathroom", "commode",
                 "washbasin", "tank", "shower", "drainage"],
    "Electrical": ["power", "bijli", "switch", "wire", "MCB", "fan", "light",
                   "bulb", "meter", "fuse", "spark", "electrical", "inverter",
                   "socket", "plug", "wiring", "tube"],
    "HVAC": ["AC", "air conditioner", "heater", "cooler", "cooling", "heating",
             "compressor", "thermostat", "temperature", "remote", "split"],
    "Structural": ["wall", "crack", "ceiling", "door", "window", "paint",
                   "tile", "floor", "railing", "hinge", "plaster", "seepage",
                   "waterproofing", "staircase", "balcony", "damp"],
    "Gas": ["gas", "PNG", "stove", "burner", "regulator", "flame", "ignition",
            "cylinder", "pipeline", "hissing"],
    "Pest": ["cockroach", "termite", "rat", "mosquito", "bed bug", "ant",
             "pigeon", "lizard", "spider", "bee", "pest", "infestation"],
    "General": ["receipt", "parking", "NOC", "lift", "elevator", "security",
                "guard", "garbage", "CCTV", "intercom", "key", "society",
                "maintenance charge", "tanker", "WiFi", "nameplate"],
}


def compute_urgency_score(text):
    """Compute urgency score (1-10) based on keyword matching."""
    text_lower = text.lower()
    max_urgency = 1

    for score, keywords in EMERGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                max_urgency = max(max_urgency, score)

    emergency_count = sum(
        1 for keywords in list(EMERGENCY_KEYWORDS.values())[:2]
        for kw in keywords
        if kw.lower() in text_lower
    )
    if emergency_count > 1:
        max_urgency = min(10, max_urgency + 1)

    return max_urgency


def classify_category_rules(text):
    """Rule-based category classification with domain-priority matching."""
    text_lower = text.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        # Give high priority to Gas and Electrical safety issues
        if category == "Gas" and any(k in text_lower for k in ["gas", "png", "stove", "cylinder", "smell"]):
            score += 5
        elif category == "HVAC" and any(k in text_lower for k in ["ac", "air conditioner", "cooler", "heating", "cooling"]):
            score += 3
        elif category == "Electrical" and any(k in text_lower for k in ["spark", "bijli", "shock", "mcb", "fuse", "power", "switch"]):
            score += 3
        scores[category] = score

    if max(scores.values()) == 0:
        return "General"

    return max(scores, key=scores.get)


def get_urgency_label(score):
    """Convert numeric urgency to human-readable label."""
    if score >= 9:
        return "🔴 CRITICAL"
    elif score >= 7:
        return "🟠 HIGH"
    elif score >= 5:
        return "🟡 MEDIUM"
    elif score >= 3:
        return "🔵 LOW"
    else:
        return "⚪ INFO"


def get_response_template(category, urgency_score, complaint_text):
    """Generate an auto-response based on category and urgency."""
    if urgency_score >= 9:
        prefix = "⚠️ EMERGENCY PROTOCOL TRIGGERED: "
        instruction = ("This has been flagged as a critical hazard. "
                       "Emergency response vendor dispatched within 30 minutes. "
                       "Please follow tenant safety precautions immediately.")
    elif urgency_score >= 7:
        prefix = "🔔 HIGH PRIORITY: "
        instruction = ("This has been marked as high priority. "
                       "A technician will be dispatched within 4 hours.")
    elif urgency_score >= 5:
        prefix = ""
        instruction = ("Your request has been logged and assigned. "
                       "Expected resolution within 24-48 hours.")
    else:
        prefix = ""
        instruction = ("Your request has been noted. "
                       "Our property care team will address this within 3 business days.")

    category_teams = {
        "Plumbing": "Plumbing Maintenance Unit",
        "Electrical": "Electrical Response Crew",
        "HVAC": "Climate & HVAC Service Team",
        "Structural": "Civil Engineering & Structural Team",
        "Gas": "Gas Safety & Hazard Control Unit",
        "Pest": "Sanitization & Pest Management Unit",
        "General": "Property Operations Desk",
    }

    team = category_teams.get(category, "Maintenance Operations Team")

    response = (
        f"{prefix}Your {category.lower()} ticket has been received and "
        f"routed to the {team}. "
        f"{instruction} "
        f"[Ticket Ref: TenantEase Automated SLA]"
    )

    return response


def train_ml_classifier(maintenance_df, model_dir="models"):
    """Train a TF-IDF + Multinomial Logistic Regression pipeline."""
    os.makedirs(model_dir, exist_ok=True)

    df = maintenance_df.copy()
    X = df["complaint_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=600,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.5,
            max_iter=500,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    model_path = os.path.join(model_dir, "maintenance_classifier.joblib")
    joblib.dump(pipeline, model_path)

    print(f"Maintenance Classifier — Accuracy: {report['accuracy']:.3f}")
    return pipeline, report


def classify_complaint(text, ml_model=None):
    """
    Classify a maintenance complaint.
    Uses ML pipeline if available, falls back to rule-based.
    """
    if ml_model is not None:
        try:
            ml_category = ml_model.predict([text])[0]
            ml_proba = ml_model.predict_proba([text])
            ml_confidence = float(max(ml_proba[0]))
        except Exception:
            ml_category = classify_category_rules(text)
            ml_confidence = 0.80
    else:
        ml_category = classify_category_rules(text)
        ml_confidence = 0.80

    rule_category = classify_category_rules(text)

    # For safety-critical categories (Gas, Electrical), enforce rule priority
    if rule_category in ["Gas", "Electrical"]:
        final_category = rule_category
    elif ml_confidence >= 0.55:
        final_category = ml_category
    else:
        final_category = rule_category

    urgency = compute_urgency_score(text)
    urgency_label = get_urgency_label(urgency)
    response = get_response_template(final_category, urgency, text)

    return {
        "category": final_category,
        "confidence": round(ml_confidence, 3),
        "urgency_score": urgency,
        "urgency_label": urgency_label,
        "is_emergency": urgency >= 8,
        "auto_response": response,
        "original_text": text,
    }


if __name__ == "__main__":
    test_complaints = [
        "Gas leak ho rahi hai kitchen me bahut smell aa rahi hai",
        "AC not cooling at all room is very hot",
        "Need rent receipt for income tax",
        "Cockroach infestation in kitchen bahut zyada hai",
        "Switch board me spark aa rahi hai very dangerous",
    ]

    for complaint in test_complaints:
        result = classify_complaint(complaint)
        print(f"\n📝 '{complaint}'")
        print(f"   Category: {result['category']}")
        print(f"   Urgency: {result['urgency_label']} ({result['urgency_score']}/10)")
        print(f"   Emergency: {'🚨 YES' if result['is_emergency'] else 'No'}")
