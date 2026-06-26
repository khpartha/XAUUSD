"""
Phase 4: Train the first baseline models (Stage 1 of the staged plan).

Steps (matching what we walked through conceptually):
1. Load the feature CSV
2. Split chronologically into train/val/test (70/15/15) - no shuffling
3. Normalize features using ONLY training-set statistics
4. Fit Logistic Regression and Random Forest baselines
5. Compare against naive baselines (majority class, "tomorrow repeats today")
6. Report validation metrics - test set is intentionally NOT touched here,
   that comes later once we're satisfied with a model.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

FEATURE_COLS = [
    'return_1d', 'return_5d', 'return_10d', 'ema20_dist', 'ema50_dist',
    'rsi_14', 'macd', 'macd_signal', 'atr_14', 'volatility_10d',
    'body_size', 'upper_wick_ratio', 'lower_wick_ratio', 'high_low_range',
    'dist_to_resistance', 'dist_to_support', 'dist_to_fib_0382',
    'dist_to_fib_0500', 'dist_to_fib_0618', 'broke_support_today',
    'broke_resistance_today'
]


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def naive_baselines(y_train, y_val):
    """Two dumb strategies to beat: majority class, and 'tomorrow repeats today'."""
    majority_class = y_train.mode()[0]
    majority_preds = np.full(len(y_val), majority_class)
    majority_acc = accuracy_score(y_val, majority_preds)

    # "tomorrow repeats today's direction" - shift the val target back by one
    # as a stand-in for "yesterday's actual direction" (approximation using
    # the label itself shifted, since target IS tomorrow's direction)
    repeat_preds = y_val.shift(1).fillna(majority_class)
    repeat_acc = accuracy_score(y_val, repeat_preds)

    return majority_acc, repeat_acc


def main():
    df = pd.read_csv("training_data/gold_features.csv")
    train_df, val_df, test_df = chronological_split(df)

    print(f"Split sizes -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} (held out, untouched)")

    X_train, y_train = train_df[FEATURE_COLS], train_df['target']
    X_val, y_val = val_df[FEATURE_COLS], val_df['target']

    # Normalize using TRAIN stats only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- Naive baselines ---
    majority_acc, repeat_acc = naive_baselines(y_train, y_val)
    print(f"\n--- Naive baselines (validation set) ---")
    print(f"Always predict majority class: {majority_acc:.1%}")
    print(f"'Tomorrow repeats today':      {repeat_acc:.1%}")

    # --- Logistic Regression ---
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_preds = logreg.predict(X_val_scaled)
    logreg_acc = accuracy_score(y_val, logreg_preds)
    logreg_prec = precision_score(y_val, logreg_preds, zero_division=0)

    print(f"\n--- Logistic Regression (validation set) ---")
    print(f"Accuracy:  {logreg_acc:.1%}")
    print(f"Precision (when it predicts UP): {logreg_prec:.1%}")
    print(f"Confusion matrix:\n{confusion_matrix(y_val, logreg_preds)}")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_preds = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)
    rf_prec = precision_score(y_val, rf_preds, zero_division=0)

    print(f"\n--- Random Forest (validation set) ---")
    print(f"Accuracy:  {rf_acc:.1%}")
    print(f"Precision (when it predicts UP): {rf_prec:.1%}")
    print(f"Confusion matrix:\n{confusion_matrix(y_val, rf_preds)}")

    # --- Feature importance from Random Forest (informative, not used for decisions yet) ---
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n--- Top 5 most useful features (Random Forest) ---")
    print(importances.head(5).to_string())

    print(f"\nNOTE: Test set ({len(test_df)} rows) was NOT touched. We only evaluate")
    print("on test once we're satisfied with model choice - see Step 8 of the training plan.")


if __name__ == "__main__":
    main()