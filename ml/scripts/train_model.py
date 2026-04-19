import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

DATA = "../data/vanet_dataset.csv"
MODEL_OUT = "../models/stability_rf.pkl"

def main():
    df = pd.read_csv(DATA)

    # Features + target
    features = ["speed", "degree", "avg_neighbor_dist", "rel_speed_avg", "keep_ratio"]
    X = df[features]
    y = df["stable_label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print("ROC-AUC:", round(auc, 4))
    except:
        pass

    # Feature importance (nice for interviews)
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("=== Feature Importance ===")
    print(imp)

    # Save
    joblib.dump(clf, MODEL_OUT)
    print("Saved model:", MODEL_OUT)

if __name__ == "__main__":
    main()
