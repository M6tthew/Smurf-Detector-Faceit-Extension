import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

# ==== CONFIGURATION ====
FILE_SMURFS = "smurfs.xlsx"
FILE_NORMAL = "normal_players.xlsx"
TARGET_COLUMN = "is_smurf"  # We'll create this column

# ==== 1. LOAD DATA ====
df_smurfs = pd.read_excel(r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\smurfingplayer_2_cleaned.xlsx")
df_normal = pd.read_excel(r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\hub_players_3_cleaned.xlsx")

# Label the data
df_smurfs[TARGET_COLUMN] = 1
df_normal[TARGET_COLUMN] = 0

# Combine the datasets
df = pd.concat([df_smurfs, df_normal], ignore_index=True)

# Drop player name (not useful as feature)
df = df.drop(columns=["Nickname"])

# ==== 2. FEATURES & TARGET ====
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ==== 3. SPLIT DATA ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==== 4. MODEL PIPELINE WITH WEIGHTING ====
# Function to reduce 'Matches Played' influence
# Function to reduce 'Matches Played' influence
# Apply weighting directly on the DataFrame before splitting or before pipeline
X_train_weighted = X_train.copy()
X_test_weighted = X_test.copy()

# Increase Matches Played impact
#X_train_weighted["Matches Played"] *= 2  # boost weight
#X_test_weighted["Matches Played"] *= 2

#X_train_weighted["WR %"] *= 2  # boost weight
#X_test_weighted["WR %"] *= 2

#X_train_weighted["CS2 Hours"] *= 3  # boost weight
#X_test_weighted["CS2 Hours"] *= 3


# Standardize other features except Matches Played
from sklearn.compose import ColumnTransformer

# numeric_features = X_train.columns.drop("Matches Played")
# preprocessor = ColumnTransformer([
#     ("scale", StandardScaler(), numeric_features)
# ], remainder="passthrough")  # keep Matches Played as-is

# pipeline = Pipeline([
#     ("preprocess", preprocessor),
#     ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
# ])

# pipeline.fit(X_train_weighted, y_train)

# Simple pipeline without scaling
pipeline = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])


# Fit the pipeline
pipeline.fit(X_train_weighted, y_train)









# ==== 5. PREDICTIONS & EVALUATION ====
y_pred = pipeline.predict(X_test)

# print("=== Classification Report ===")
# print(classification_report(y_test, y_pred))

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, y_pred))

# print("=== Accuracy Score ===")
# print(accuracy_score(y_test, y_pred))

# ==== 5b. EVALUATE MODEL WITH CUSTOM THRESHOLD (0.8) ====

# Get predicted probabilities for the test set
y_probs = pipeline.predict_proba(X_test)[:, 1]  # probability of being a smurf

# Apply 0.59 threshold
threshold = 0.59
y_pred_threshold = (y_probs >= threshold).astype(int)

#Print updated evaluation metrics
# print(f"=== Classification Report (Threshold = {threshold}) ===")
# print(classification_report(y_test, y_pred_threshold))

# print(f"=== Confusion Matrix (Threshold = {threshold}) ===")
# print(confusion_matrix(y_test, y_pred_threshold))

# print(f"=== Accuracy Score (Threshold = {threshold}) ===")
# print(accuracy_score(y_test, y_pred_threshold))


# ==== 6. SAVE MODEL ====
import joblib
joblib.dump(pipeline, "smurf_detector_model.pkl")
print("[+] Model saved as smurf_detector_model.pkl")



# ==== 7. PREDICT ON NEW DATA WITH CONFIDENCE ====
FILE_NEW_PLAYERS = r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\runner\match_players.xlsx"

# Load new data
df_new = pd.read_excel(FILE_NEW_PLAYERS)

# Keep player names for printing
player_names = df_new["Nickname"].tolist()

# Drop 'Nickname' column
df_new_features = df_new.drop(columns=["Nickname"])

# Predict probabilities
probs = pipeline.predict_proba(df_new_features)

# Print smurf confidence scores (0 to 100)
for name, prob in zip(player_names, probs):
    smurf_score = prob[1] * 100  # probability of class 1 (smurf)
    if smurf_score <= threshold * 100:
        print(f"{name}: Probably legit {smurf_score:.1f}%")
    else:
        print(f"{name}: Smurf {smurf_score:.1f}%")


import pandas as pd


import json

results = {}
for name, prob in zip(player_names, probs):
    smurf_score = prob[1] * 100
    results[name] = round(smurf_score, 1)

with open("predictions.json", "w") as f:
    json.dump(results, f)





# After training the pipeline
clf = pipeline.named_steps["clf"]
feature_names = X_train.columns
importances = clf.feature_importances_

# Combine names and importance
feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp = feat_imp.sort_values(by="importance", ascending=False)

print(feat_imp)

# ==== 8. PLOT THRESHOLD VS ACCURACY ====
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Get predicted probabilities for the test set
y_probs = pipeline.predict_proba(X_test)[:, 1]

# Define thresholds to test
# thresholds = np.linspace(0, 1, 500)  # 0.0 to 1.0 in 500 steps
# accuracies = []

# # Calculate accuracy for each threshold
# for t in thresholds:
#     y_pred_t = (y_probs >= t).astype(int)
#     acc = accuracy_score(y_test, y_pred_t)
#     accuracies.append(acc)

# max_acc = max(accuracies)
# best_threshold = thresholds[np.argmax(accuracies)]

# print(f"[+] Highest accuracy: {max_acc:.4f} at threshold: {best_threshold:.2f}")

# # Plot threshold vs accuracy
# plt.figure(figsize=(8, 5))
# plt.plot(thresholds, accuracies, marker='o')
# plt.title("Accuracy vs Threshold for Smurf Detection")
# plt.xlabel("Threshold")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.show()

