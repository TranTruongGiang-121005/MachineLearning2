import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Load Data ---
df = pd.read_csv('./results/features_enhanced_all.csv') #file path

X = df.drop(['filename','label'], axis=1)
y = df['label']
groups = df['filename']

# --- robust splitting (Prevent Data Leakage) ---
#split by SONG
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test=X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test= y.iloc[train_idx], y.iloc[test_idx]

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Encoding ---
le=LabelEncoder()
y_train_enc=le.fit_transform(y_train)
y_test_enc=le.transform(y_test)

print(f"Training on {len(X_train)} samples")
print(f"Testing on {len(X_test)} samples")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=1
)
#xgb: slow learning rate 0.5 to avoid overfitting
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05, 
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1,
    eval_metric='mlogloss'
)
#SVM
svm = SVC(
    C=10,
    kernel='rbf',
    probability=True, #required for soft voting
    random_state=42
)
#ensemble
ensemble = VotingClassifier(
    estimators=[('rf',rf),('xgb',xgb),('svm',svm)],
    voting='soft'
)
#tuning the weights
param_grid = {
    'weights': [
        [1, 1, 1],  # Equal democracy
        [2, 1, 1],  # Trust Random Forest more
        [1, 2, 1],  # Trust XGBoost more
        [1, 1, 2],  # Trust SVM more (often best for high-dim data)
        [2, 2, 1],  # Trust Trees more than SVM
        [1, 2, 2]   # Trust Math (XGB+SVM) more than RF
    ]
}
#use GroupShuffleSplit INSIDE the Cross-Validation too
cv_splitter = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
groups_train = groups.iloc[train_idx]

print("\nStarting Hyperparameter Tuning...")
search = GridSearchCV(
    estimator=ensemble, 
    param_grid=param_grid,
    cv=cv_splitter,
    n_jobs=2,           # Safe parallelization
    verbose=1,
    scoring='accuracy'
)
search.fit(X_train_scaled, y_train_enc, groups=groups_train)

print(f"\nBest Weights Found: {search.best_params_['weights']}")
print(f"Best Validation Accuracy: {search.best_score_:.2%}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test_enc, y_pred)

print(f"\nFINAL TEST ACCURACY: {acc:.2%}")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Save Confusion Matrix for Report
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Final Ensemble Matrix (Acc: {acc:.2%})')
plt.savefig('./results/final_ensemble_matrix.png')

joblib.dump(best_model,'./results/final_ensemble_model.pkl')
joblib.dump(scaler, './results/scaler.pkl')
joblib.dump(le,'./results/label_encoder.pkl')
print("Model, Scaler and Encoder saved!")