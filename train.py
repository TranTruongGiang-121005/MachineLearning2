import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

INPUT_FILE="../spectrogram_vectors.npz"
PCA_COMPONENTS=675 # 675 is the optimal number (90%) or use 0.95 for 95% variance
N_ITER = 20
N_JOBS = 2

def train_and_evaluate():
    #load vectors from the compressed file
    data=np.load(INPUT_FILE)
    X=data['X']
    X=X.astype(np.float32)
    y=data['y']
    groups=data['groups']

    #---preprocessing the vectors---
    #encode labels
    le=LabelEncoder()
    y_encoded=le.fit_transform(y=y)

    gss=GroupShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
    train_idx, test_idx=next(gss.split(X,y_encoded,groups=groups))   

    X_train, X_test=X[train_idx], X[test_idx]
    y_train, y_test=y_encoded[train_idx], y_encoded[test_idx] 
    groups_train=groups[train_idx]

    #---define the pipeline---
    #scaler -> PCA -> Classifier
    pipeline=Pipeline([
        ('scaler',StandardScaler()),
        ('pca',PCA(n_components=PCA_COMPONENTS)),
        ('rf',RandomForestClassifier(random_state=42))
    ])

    paramgrid= {
        'rf__n_estimators': [100,200,300,500],
        'rf__max_depth': [None,10,20,30],
        'rf__min_samples_split': [2,5,10],
        'rf__min_samples_leaf': [1,2,4],
        'rf__bootstrap': [True, False],
    }

    #split the training set into sub-train/val based on groups
    cv_splitter=GroupShuffleSplit(n_splits=3,test_size=0.2,random_state=42)

    print("Starting Randomized Search (this may take a while)...")
    search=RandomizedSearchCV(
        pipeline,
        param_distributions=paramgrid,
        n_iter=N_ITER,
        cv=cv_splitter,
        verbose=2,
        n_jobs=N_JOBS,
        scoring='accuracy',
        random_state=42
    )
    search.fit(X_train, y_train, groups=groups_train)

    #results
    print(f"\nBest Parameters: {search.best_params_}")
    print(f"\nBest CV Accuracy: {search.best_score_}")

    print("Evaluating on Hold-Out Test Set...")
    best_model=search.best_estimator_
    y_pred=best_model.predict(X_test)
    
    acc=accuracy_score(y_true=y_test,y_pred=y_pred)
    print(f"Final Test Accuracy: {acc}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__=="__main__":
    train_and_evaluate()