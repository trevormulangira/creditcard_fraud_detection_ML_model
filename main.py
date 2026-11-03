import pandas as pd
import numpy as np

train_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Mulanjoe\Datasets\creditcard_fraud_detection_legit\archive (2)\train.csv")
#print(train_df.columns)
#print(train_df.head())
#print(train_df.dtypes)

                     ###STEP 3###
X = train_df.drop(columns=["id", "Class", "Time"])
y = train_df["Class"]

test_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Mulanjoe\Datasets\creditcard_fraud_detection_legit\archive (2)\test.csv")
#print(test_df.columns)
#print(test_df.head())

X_test = test_df.drop(columns=["id", "Time"])

#print("X shape:", X.shape)
#print("X_test shape:", X_test.shape)
#print("Columns match:", list(X.columns) == list(X_test.columns))

                         ###STEP 4###
#Test the model before giving it the final data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


                         ###STEP 5###
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

numeric_features = ['Amount']
passthrough_features = [col for col in X.columns if col != 'Amount']

preprocessor = ColumnTransformer(
    transformers=[
        ('scale_amount', StandardScaler(), numeric_features),
        ('keep_rest', 'passthrough', passthrough_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1))
])

fresh_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1))
])

                         ###STEP 6###
#pipeline.fit(X_train, y_train)
#y_preds = pipeline.predict(X_val)
#y_proba = pipeline.predict_proba(X_val)[:, 1]

                        ###STEP 7###
import joblib
best_pipeline = joblib.load('tuned_random_forest_v1.joblib')
print("Loaded tuned model from disk")

y_proba_tuned = best_pipeline.predict_proba(X_val)[:, 1]

from ml_utils import evaluate_model #Evaluate the model using the evaluate_model function from ml_utils.py
#evaluate_model(y_val, y_proba_tuned, threshold=0.5, model_name='0.5 Tuned Model')

from ml_utils import cross_validate_model #Evaluate the model using the cross_validate_model function from ml_utils.py
#cross_validate_model(best_pipeline, X, y, cv_folds=5)

#Testing model on test data
test_proba = best_pipeline.predict_proba(X_test)[:, 1]
FINAL_THRESHOLD = 0.7
final_predictions = (test_proba >= FINAL_THRESHOLD).astype(int)
print("PREDICTIONS GENERATED")

submission = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Mulanjoe\Datasets\creditcard_fraud_detection_legit\archive (2)\sample_submission.csv")

submission['Class'] = final_predictions

submission.to_csv('my_final_submission.csv', index=False)
print("File saved as my_final_submission.csv")
print("Total test rows:", len(submission))
print("Predicted Frauds(1s):", submission['Class'].sum())
print("Predicted Legits(0s):", (submission['Class']==0).sum())
