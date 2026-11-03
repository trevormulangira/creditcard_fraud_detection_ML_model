############## EVALUATE MODEL #################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score

def evaluate_model(y_val, y_proba, threshold=0.5, model_name='Model'):#Evaluates a claassification model and generates visual reports
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_val, y_proba)
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = auc(recall, precision)

    print(f"========================================================")
    print(f"EVALUATION METRICS:{model_name} (Threshold: {threshold})")
    print(f"========================================================")
    print(f"ROC_AUC Score: {roc_auc:.4f}")
    print(f"PR_AUC Score: {pr_auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Legit(0)', 'Fraud(1)']))

    #Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Pred Legit', 'Pred Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'])
    axes[0].set_title(f'Confusion Matrix (Threshold: {threshold})')

    #PR Curve
    axes[1].plot(recall, precision, color='purple', lw=2, label=f'PR Curve(AUC = {pr_auc:.4f})')
    axes[1].set_xlabel('Recall(Catching Fraud)')
    axes[1].set_ylabel('Precision(Accuracy of Fraud Alarms)')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

####################CROSS VALIDATE MODEL#####################
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold


def cross_validate_model(pipeline, X, y, cv_folds=5):
    # Runs Stratified K Fold and returns the mean and std of key metrics
    scoring_metrics = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'recall': 'recall',
        'precision': 'precision'
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring_metrics, n_jobs=-1)

    print("========================================")
    print(f"{cv_folds}-Fold Cross-Validation Results")
    print("========================================")

    for metric in scoring_metrics.keys():
        scores = results[f'test_{metric}']
        print(f"{metric.upper():<10}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results

########### HYPERPARAMETER TUNING ############
#inputs; pipeline, param_grid, X and y, search_type, n_iter(only if search_type='random, cv_folds)#
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import time

def tune_hyperparameters(pipeline, param_grid, X, y, search_type='random', n_iter=20, cv_folds=5):
    print(f"Starting {search_type.upper()} search with {cv_folds}-Fold CV...")
    start_time = time.time()

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)#setup stratifiedkfold for imbalanced data
    if search_type.lower() == 'grid':
        search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=skf, scoring='average_precision', n_jobs=-1, verbose=1)
    else:
        search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=n_iter, cv=skf, scoring='average_precision', n_jobs=-1, random_state=42, verbose=1)

    search.fit(X, y)#Run the search

    #Calculate time taken
    end_time =time.time()
    minutes, seconds = divmod(end_time - start_time, 60)

    print("===================================================")
    print(f"Search Complete in {int(minutes)}m:{int(seconds)}s")
    print(f"Best PR_AUC Score:{search.best_score_:.4f}")
    print("Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"  -{param}: {value}")
    print("===================================================")

    return search.best_estimator_ #Return best model i.e. pipeline with the best paramters