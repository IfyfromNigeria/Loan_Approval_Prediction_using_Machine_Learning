# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:26:09 2025

@author: ADMIN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Import dataset
data = pd.read_csv('Loan_approval_dataset.csv')
head = data.head()

def detect_outliers(column, iqr_multiplier = 1.5): # Doing this to know which fill method is suitable for handling missing numeric data
    """
    Detects if a column has outliers using the Interquartile (IQR) method.
    
    Parameters:
        column (pd.Series): The numeric column to check.
        iqr_multiplier (float): The IQR multiplier (default is set to 1.5, use 3.0 (Tukey's fences) for stricter outliers)
                                                                                    
    Returns:
        True if outliers exist, otherwise False.
    """
    if column.nunique() <= 2:  # Skips categorical-like numeric columns (binary values)
        return False    
    Q1 = column.quantile(0.25)  # 25th percentile
    Q3 = column.quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile Range
    lower_bound = Q1 - (iqr_multiplier * IQR)
    upper_bound = Q3 + (iqr_multiplier * IQR)

    # Count outliers
    outliers = column[(column < lower_bound) | (column > upper_bound)]

    return len(outliers) > 0  # Returns True if there are outliers

# DATA PREPROCESSING
def preprocess(data, drop_threshold=0.5):
    '''Cleans the dataset by handling missing values, removing duplicates, displaying info about the dataset
    and generaing summary statistics.
    Parameters:
        data(pd.DataFrame): The input dataframe.
        drop_threshold (float): If a row has more than this fraction of missing values, it will be dropped(0.5 means 50%)
        default value for drop_threshold is 0.5 but user can specify their own
    Returns:
        pd.DataFrame: The cleaned dataframe.
        '''
    print("*** Initial Dataset Info ***\n")
    print(data.info())
    
    # 1. Handling missing values
    missing_values = data.isnull().sum()
    print('\n*** Missing values before cleaning ***\n',missing_values[missing_values > 0])
    
    # Drop rows with too many missing values
    data = data.dropna(thresh = int(data.shape[1] * drop_threshold))
    for column in data.columns:
        if data[column].isnull().sum() > 0: # To only process columns with missing values
            if data[column].dtype in ['int64','float64']: # Numeric columns
                if detect_outliers(data[column]): # Check for outliers
                    print(f' Outliers found in {column}. Using median as fill method.')
                    data[column] = data[column].fillna(data[column].median())
                else: 
                    print(f' No significant outliers in {column}. Using mean as fill method.')
                    data[column] = data[column].fillna(data[column].mean())
            else:
                data[column] = data[column].fillna(data[column].mode()[0])

    print('\n*** Missing values after cleaning ***\n', data.isnull().sum())
    
    # 2. Remove Duplicate Records
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    final_rows = data.shape[0]
    print(f"\n### Removed {initial_rows - final_rows} Duplicate Rows ###")
    
    # 3. Display Cleaned Dataset Info
    print("\n*** Cleaned Dataset Info ***")
    print(data.info())
        
    # 4. Generate Summary statistics
    summary_stats = data.describe().T
    print("\n*** Summary Statistics ***")
    print(summary_stats)
    
    return data

cleaned_data = preprocess(data)

# TO MAINTAIN CONSISTENCY, ASSIGN CLEANED DATA AS 'df', NOT TO BE COFUSED WITH THE ORIGINAL UNPROCESSED DATA, 'data'
df = cleaned_data.copy()


# Distribution of Loan Status
sns.countplot(df, x=' loan_status') 
plt.title('Distribution of Loan Status')
plt.savefig('Distri Loan Status', dpi=300)
plt.show()

# Loan Amount VS Applicant Income
sns.scatterplot(df, x=' income_annum', y=' loan_amount')
plt.title('Loan Amount vs. Applicant Income')
plt.savefig('Loan amount vs App income', dpi=300)
plt.show()

# Credit history VS Loan Status
sns.scatterplot(df, x=' loan_amount', y=' cibil_score')
plt.title('Credit Score vs. Applicant Income')
plt.savefig('Credit Hist vs App income', dpi=300)
plt.show()

# LOAN AMOUNT VS EDUCATION LEVEL
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=' education', y=' loan_amount')
plt.title('Loan Amount by Education Level')
plt.xlabel('Education')
plt.ylabel('Loan Amount')
plt.savefig('Loan amount by education level', dpi=300)
plt.show()

# FEATURE ENGINEERING
# Define mappings for multiple columns
mapping_dict = {
    ' education': {' Not Graduate': 0, ' Graduate': 1},  # Fixed Graduate → 1
    ' self_employed': {' No': 0, ' Yes': 1},
}
df.replace(mapping_dict, inplace=True)

# label encoding
le = LabelEncoder()
df[' loan_status'] = le.fit_transform(df[' loan_status'])

# DROPPING REDUNDANT COLUMN, LOAN_ID
df.drop('loan_id', axis=1, inplace=True)

# Correlation between Features and Label
corr_matrix = df.corr()  # Compute correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.savefig('Corr Heatmap', dpi=300)
plt.show()

# # CLASS IMBALANCE
class_counts = data[" loan_status"].value_counts()
class_ratio = class_counts.min() / class_counts.max()

print(f"Class 0: {class_counts[0]}")
print(f"Class 1: {class_counts[1]}")
print(f"Class Ratio: {class_ratio:.2f}")

# SPLITTING THE DATA INTO TRAINING AND TESTING
X = df.drop([' loan_status'], axis=1)
y = df[' loan_status']

# # ADDRESSING CLASS IMBALANCE
# smote = SMOTE(random_state=16)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Visualize after applying Smote
# sns.countplot(x=y_resampled)
# plt.title("Balanced Class Distribution After SMOTE")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=16)


# # SCALING THE DATA
scaler = StandardScaler()
num_features = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
                ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
                ' luxury_assets_value', ' bank_asset_value']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# RENAMING THE SCALED DATA
X_train_scaled = X_train
X_test_scaled = X_test

# APPLYING PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)  # Transform test set using same PCA

# Plot cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Choosing the Optimal Number of Components')
plt.axhline(y=0.98, color='r', linestyle='--', label='98% Variance')
plt.savefig('PCA', dpi=300)
plt.legend()
plt.show()

sorted_explained_variance = np.sort(pca.explained_variance_ratio_)[::-1]
print(sorted_explained_variance)

pca_components = pca.components_  # Select only the first 8 components

# # Convert to DataFrame for readability
# feature_contributions = pd.DataFrame(pca_components.T,  # Transpose to match features
#                                       index=X.columns,    # Use original feature names
#                                       columns=[f'PC{i+1}' for i in range(8)])  # Name components

# print(feature_contributions)

# REAPPLYING PCA WITH 8 COMPONENTS
pca = PCA(n_components=9)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# MODEL TRAINING, TESTING & EVALUATION
# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(penalty='l2', C=1.0, random_state=16, solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(random_state=16),
    "SVM": SVC(kernel='linear', probability=True, random_state=16)
}

# STORE RESULTS
results = []
conf_matrices = {}

# TRAIN, PREDICT & EVALUATE MODELS
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    
    # Test Accuacy & CV Scores
    test_acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='accuracy').mean()
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1_score_ = f1_score(y_test, y_pred)
    
    # Store Results
    results.append({"Model": model_name, "Test Accuracy": test_acc, "Precision": precision,
                    "F1 Score": f1_score_, "Recall": recall,"Cross-Val Accuracy": cv_acc})
    
    # Visualizations for each model
    if model_name == 'Logistic Regression':
        coefficients = pd.Series(model.coef_[0], index=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
        coefficients.sort_values().plot(kind='barh')
        plt.title("Feature Impact in Logistic Regression")
        plt.savefig('Feature Impact LOGREG', dpi=300)
        plt.show()
    elif model_name == 'Decision Tree':
        plot_tree(model, feature_names=[f'PC{i+1}' for i in range(X_test_pca.shape[1])], 
                  class_names=['No', 'Yes'], filled=True)
        plt.title('Decision Trees')
        plt.savefig('Decision Trees', dpi=300)
        plt.show()
        
    # Classification Report
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    conf_matrices[model_name] = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrices[model_name])
    disp.plot(cmap="Blues") 
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f'Confusion Matrix {model_name}', dpi=300)
    plt.show()
    
    # ROC CURVE
    RocCurveDisplay.from_estimator(model, X_test_pca, y_test)
    plt.title(f'ROC Curve-- {model_name}')
    plt.savefig(f'ROC Curve {model_name}', dpi=300)
    plt.show()

# CONVERT RESULTS TO DATAFRAME
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:\n")
print(results_df)

# PLOTTING MODEL PERFORMANCE
results_df.set_index('Model').plot(kind='line', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Test Accuracy")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Model Perf. Comp.', dpi=300)
plt.show()

# PLOT CONFUSION MATRICES
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for ax, (model_name, cm) in zip(axes.flatten(), conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
plt.tight_layout()
plt.savefig('Comparative ConfMatrix', dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for i, (model_name, model) in enumerate(models.items()):
    ax = axes[i]
    RocCurveDisplay.from_estimator(model, X_test_pca, y_test, name=model_name, ax=ax)
fig.suptitle("Comparative ROC Curve")
for ax in axes.flatten():
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig('Comparative ROC Curve', dpi=300)
plt.show()

# HYPERPARAMETER TUNING
best_models = {}
models_CV = {'log_reg': LogisticRegression(),
          'dt_clf': DecisionTreeClassifier(),
          'svm': SVC()}
param_grid = {'log_reg': {'penalty': ['elasticnet', 'l1', 'l2'],
                        'C': [0.01, 0.1, 1.0],
                        'solver': ['liblinear', 'lbfgs', 'sag'],
                        'max_iter': [100, 200, 500, 1000],
                        'class_weight': [None, 'balanced']},
              'dt_clf': {'criterion': ['gini', 'entropy', 'log_loss'],
                        'max_depth': [50, 100, 200],
                        'min_samples_split': [5, 10, 20, 50],
                        'min_samples_leaf': [2, 3, 5, 10],
                        'class_weight': [None, 'balanced']},
              'svm': {'C': [0.1, 1, 10],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                      'class_weight': [None, 'balanced']}
                }

for model_key, model_value in models_CV.items():
    print(f"Running GridSearchCV for {model_key}...")
    
    grid = param_grid[model_key]
    grid_search = GridSearchCV(estimator=model_value, param_grid=grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_pca, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    # EXTRACTING THE BEST PARAMETERS
    best_models[model_key] = {
        'model': best_model,
        'best_params': best_params,
        'cross_val_score': best_score
    }

# Display the dictionary with the models, their best parameters, and CV scores.
print("Best Models and Hyperparameters:\n")
for model_key, results in best_models.items():
    print(f"\n{model_key}:")
    print(f"Model: {results['model']}")
    print(f"Best Hyperparameters: {results['best_params']}")
    print(f"Cross-Validation Score: {results['cross_val_score']}")
print("Hyperparameter tuning completed for all models.")

# APPLYING BEST PARAMETERS OBTAINED FROM HYPERPARAMETER TUNING FOR MODEL TRAINING
opt_results = []
opt_conf_mat = {}
for model_key, model_value in models_CV.items():
    best_params_for_model = best_models[model_key]['best_params']
    
    if model_key == 'log_reg':
        opt_model = LogisticRegression(**best_params_for_model)
    elif model_key == 'dt_clf':
        opt_model = DecisionTreeClassifier(**best_params_for_model)
    elif model_key == 'svm':
        opt_model = SVC(**best_params_for_model)
        
    opt_model.fit(X_train_pca, y_train)
    y_pred_opt = opt_model.predict(X_test_pca)    
    
    opt_test_acc = accuracy_score(y_test, y_pred_opt)
    cv_acc_opt = cross_val_score(opt_model, X_train_pca, y_train, cv=5, scoring='accuracy').mean()
    recall_opt = recall_score(y_test, y_pred_opt)
    precision_opt = precision_score(y_test, y_pred_opt)
    f1_score_opt = f1_score(y_test, y_pred_opt)
    
    # Store Results
    opt_results.append({"Model": model_key, "Test Accuracy": opt_test_acc, "Precision": precision_opt,
                        "F1 Score": f1_score_opt, "Recall": recall_opt, "Cross-Val Accuracy": cv_acc_opt,})

    # CLASSIFICATION REPORT & CONFUSION MATRIX  
    
    print(f"\nClassification Report for {model_key} after Hyperparameter Tuning:\n")
    print(classification_report(y_test, y_pred_opt))
 
    opt_conf_mat[model_key] = confusion_matrix(y_test, y_pred_opt)
    disp = ConfusionMatrixDisplay(opt_conf_mat[model_key])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_key} after Hyperparameter Tuning")
    plt.savefig(f'CM_OPT-{model_key}', dpi=300)
    plt.show()
    
# CONVERT RESULTS TO DATAFRAME
opt_results_df = pd.DataFrame(opt_results)
print("\nModel Performance Summary after Hyperparameter Tuning:\n")
print(opt_results_df)

# PLOTTING MODEL PERFORMANCE
opt_results_df.set_index('Model').plot(kind='line', figsize=(10, 6))
plt.title("Model Performance Comparison (After Hyperparameter Tuning")
plt.ylabel("Test Accuracy")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Comp Opt Model Perf.', dpi=300)
plt.show()

# PLOT CONFUSION MATRICES
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for ax, (model_key, cm) in zip(axes.flatten(), opt_conf_mat.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix - {model_key} After Hyperparameter Tuning")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
plt.tight_layout()
plt.savefig('Comparative Opt ConfMatrix', dpi=300)
plt.show()

# PLOT ROC CURVE
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for i, (model_key, opt_model) in enumerate(models.items()):
    ax = axes[i]
    RocCurveDisplay.from_estimator(opt_model, X_test_pca, y_test, name=model_key, ax=ax)
fig.suptitle("Comparative ROC Curve of Models after Hyperparameter Tuning")
for ax in axes.flatten():
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig('Comparative ROC Curve Opt', dpi=300)
plt.show()