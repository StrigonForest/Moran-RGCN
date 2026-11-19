import pandas as pd
import numpy as np
import torch
import os
from typing import Optional, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    average_precision_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


def run_RandomForest(
        content_filename: str,
        seed: int = 42,
        split_indices: Optional[Dict[str, torch.Tensor]] = None, ):
    data = pd.read_csv(content_filename)

    # Select Presence as the target variable y
    y = data['Presence']
    # Drop unnecessary columns
    X = data.drop(columns=['OID_', 'Shape_Length', 'Shape_Area', 'Presence'])

    X_np = np.array(X)
    y_np = y.to_numpy()

    train_indices, val_indices, test_indices = split_indices

    X_train, X_val, X_test = X_np[train_indices], X_np[val_indices], X_np[test_indices]
    y_train, y_val, y_test = y_np[train_indices], y_np[val_indices], y_np[test_indices]

    # # Set the hyperparameter search space
    # param_grid = {
    #     'n_estimators': [100, 300, 500, 1000, 1500],
    #     'max_depth': np.arange(1, 101, 10),
    # }
    #
    # # Define the random forest classifier
    # rf = RandomForestClassifier(random_state=seed)
    #
    # # Use StratifiedKFold to preserve class proportions
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    #
    # # Define GridSearchCV with built-in AUC scoring
    # grid_search = GridSearchCV(
    #     estimator=rf,
    #     param_grid=param_grid,
    #     scoring='roc_auc',  # Built-in AUC scoring
    #     cv=kf,
    #     n_jobs=5,
    #     verbose=2
    # )
    #
    # # Start hyperparameter tuning
    # grid_search.fit(X_train, y_train)
    #
    # # Get the best parameters and score
    # best_params = grid_search.best_params_
    # best_auc = grid_search.best_score_
    #
    # print(f"Best parameters: {best_params}")
    # print(f"Best AUC: {best_auc:.4f}")
    # # # Best parameters: {'max_depth': 21, 'n_estimators': 1500}
    # #
    # # print("Training set class counts:", np.bincount(y_train))
    # # print("Validation set class counts:", np.bincount(y_val))
    # # print("Test set class counts:", np.bincount(y_test))

    # Create the model
    rf_model = RandomForestClassifier(max_depth=21, n_estimators=1500, random_state=seed)

    # Train the model on the training set
    rf_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n──── The RF Final ───")
    print(f"Accuracy  : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # # Create a DataFrame
    # results_df = pd.DataFrame({
    #     'Acc': [acc * 100],
    #     'AUC': [auc * 100],
    #     'Precision': [prec * 100],
    #     'Recall': [rec * 100],
    #     'F1': [f1 * 100]
    # })
    # # Specify the CSV file path
    # file_path = './comparison/' + 'RF' + '_results.csv'
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # # Append results to the CSV file
    # results_df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

    # # Export predictions for plotting
    # all_probs = rf_model.predict_proba(X_np)[:, 1]
    #
    # probs_df = pd.DataFrame({
    #     'slope_idx': data.index + 1,  # or OID_
    #     'true_label': y_np,
    #     'rf_prob': all_probs
    # })
    #
    # all_prob_path = './comparison/RF_probs.csv'
    # probs_df.to_csv(all_prob_path, index=False)
    # print(f'\nPer-sample probabilities written to {all_prob_path}')


def run_SVM(
        content_filename: str,
        seed: int = 42,
        split_indices: Optional[Dict[str, torch.Tensor]] = None, ):
    data = pd.read_csv(content_filename)

    # Select Presence as the target variable y
    y = data['Presence']
    # Drop unnecessary columns
    X = data.drop(columns=['OID_', 'Shape_Length', 'Shape_Area', 'Presence'])

    X_np = np.array(X)
    y_np = y.to_numpy()

    train_indices, val_indices, test_indices = split_indices

    X_train, X_val, X_test = X_np[train_indices], X_np[val_indices], X_np[test_indices]
    y_train, y_val, y_test = y_np[train_indices], y_np[val_indices], y_np[test_indices]

    # 3. Standardize (fit on the training set only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # # Set the hyperparameter search space
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    #     'kernel': ['rbf', 'linear', 'poly'],
    # }
    #
    # # Define the SVM classifier
    # svm = SVC(random_state=seed)
    #
    # # Use StratifiedKFold to preserve class proportions
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    #
    # # Define GridSearchCV with built-in AUC scoring
    # grid_search = GridSearchCV(
    #     estimator=svm,
    #     param_grid=param_grid,
    #     scoring='roc_auc',  # Built-in AUC scoring
    #     cv=kf,
    #     n_jobs=5,
    #     verbose=2
    # )
    #
    # # Start hyperparameter tuning
    # grid_search.fit(X_train, y_train)
    #
    # # Get the best parameters and score
    # best_params = grid_search.best_params_
    # best_auc = grid_search.best_score_
    #
    # print(f"Best parameters: {best_params}")
    # print(f"Best AUC: {best_auc:.4f}")
    # # # Best parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
    # #
    # # print("Training set class counts:", np.bincount(y_train))
    # # print("Validation set class counts:", np.bincount(y_val))
    # # print("Test set class counts:", np.bincount(y_test))

    # Create the model
    svm_model = SVC(C=100, kernel='rbf', gamma=0.01, probability=True, random_state=seed)

    # Train the model on the training set
    svm_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n──── The SVM Final ───")
    print(f"Accuracy  : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # # Create a DataFrame
    # results_df = pd.DataFrame({
    #     'Acc': [acc * 100],
    #     'AUC': [auc * 100],
    #     'Precision': [prec * 100],
    #     'Recall': [rec * 100],
    #     'F1': [f1 * 100]
    # })
    # # Specify the CSV file path
    # file_path = './comparison/' + 'SVM' + '_results.csv'
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # # Append results to the CSV file
    # results_df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
    #
    # # Export predictions for plotting
    # X_np = scaler.fit_transform(X_np)
    # all_probs = svm_model.predict_proba(X_np)[:, 1]
    #
    # probs_df = pd.DataFrame({
    #     'slope_idx': data.index + 1,  # or OID_
    #     'true_label': y_np,
    #     'svm_prob': all_probs
    # })
    #
    # all_prob_path = './comparison/SVM_probs.csv'
    # probs_df.to_csv(all_prob_path, index=False)
    # print(f'\nPer-sample probabilities written to {all_prob_path}')


def run_LR(
        content_filename: str,
        seed: int = 42,
        split_indices: Optional[Dict[str, torch.Tensor]] = None, ):
    data = pd.read_csv(content_filename)

    # Select Presence as the target variable y
    y = data['Presence']
    # Drop unnecessary columns
    X = data.drop(columns=['OID_', 'Shape_Length', 'Shape_Area', 'Presence'])

    X_np = np.array(X)
    y_np = y.to_numpy()

    train_indices, val_indices, test_indices = split_indices

    X_train, X_val, X_test = X_np[train_indices], X_np[val_indices], X_np[test_indices]
    y_train, y_val, y_test = y_np[train_indices], y_np[val_indices], y_np[test_indices]

    # 3. Standardize (fit on the training set only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # # Set the hyperparameter search space
    # param_grid = [
    #     # Standard L2 regularization (lbfgs/newton-cg/sag)
    #     {
    #         "solver": ["lbfgs", "newton-cg", "sag"],
    #         "penalty": ["l2"],
    #         "C": np.logspace(-4, 4, 10),
    #         "multi_class": ["auto", "multinomial"],
    #     },
    #     # Sparse L1 regularization: liblinear / saga
    #     {
    #         "solver": ["liblinear"],
    #         "penalty": ["l1", "l2"],
    #         "C": np.logspace(-4, 4, 10),
    #         "multi_class": ["ovr"],
    #     },
    #     # Elastic net regularization (supported only by saga)
    #     {
    #         "solver": ["saga"],
    #         "penalty": ["elasticnet"],
    #         "l1_ratio": [0.1, 0.5, 0.9],
    #         "C": np.logspace(-4, 4, 10),
    #         "multi_class": ["auto", "multinomial"],
    #     },
    # ]
    #
    # # Define the logistic regression classifier
    # lr = LogisticRegression(max_iter=1000, random_state=seed)
    #
    # # Use StratifiedKFold to preserve class proportions
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    #
    # # Define GridSearchCV with built-in AUC scoring
    # grid_search = GridSearchCV(
    #     estimator=lr,
    #     param_grid=param_grid,
    #     scoring='roc_auc',  # Built-in AUC scoring
    #     cv=kf,
    #     n_jobs=5,
    #     verbose=2
    # )
    #
    # # Start hyperparameter tuning
    # grid_search.fit(X_train, y_train)
    #
    # # Get the best parameters and score
    # best_params = grid_search.best_params_
    # best_auc = grid_search.best_score_
    #
    # print(f"Best parameters: {best_params}")
    # print(f"Best AUC: {best_auc:.4f}")
    # # Best parameters: {'C': 0.3593813663804626, 'l1_ratio': 0.1, 'multi_class': 'multinomial', 'penalty': 'elasticnet', 'solver': 'saga'}
    #
    # print("Training set class counts:", np.bincount(y_train))
    # print("Validation set class counts:", np.bincount(y_val))
    # print("Test set class counts:", np.bincount(y_test))

    # Create the model
    lr_model = LogisticRegression(
        penalty='elasticnet',
        multi_class='multinomial',
        C=3593813663804626,
        l1_ratio=0.1,
        solver='saga',
        max_iter=1000,  # Set the iteration limit
        random_state=seed
    )

    # Train the model on the training set
    lr_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n──── The LR Final ───")
    print(f"Accuracy  : {acc:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # # Create a DataFrame
    # results_df = pd.DataFrame({
    #     'Acc': [acc * 100],
    #     'AUC': [auc * 100],
    #     'Precision': [prec * 100],
    #     'Recall': [rec * 100],
    #     'F1': [f1 * 100]
    # })
    # # Specify the CSV file path
    # file_path = './comparison/' + 'LR' + '_results.csv'
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # # Append results to the CSV file
    # results_df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
    #
    # # Export predictions for plotting
    # X_np = scaler.fit_transform(X_np)
    # all_probs = lr_model.predict_proba(X_np)[:, 1]
    #
    # probs_df = pd.DataFrame({
    #     'slope_idx': data.index + 1,  # or OID_
    #     'true_label': y_np,
    #     'lr_prob': all_probs
    # })
    #
    # all_prob_path = './comparison/LR_probs.csv'
    # probs_df.to_csv(all_prob_path, index=False)
    # print(f'\nPer-sample probabilities written to {all_prob_path}')


class GCN(nn.Module):
    def __init__(self, num_layers, in_feats, hidden_feats, out_feats, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.neigh1_layers = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.neigh1_layers.append(
            GCNConv(in_feats, hidden_feats, add_self_loops=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.neigh1_layers.append(
                GCNConv(hidden_feats, hidden_feats, add_self_loops=True))

        # Output layer
        self.neigh1_layers.append(
            GCNConv(hidden_feats, out_feats, add_self_loops=True))

        self.reset_parameters()

    def reset_parameters(self):
        for _, neigh1_layer in enumerate(self.neigh1_layers):
            neigh1_layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers - 1):
            x = self.neigh1_layers[i](x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer
        x = self.neigh1_layers[-1](x, edge_index=edge_index, edge_weight=edge_weight)
        return x
