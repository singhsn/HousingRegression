# Assignment_1
# reg_branch, hyper_branch and main branch


Models Used:
-Ridge Regression
-Decision Tree Regressor
-Random Forest Regressor

Branches:
-reg_branch: Implements and compares 3 regression models.
-hyper_branch: Extends reg_branch with hyperparameter tuning (GridSearchCV with ≥3 parameters per model).

Best Model: DecisionTreeRegressor with lowest MSE and highest R²

Results after hyperparameter tuning:

Ridge: MSE = 24.29, R² = 0.67
Decision Tree: MSE = 9.34, R² = 0.87
Random Forest: MSE = 10.69, R² = 0.85

CI/CD Automation:
-Implemented using GitHub Actions to automatically train and evaluate models on push