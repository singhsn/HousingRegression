from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE={mse:.2f}, R²={r2:.2f}")
    return mse, r2
