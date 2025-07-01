import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

def calculate_metrics(y_true, y_pred, label=""):
    """
    Calcula métricas de avaliação para regressão
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{label} Metrics:")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return mse, rmse, mae, mape, r2

def create_baseline_model(y_train):
    """
    Cria modelo baseline (média simples por saída)
    """
    return np.mean(y_train, axis=0)

def evaluate_models(model, X_train, y_train, X_test, y_test):
    """
    Avalia modelo treinado e baseline, retorna DataFrame com todas as métricas
    """
    print("[INFO] Avaliando modelos...")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    baseline_mean = create_baseline_model(y_train)
    baseline_train_pred = np.tile(baseline_mean, (y_train.shape[0], 1))
    baseline_test_pred = np.tile(baseline_mean, (y_test.shape[0], 1))
    
    train_metrics = calculate_metrics(y_train, train_pred, "Treinamento")
    test_metrics = calculate_metrics(y_test, test_pred, "Teste")
    baseline_train_metrics = calculate_metrics(y_train, baseline_train_pred, "Baseline Treinamento")
    baseline_test_metrics = calculate_metrics(y_test, baseline_test_pred, "Baseline Teste")
    
    # Organizar resultados em DataFrame
    metrics_data = {
        "Dataset": ["Treinamento", "Treinamento", "Teste", "Teste"],
        "Modelo": ["Random Forest", "Baseline", "Random Forest", "Baseline"],
        "MSE": [train_metrics[0], baseline_train_metrics[0], test_metrics[0], baseline_test_metrics[0]],
        "RMSE": [train_metrics[1], baseline_train_metrics[1], test_metrics[1], baseline_test_metrics[1]],
        "MAE": [train_metrics[2], baseline_train_metrics[2], test_metrics[2], baseline_test_metrics[2]],
        "MAPE": [train_metrics[3], baseline_train_metrics[3], test_metrics[3], baseline_test_metrics[3]],
        "R2": [train_metrics[4], baseline_train_metrics[4], test_metrics[4], baseline_test_metrics[4]],
    }

    return pd.DataFrame(metrics_data)
