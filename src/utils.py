import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_submission_file(predictions, test_files, filename="results/submission.csv"):
    """
    Cria arquivo de submissão
    """
    os.makedirs("results", exist_ok=True)

    header = ["id", "mean_1", "stdev_1", "mean_2", "stdev_2"]

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for i, pred in enumerate(predictions):
            file_id = os.path.splitext(os.path.basename(test_files[i]))[0]
            writer.writerow([file_id, *pred])

    print(f"[INFO] Arquivo de submissão criado: {filename}")

def save_metrics_report(metrics_df, filename="results/metrics_report.csv"):
    """
    Salva relatório de métricas com RMSE (se aplicável)
    """
    os.makedirs("results", exist_ok=True)

    if 'RMSE' not in metrics_df.columns and 'MSE' in metrics_df.columns:
        metrics_df["RMSE"] = metrics_df["MSE"].apply(lambda x: np.sqrt(x))

    metrics_df.to_csv(filename, index=False, encoding="utf-8")
    print(f"[INFO] Relatório de métricas salvo: {filename}")

def save_best_params(best_params, filename="results/best_hyperparameters.csv"):
    """
    Salva melhores hiperparâmetros
    """
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(best_params)
    df.insert(0, "Output", ["Mean_1", "Stdev_1", "Mean_2", "Stdev_2"])
    df.to_csv(filename, index=False)
    print(f"[INFO] Melhores hiperparâmetros salvos: {filename}")

def create_features_documentation():
    """
    Cria documentação das features
    """
    os.makedirs("results", exist_ok=True)

    features_doc = pd.DataFrame({
        "Feature": [
            "Cliente ID", "Servidor ID", "Taxa Média", "Desvio Padrão", 
            "Última Taxa", "Último Desvio", "Coef. Variação", "Delta", "Tendência"
        ],
        "Description": [
            "ID numérico do cliente",
            "ID numérico do servidor", 
            "Média das taxas nos últimos 10 intervalos",
            "Desvio padrão das taxas nos últimos 10 intervalos",
            "Última taxa registrada",
            "Último desvio padrão registrado",
            "Coeficiente de variação (std/mean)",
            "Diferença entre as últimas duas médias",
            "Inclinação da regressão linear nos últimos 10 intervalos"
        ],
        "Justification": [
            "Identifica o cliente",
            "Identifica o servidor",
            "Captura a tendência geral do desempenho",
            "Mede a variação na estabilidade",
            "Valor mais recente disponível",
            "Variação mais recente",
            "Relaciona variação com magnitude",
            "Captura mudanças recentes",
            "Representa a tendência temporal"
        ]
    })

    features_doc.to_csv("results/features_documentation.csv", index=False)
    print("[INFO] Documentação de features criada: results/features_documentation.csv")

def create_visualizations(metrics_df):
    """
    Cria gráficos de comparação para MSE, MAE, MAPE, RMSE, R² (se disponíveis)
    """
    os.makedirs("results/figures", exist_ok=True)

    metrics_to_plot = ["MSE", "MAE", "MAPE", "RMSE", "R2"]
    y_labels = {
        "MSE": "Mean Squared Error",
        "MAE": "Mean Absolute Error",
        "MAPE": "Mean Absolute Percentage Error",
        "RMSE": "Root Mean Squared Error",
        "R2": "Coeficiente de Determinação (R²)"
    }

    for metric in metrics_to_plot:
        if metric in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=metrics_df, x="Dataset", y=metric, hue="Modelo")
            plt.title(f"Comparação de {metric} entre Modelos e Datasets")
            plt.ylabel(y_labels.get(metric, metric))
            plt.xlabel("Conjunto de Dados")
            plt.legend(title="Modelo")
            plt.tight_layout()
            plt.savefig(f"results/figures/{metric.lower()}_comparison.png")
            plt.close()

    print("[INFO] Gráficos de métricas salvos em results/figures/")
