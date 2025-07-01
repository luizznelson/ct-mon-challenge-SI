import sys
import os

# Adicionar src ao path
sys.path.append('src')

from data_loader import download_data, get_training_files, get_test_files
from preprocessor import process_training_data, process_test_data
from model import CTMonModel
from evaluator import evaluate_models
from utils import (create_submission_file, save_metrics_report, 
                  save_best_params, create_features_documentation, 
                  create_visualizations)

def main():
    """
    Função principal do pipeline
    """
    print("="*60)
    print("CT-MON CHALLENGE - PREVISÃO DE MÉTRICAS DE REDE")
    print("Autores: Arthur Sabino Santos, Luiz Nelson dos Santos Lima")
    print("="*60)
    
    # 1. Download dos dados
    print("\n[ETAPA 1] Download e extração dos dados")
    download_data()
    
    # 2. Carregamento dos arquivos
    print("\n[ETAPA 2] Carregamento dos arquivos")
    training_files = get_training_files()
    test_files = get_test_files()
    
    print(f"Arquivos de treinamento encontrados: {len(training_files)}")
    print(f"Arquivos de teste encontrados: {len(test_files)}")
    
    # 3. Pré-processamento
    print("\n[ETAPA 3] Pré-processamento dos dados")
    X_train, y_train = process_training_data(training_files)
    X_test = process_test_data(test_files)
    
    # 4. Treinamento do modelo
    print("\n[ETAPA 4] Treinamento do modelo")
    model = CTMonModel(random_state=42)
    model.train(X_train, y_train)
    
    # 5. Obter melhores parâmetros
    print("\n[ETAPA 5] Melhores hiperparâmetros")
    best_params = model.get_best_params()
    save_best_params(best_params)
    
    # 6. Predições para teste
    print("\n[ETAPA 6] Predições para o conjunto de teste")
    test_predictions = model.predict(X_test)
    
    # 7. Avaliação (usando parte dos dados de treino como validação)
    print("\n[ETAPA 7] Avaliação dos modelos")
    # Dividir treino em treino/validação (80/20)
    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    metrics_df = evaluate_models(model, X_train_split, y_train_split, X_val, y_val)
    
    # 8. Salvar resultados
    print("\n[ETAPA 8] Salvando resultados")
    create_submission_file(test_predictions, test_files)
    save_metrics_report(metrics_df)
    create_features_documentation()
    create_visualizations(metrics_df)
    
    print("\n" + "="*60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\nArquivos gerados:")
    print("- results/submission.csv")
    print("- results/metrics_report.csv") 
    print("- results/best_hyperparameters.csv")
    print("- results/features_documentation.csv")
    print("- results/figures/ (gráficos)")

if __name__ == "__main__":
    main()