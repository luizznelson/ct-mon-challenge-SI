# 🌐 CT-MON Challenge SI

## 📋 Descrição do Projeto

Este projeto implementa um pipeline completo de predição multivariada para métricas de qualidade de transmissão em redes acadêmicas. Desenvolvido com base nos dados do Desafio de Dados CT-MON da RNP (Rede Nacional de Ensino e Pesquisa), o sistema é capaz de prever com 10 minutos de antecedência a qualidade do tráfego DASH entre clientes e servidores distribuídos regionalmente.

📂 Dataset disponível em:
https://www.kaggle.com/competitions/open-data-challenge-ct-mon-rnp/data

### 🎯 Objetivo Principal

Desenvolver um modelo de machine learning para predizer:
- **Taxa de transmissão média** futura (5 e 10 minutos à frente)
- **Desvio padrão** da taxa de transmissão (5 e 10 minutos à frente)

Essas predições permitem antecipar problemas de qualidade de rede e otimizar a experiência do usuário em transmissões de vídeo DASH.

## 🔍 Problema de Negócio

Em redes acadêmicas, a **qualidade da transmissão de dados** é crucial para:
- Aulas online e videoconferências
- Pesquisas colaborativas
- Acesso a recursos educacionais digitais

A capacidade de **prever degradações na qualidade** permite:
- ✅ Ações preventivas de manutenção
- ✅ Roteamento inteligente de tráfego
- ✅ Melhor experiência do usuário final
- ✅ Otimização de recursos de rede

## 🏗️ Arquitetura da Solução

### Abordagem Técnica

- **Modelo**: Random Forest com múltiplas saídas (MultiOutputRegressor)
- **Features**: Extração de características estatísticas de séries temporais
- **Validação**: Validação cruzada temporal (TimeSeriesSplit)
- **Otimização**: Busca aleatória de hiperparâmetros (RandomizedSearchCV)
- **Baseline**: Comparação com modelo estatístico (média histórica)

### Variáveis Alvo

| Variável | Descrição |
|----------|-----------|
| `mean_1` | Taxa média no minuto T+5 |
| `std_1` | Desvio padrão no minuto T+5 |
| `mean_2` | Taxa média no minuto T+10 |
| `std_2` | Desvio padrão no minuto T+10 |

## 📂 Estrutura do Projeto

```
ct-mon-challenge-SI/
├── 📄 main.py                         # Script principal - orquestra todo o pipeline
├── 📄 requirements.txt                # Dependências do projeto
├── 📄 README.md                       # Este arquivo
├── 📁 src/
│   ├── 📄 data_loader.py              # Download e carregamento dos dados
│   ├── 📄 preprocessor.py             # Pré-processamento e feature engineering
│   ├── 📄 model.py                    # Modelo Random Forest e otimização
│   ├── 📄 evaluator.py                # Avaliação e métricas de performance
│   ├── 📄 utils.py                    # Utilitários e visualizações
│   └── 📁 __pycache__/                # Cache Python (ignorado no Git)
├── 📁 data/                           # Dados baixados e processados (gerada automaticamente)
│   ├── 📄 open-data.zip               # Arquivo compactado baixado do Google Drive
│   ├── 📁 Train/                      # Dados de treinamento organizados
│   │   └── 📁 dash/                   # Dados DASH por cliente/servidor
│   │       ├── 📁 ba/                 # Clientes da Bahia
│   │       ├── 📁 rj/                 # Clientes do Rio de Janeiro
│   │       └── 📁 ...                 # Outras regiões
│   └── 📁 Test/                       # Arquivos de teste para predição
│       └── 📄 *.json                  # Requisições de predição (submissão)
└── 📁 results/                        # Resultados e artefatos (gerada automaticamente)
    ├── 📄 submission.csv              # Predições finais para submissão
    ├── 📄 metrics_report.csv          # Relatório comparativo de métricas
    ├── 📄 best_hyperparameters.csv    # Melhores hiperparâmetros encontrados
    ├── 📄 features_documentation.csv  # Documentação das features criadas
    └── 📁 figures/                    # Visualizações geradas
        ├── 📊 mse_comparison.png      # Comparação MSE
        ├── 📊 mae_comparison.png      # Comparação MAE
        └── 📊 mape_comparison.png     # Comparação MAPE
```

## 🔄 Pipeline de Execução

### 1. 📥 Coleta de Dados (`data_loader.py`)

## 📁 Detalhamento das Pastas

### 📥 Pasta `data/` - Armazenamento dos Dados

Esta pasta é **criada automaticamente** pelo `data_loader.py` e contém todos os dados do **Desafio CT-MON da RNP**.

#### Estrutura dos Dados:
```
data/
├── open-data.zip            # Arquivo compactado original (Google Drive)
├── Train/                   # Dados de treinamento
│   └── dash/                # Dados DASH organizados por região
│       ├── ba/              # Clientes da Bahia
│       ├── rj/              # Clientes do Rio de Janeiro
│       ├── sp/              # Clientes de São Paulo
│       └── ...              # Outras regiões brasileiras
└── Test/                    # Dados de teste para submissão
    └── *.json               # Arquivos JSON com séries temporais
```

#### Conteúdo Detalhado:
- **`Train/dash/*/`**: Dados de treinamento organizados por pares cliente-servidor
  - Cada pasta regional contém arquivos `.json` com métricas de tráfego DASH
  - Dados incluem timestamps, taxas de transmissão e variabilidade
- **`Test/`**: Arquivos de teste com as últimas 10 observações
  - Utilizados para gerar predições dos próximos 2 instantes (T+5 e T+10 minutos)
- **`open-data.zip`**: Arquivo original baixado automaticamente via Google Drive

### 📊 Pasta `results/` - Artefatos e Resultados

Esta pasta é **gerada automaticamente** durante a execução e contém todos os **artefatos produzidos pelo pipeline**.

#### Estrutura dos Resultados:
```
results/
├── submission.csv                  # Arquivo final de submissão
├── metrics_report.csv              # Relatório de performance
├── best_hyperparameters.csv        # Hiperparâmetros otimizados
├── features_documentation.csv      # Documentação das features
└── figures/                        # Visualizações comparativas
    ├── mse_comparison.png          # Gráfico de comparação MSE
    ├── mae_comparison.png          # Gráfico de comparação MAE
    └── mape_comparison.png         # Gráfico de comparação MAPE
```

#### Descrição dos Arquivos:

| Arquivo | Descrição | Formato |
|---------|-----------|---------|
| `submission.csv` | Predições finais (`id`, `mean_1`, `std_1`, `mean_2`, `std_2`) | Pronto para submissão |
| `metrics_report.csv` | Métricas comparativas (MSE, RMSE, MAE, MAPE, R²) | Análise de performance |
| `best_hyperparameters.csv` | Melhores configurações do RandomizedSearchCV | Para reprodutibilidade |
| `features_documentation.csv` | Justificativas técnicas de cada feature criada | Documentação científica |
| `figures/*.png` | Gráficos de barras comparando modelos vs baseline | Visualização executiva |
- Download automático dos dados via Google Drive
- Extração do arquivo ZIP na pasta `data/`
- Verificação de integridade dos arquivos
- Criação automática da estrutura de pastas se não existir

### 2. 🛠️ Pré-processamento (`preprocessor.py`)

#### Agregação Temporal
- **Janelas de 5 minutos**: Reagrupamento dos dados originais
- **Sequências de 1 hora**: 12 janelas consecutivas para extração de features

#### Feature Engineering
Para cada sequência, são extraídas as seguintes características:

| Feature | Descrição | Justificativa |
|---------|-----------|---------------|
| `rate_mean` | Média das taxas nos últimos 10 intervalos | Tendência central dos dados |
| `rate_std` | Desvio padrão nos últimos 10 intervalos | Variabilidade da rede |
| `last_rate` | Última taxa observada | Estado mais recente |
| `last_std` | Último desvio padrão observado | Variabilidade recente |
| `coef_var` | Coeficiente de variação (std/mean) | Estabilidade relativa |
| `delta` | Diferença entre penúltimo e último valor | Tendência de mudança |
| `slope` | Inclinação da regressão linear | Direção da tendência |

### 3. 🤖 Modelagem (`model.py`)

#### Algoritmo Escolhido
**Random Forest** foi selecionado por:
- ✅ Robustez a outliers
- ✅ Capacidade de capturar relações não-lineares
- ✅ Interpretabilidade através da importância das features
- ✅ Performance consistente em dados tabulares

#### Otimização de Hiperparâmetros
```python
param_distributions = {
    'estimator__n_estimators': [50, 100, 200, 300],
    'estimator__max_depth': [10, 20, 30, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__max_features': ['sqrt', 'log2', None]
}
```

### 4. 📊 Avaliação (`evaluator.py`)

#### Métricas de Performance
- **MSE** (Mean Squared Error): Penaliza erros grandes
- **RMSE** (Root Mean Squared Error): Interpretável na unidade original
- **MAE** (Mean Absolute Error): Robusta a outliers
- **MAPE** (Mean Absolute Percentage Error): Erro relativo
- **R²** (Coeficiente de Determinação): Proporção da variância explicada

#### Validação Temporal
- **TimeSeriesSplit**: Respeita a ordem cronológica dos dados
- **5 folds**: Avaliação robusta da performance

### 5. 📈 Resultados e Visualizações (`utils.py`)

#### Arquivos Gerados na Pasta `results/`
- `submission.csv`: Predições finais no formato solicitado
- `metrics_report.csv`: Comparação detalhada das métricas
- `best_hyperparameters.csv`: Melhores configurações encontradas
- `features_documentation.csv`: Documentação das features criadas
- `performance_comparison.png`: Gráfico comparativo dos modelos

> **Nota**: A pasta `results/` é criada automaticamente durante a execução

## 🚀 Como Executar

### 1. Pré-requisitos
```bash
# Python 3.12 ou superior
python --version

# Instalar dependências
pip install -r requirements.txt
```

### 2. Execução
```bash
# Executar pipeline completo
python main.py
```

O script irá:
1. Baixar os dados automaticamente para `data/`
2. Executar todo o pipeline de ML
3. Gerar todos os arquivos de resultado em `results/`
4. Exibir métricas no console

> **📁 Pastas Criadas Automaticamente:**
> - `data/`: Contém os dados originais baixados e extraídos
> - `results/`: Contém todos os artefatos gerados pelo pipeline

### 3. Estrutura dos Arquivos Após Execução
```
ct-mon-challenge-SI/
├── 📁 data/
│   ├── 📄 open-data.zip               # Dataset original compactado
│   ├── 📁 Train/dash/                 # Dados de treinamento por região
│   └── 📁 Test/                       # Dados de teste (*.json)
└── 📁 results/
    ├── 📄 submission.csv              # Predições finais
    ├── 📄 metrics_report.csv          # Relatório de métricas
    ├── 📄 best_hyperparameters.csv    # Melhores hiperparâmetros
    ├── 📄 features_documentation.csv  # Documentação das features
    └── 📁 figures/                    # Visualizações (PNG)
        ├── 📊 mse_comparison.png
        ├── 📊 mae_comparison.png
        └── 📊 mape_comparison.png
```

## 📊 Resultados Obtidos

### Performance do Modelo

| Dataset | Modelo | MSE | RMSE | MAE | MAPE | R² |
|---------|--------|-----|------|-----|------|-----|
| **Treinamento** | Random Forest | 460M | 21.4K | 11.4K | 0.167 | **0.671** |
| Treinamento | Baseline | 1.38Bi | 37.2K | 22.4K | 0.250 | 0.000 |
| **Teste** | Random Forest | 477M | 21.8K | 11.5K | 0.064 | **0.654** |
| Teste | Baseline | 1.64Bi | 40.5K | 27.3K | 0.133 | -0.020 |

### 🎯 Principais Conquistas

✅ **67% de variância explicada** no conjunto de treinamento  
✅ **65% de variância explicada** no conjunto de teste  
✅ **Redução de 66% no MSE** comparado ao baseline  
✅ **Redução de 48% no MAE** comparado ao baseline  
✅ **Generalização consistente** entre treino e teste  

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.12+**: Linguagem principal
- **Scikit-learn**: Framework de machine learning
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica

### Específicas
- **RandomForestRegressor**: Algoritmo de predição
- **MultiOutputRegressor**: Múltiplas saídas simultâneas
- **TimeSeriesSplit**: Validação temporal
- **RandomizedSearchCV**: Otimização de hiperparâmetros

### Utilitários
- **Matplotlib/Seaborn**: Visualizações
- **gdown**: Download automático de datasets
- **CSV/JSON**: Persistência de dados

## 🔮 Potenciais Extensões

### Curto Prazo
- [ ] **XGBoost/LightGBM**: Algoritmos de boosting
- [ ] **Feature Selection**: Seleção automática de características
- [ ] **Cross-validation estratificada**: Por cliente/servidor

### Médio Prazo
- [ ] **LSTM/GRU**: Redes neurais recorrentes
- [ ] **Prophet**: Modelagem de sazonalidade
- [ ] **Ensemble Methods**: Combinação de modelos

### Longo Prazo
- [ ] **Graph Neural Networks**: Explorar topologia da rede
- [ ] **Real-time Prediction**: Pipeline em tempo real
- [ ] **MLOps**: Deploy com MLflow/Kubeflow

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
