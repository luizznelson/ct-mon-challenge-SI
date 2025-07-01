import os
import json
import pandas as pd
import numpy as np
from scipy.stats import linregress
from glob import glob

# Mapeamento de clientes e servidores
CLIENTES_SERVIDORES = {'ba': 0, 'rj': 1, 'ce': 0, 'df': 1, 'es': 2, 'pi': 3}

def process_training_data(clientes_dash):
    """
    Processa dados de treinamento e retorna features e targets
    """
    X, y = [], []
    
    print("[INFO] Processando dados de treinamento...")
    
    for cliente in clientes_dash:
        servidores = glob(os.path.join(cliente, '*'))
        
        for servidor in servidores:
            requisicoes = glob(os.path.join(servidor, '*'))
            dash_values = []
            
            # Processar cada arquivo de requisição
            for file_path in requisicoes:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines[:-1]:
                        try:
                            data = json.loads(line)
                            dash_values.append({
                                'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                                'rate': data['rate']
                            })
                        except (KeyError, ValueError):
                            continue

            if not dash_values:
                continue

            # Criar série temporal
            dash_serie = pd.DataFrame(dash_values).set_index("timestamp").sort_index()
            dash_serie_5min = dash_serie.resample('5min').agg({'rate': ['mean', 'std']}).dropna()
            dash_serie_5min.columns = ['rate_mean', 'rate_std']

            # Agrupar em janelas de 1 hora (12 períodos de 5min)
            grouped = [
                dash_serie_5min.iloc[i:i + 12].copy()
                for i in range(0, dash_serie_5min.shape[0], 12)
                if i + 12 <= dash_serie_5min.shape[0]
            ]

            # Extrair features de cada grupo
            for group in grouped:
                features, targets = extract_features_and_targets(group, cliente, servidor)
                if features is not None:
                    X.append(features)
                    y.append(targets)
    
    print(f"[INFO] Total de amostras geradas: {len(X)}")
    return np.array(X), np.array(y)

def extract_features_and_targets(group, cliente, servidor):
    """
    Extrai features e targets de um grupo de dados
    """
    group.reset_index(drop=True, inplace=True)

    if group.shape[0] < 12:
        return None, None

    # Extrair features dos primeiros 10 pontos
    features = [
        CLIENTES_SERVIDORES[os.path.basename(cliente)],
        CLIENTES_SERVIDORES[os.path.basename(servidor)],
        group['rate_mean'][:10].mean(),
        group['rate_std'][:10].std(),
        group['rate_mean'][9],
        group['rate_std'][9],
        group['rate_std'][:10].mean() / (group['rate_mean'][:10].mean() + 1e-5),
        group['rate_mean'][9] - group['rate_mean'][8] if group.shape[0] >= 10 else 0,
        linregress(range(10), group['rate_mean'][:10]).slope if len(group['rate_mean'][:10].unique()) > 1 else 0
    ]
    
    # Targets são os próximos 2 pontos
    targets = [
        group['rate_mean'][10],
        group['rate_std'][10],
        group['rate_mean'][11],
        group['rate_std'][11]
    ]
    
    return features, targets

def process_test_data(test_files):
    """
    Processa dados de teste e retorna features
    """
    features = []
    
    print("[INFO] Processando dados de teste...")
    
    for test_file in test_files:
        with open(test_file, 'r') as file:
            data = json.load(file)
        
        # Extrair taxas
        rates_mean = [np.mean(dash['rate']) for dash in data['dash'] if 'rate' in dash]
        rates_std = [np.std(dash['rate']) for dash in data['dash'] if 'rate' in dash]

        if not rates_mean or not rates_std:
            rates_mean = [0] * 10
            rates_std = [0] * 10

        # Criar features
        feat = [
            CLIENTES_SERVIDORES[data['cliente']],
            CLIENTES_SERVIDORES[data['servidor']],
            np.mean(rates_mean),
            np.std(rates_mean),
            rates_mean[-1],
            rates_std[-1],
            np.std(rates_std) / (np.mean(rates_mean) + 1e-5),
            rates_mean[-1] - rates_mean[-2] if len(rates_mean) > 1 else 0,
            linregress(range(len(rates_mean)), rates_mean).slope if len(rates_mean) > 1 else 0
        ]
        features.append(feat)
    
    return np.array(features)