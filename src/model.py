import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

class CTMonModel:
    """
    Modelo para previsão de métricas CT-MON
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
    
    def train(self, X, y):
        """
        Treina o modelo com os dados fornecidos
        """
        print("[INFO] Normalizando os dados...")
        
        # Tratar valores inválidos
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        print("[INFO] Configurando o modelo...")
        
        # Definir modelo base
        rf_model = RandomForestRegressor(random_state=self.random_state)
        
        # Configurar busca de hiperparâmetros
        param_grid = {
            'n_estimators': [300, 500, 800],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt'],
            'bootstrap': [True, False]
        }
        
        # Busca randomizada com validação temporal
        search = RandomizedSearchCV(
            rf_model,
            param_distributions=param_grid,
            n_iter=50,
            cv=TimeSeriesSplit(n_splits=5),
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Modelo multi-output
        self.model = MultiOutputRegressor(search)
        
        print("[INFO] Treinando o modelo...")
        self.model.fit(X_scaled, y_clean)
        self.is_trained = True
        
        print("[INFO] Treinamento concluído.")
        
        return self
    
    def predict(self, X):
        """
        Faz predições com o modelo treinado
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro!")
        
        # Tratar valores inválidos e normalizar
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_clean)
        
        # Fazer predições
        predictions = self.model.predict(X_scaled)
        
        # Tratar valores inválidos nas predições
        return np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    
    def get_best_params(self):
        """
        Retorna os melhores parâmetros encontrados
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro!")
        
        best_params = []
        for i, estimator in enumerate(self.model.estimators_):
            best_params.append(estimator.best_params_)
            print(f"Melhores parâmetros para a saída {i + 1}: {estimator.best_params_}")
        
        return best_params