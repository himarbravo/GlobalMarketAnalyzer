import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import pickle
import os

class RegimeDetector:
    """
    Detecta automáticamente el régimen actual del mercado (BULL, BEAR, NEUTRAL)
    usando Hidden Markov Models sobre los retornos del SPY.
    """
    
    def __init__(self, n_regimes=3, model_path="regime_model.pkl"):
        self.n_regimes = n_regimes
        self.model_path = model_path
        self.regime_names = ["BEAR", "NEUTRAL", "BULL"]  # Ordenados por μ esperado
        self.model = None
        
    def train(self, spy_returns, save=True):
        """
        Entrena el HMM con retornos históricos del SPY.
        
        Args:
            spy_returns: Series de pandas con retornos diarios
            save: Si True, guarda el modelo entrenado
        """
        print("🧠 Entrenando Detector de Régimen (HMM)...")
        
        # Reshape para HMM (necesita (n_samples, n_features))
        X = spy_returns.values.reshape(-1, 1)
        
        # Crear y entrenar modelo
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(X)
        
        # Ordenar estados por media (de negativo a positivo)
        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)
        
        # Reordenar parámetros del modelo
        self.model.means_ = self.model.means_[sorted_indices]
        self.model.covars_ = self.model.covars_[sorted_indices]
        self.model.transmat_ = self.model.transmat_[sorted_indices][:, sorted_indices]
        self.model.startprob_ = self.model.startprob_[sorted_indices]
        
        if save:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✅ Modelo guardado en {self.model_path}")
        
        # Mostrar estadísticas de cada régimen
        print("\n📊 Parámetros de Régimen:")
        for i, name in enumerate(self.regime_names):
            mu_daily = self.model.means_[i][0]
            sigma_daily = np.sqrt(self.model.covars_[i][0][0])
            print(f"  {name}: μ={mu_daily*252*100:.1f}% anual, σ={sigma_daily*np.sqrt(252)*100:.1f}% anual")
    
    def load_model(self):
        """Carga un modelo previamente entrenado."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Modelo cargado desde {self.model_path}")
            return True
        return False
    
    def predict_current_regime(self, recent_returns):
        """
        Predice el régimen actual basado en retornos recientes.
        
        Args:
            recent_returns: Series con últimos N retornos (recomendado 20-60 días)
        
        Returns:
            tuple: (regime_name, confidence, regime_params)
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Modelo no entrenado. Ejecuta train() primero.")
        
        X = recent_returns.values.reshape(-1, 1)
        
        # Predecir secuencia de estados
        states = self.model.predict(X)
        current_state = states[-1]
        
        # Calcular confianza (% de días recientes en este estado)
        confidence = np.mean(states[-20:] == current_state) * 100
        
        regime_name = self.regime_names[current_state]
        
        # Parámetros del régimen actual
        regime_params = {
            "mu_annual": self.model.means_[current_state][0] * 252,
            "sigma_annual": np.sqrt(self.model.covars_[current_state][0][0]) * np.sqrt(252)
        }
        
        return regime_name, confidence, regime_params
    
    def get_regime_weights(self, regime_name):
        """
        Devuelve los pesos de análisis recomendados para cada régimen.
        
        En BEAR: priorizar protección (técnico + sentiment)
        En BULL: confiar en fundamentales
        En NEUTRAL: balanced
        """
        weights = {
            "BEAR": {
                "technical": 0.40,
                "fundamental": 0.20,
                "sentiment": 0.25,
                "ito": 0.15
            },
            "NEUTRAL": {
                "technical": 0.35,
                "fundamental": 0.35,
                "sentiment": 0.20,
                "ito": 0.10
            },
            "BULL": {
                "technical": 0.30,
                "fundamental": 0.40,
                "sentiment": 0.20,
                "ito": 0.10
            }
        }
        
        return weights.get(regime_name, weights["NEUTRAL"])
