import pandas as pd
import numpy as np
import random
import os

# Função de ativação (degrau bipolar)
def sign(v):
    return 1 if v >= 0 else -1

# Treinamento do Perceptron
def train_perceptron(X, d, eta=0.01, max_epochs=10000):
    # Inicializa os pesos com valores aleatórios entre 0 e 1
    w = np.random.rand(X.shape[1])
    w_initial = w.copy()
    
    epochs = 0
    while epochs < max_epochs:
        error_count = 0
        for i in range(len(X)):
            x_i = X[i]
            d_i = d[i]
            
            # v = w^T * x
            v = np.dot(w, x_i)
            y = sign(v)
            
            # Atualização de pesos (Regra do Perceptron)
            if y != d_i:
                w = w + eta * (d_i - y) * x_i
                error_count += 1
                
        epochs += 1
        if error_count == 0:
            break
            
    return w_initial, w, epochs

# Classificação
def test_perceptron(X, w):
    predictions = []
    for i in range(len(X)):
        v = np.dot(w, X[i])
        predictions.append(sign(v))
    return predictions

def main():
    # Carregar dados de treinamento
    df_train = pd.read_csv('../data/oleo_dataset.csv')
    
    # Preparar X e d
    # x0 = -1 (bias)
    X_train = np.c_[np.full(len(df_train), -1), df_train['x1'], df_train['x2'], df_train['x3']]
    d_train = df_train['d'].values
    
    # Executar 5 treinamentos
    print("--- Resultados de Treinamento ---")
    models = []
    for t in range(5):
        w_init, w_final, epochs = train_perceptron(X_train, d_train, eta=0.01)
        models.append(w_final)
        print(f"Treinamento {t+1}:")
        print(f"  Vetor de Pesos Inicial : w0={w_init[0]:.4f}, w1={w_init[1]:.4f}, w2={w_init[2]:.4f}, w3={w_init[3]:.4f}")
        print(f"  Vetor de Pesos Final   : w0={w_final[0]:.4f}, w1={w_final[1]:.4f}, w2={w_final[2]:.4f}, w3={w_final[3]:.4f}")
        print(f"  Número de Épocas       : {epochs}")
        print("-" * 40)
        
    # Carregar dados de teste
    df_test = pd.read_csv('../data/oleo_teste.csv')
    X_test = np.c_[np.full(len(df_test), -1), df_test['x1'], df_test['x2'], df_test['x3']]
    
    # Classificar
    print("\n--- Resultados da Classificação ---")
    predictions_table = {"Amostra": df_test["Amostra"].values}
    
    for t in range(5):
        preds = test_perceptron(X_test, models[t])
        predictions_table[f"y(T{t+1})"] = preds
        
    df_results = pd.DataFrame(predictions_table)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
