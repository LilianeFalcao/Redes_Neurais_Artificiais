import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_eqm(X, d, w):
    """Calcula o Erro Quadrático Médio (EQM) para todo o conjunto de dados."""
    eqm = 0
    for i in range(len(X)):
        u = np.dot(w, X[i])
        eqm += (d[i] - u) ** 2
    return eqm / len(X)

def train_adaline(X, d, eta=0.0025, epsilon=1e-6, max_epochs=10000):
    """
    Treina a rede ADALINE utilizando a Regra Delta.
    Retorna os pesos iniciais, pesos finais, número de épocas e histórico do EQM.
    """
    # Inicializa pesos com valores aleatórios entre 0 e 1
    w = np.random.rand(X.shape[1])
    w_initial = w.copy()
    
    epochs = 0
    eqm_history = []
    
    # Calcula e guarda o EQM da configuração inicial (Época 0)
    current_eqm = calculate_eqm(X, d, w)
    eqm_history.append(current_eqm)
    
    while epochs < max_epochs:
        # Atualização online: peso é atualizado a cada padrão/amostra
        for i in range(len(X)):
            x_i = X[i]
            d_i = d[i]
            
            # v (ou u) = W * X
            u = np.dot(w, x_i)
            
            # Atualização de pesos: Regra Delta -> w = w + eta * (d - u) * x
            w = w + eta * (d_i - u) * x_i
            
        # Calcula novo EQM após a época (após apresentar todos os padrões)
        new_eqm = calculate_eqm(X, d, w)
        eqm_history.append(new_eqm)
        epochs += 1
        
        # Critério de parada: variação do erro menor que a precisão estabelecida
        if abs(new_eqm - current_eqm) < epsilon:
            break
            
        current_eqm = new_eqm
        
    return w_initial, w, epochs, eqm_history

def sign(u):
    """Função de ativação degrau bipolar."""
    return 1 if u >= 0 else -1

def test_adaline(X, w):
    """Aplica o modelo treinado nos dados de teste."""
    predictions = []
    for i in range(len(X)):
        u = np.dot(w, X[i])
        predictions.append(sign(u))
    return predictions

def main():
    # Caminhos para os arquivos
    train_path = '../data/adaline_treinamento.csv'
    test_path = '../data/adaline_teste.csv'
    
    # Carregar dados de treinamento
    df_train = pd.read_csv(train_path)
    
    # Preparar X e d
    # O dataset possui as colunas: x1, x2, x3, x4. Precisamos adicionar x0 = -1 (bias)
    X_train = np.c_[np.full(len(df_train), -1), df_train['x1'], df_train['x2'], df_train['x3'], df_train['x4']]
    d_train = df_train['d'].values
    
    # Estruturas para guardar os resultados dos 5 treinamentos
    models = []
    histories = []
    
    print("--- Resultados de Treinamento (ADALINE) ---")
    for t in range(5):
        w_init, w_final, epochs, eqm_history = train_adaline(X_train, d_train)
        models.append(w_final)
        histories.append(eqm_history)
        
        print(f"Treinamento {t+1}:")
        print(f"  Vetor de Pesos Inicial : w0={w_init[0]:.4f}, w1={w_init[1]:.4f}, w2={w_init[2]:.4f}, w3={w_init[3]:.4f}, w4={w_init[4]:.4f}")
        print(f"  Vetor de Pesos Final   : w0={w_final[0]:.4f}, w1={w_final[1]:.4f}, w2={w_final[2]:.4f}, w3={w_final[3]:.4f}, w4={w_final[4]:.4f}")
        print(f"  Número de Épocas       : {epochs}")
        print("-" * 60)

    # Plotar o Gráfico EQM x Épocas para os dois primeiros treinamentos
    plt.figure(figsize=(10, 5))
    plt.plot(histories[0], label='Treinamento 1 (T1)', color='blue', linewidth=2)
    plt.plot(histories[1], label='Treinamento 2 (T2)', color='orange', linewidth=2, linestyle='--')
    plt.title('Erro Quadrático Médio (EQM) vs Épocas de Treinamento')
    plt.xlabel('Número de Épocas')
    plt.ylabel('EQM')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Salva a imagem no diretório atual
    plt.savefig('grafico_eqm.png')
    print("\nGráfico 'grafico_eqm.png' gerado com sucesso para os dois primeiros treinamentos.\n")
    
    # Carregar dados de teste
    df_test = pd.read_csv(test_path)
    
    # Preparar matriz de teste
    X_test = np.c_[np.full(len(df_test), -1), df_test['x1'], df_test['x2'], df_test['x3'], df_test['x4']]
    
    # Classificar e gerar tabela
    print("--- Resultados da Classificação (Teste) ---")
    predictions_table = {"Amostra": df_test["Padrão"].values if "Padrão" in df_test.columns else df_test.index + 1}
    
    for t in range(5):
        preds = test_adaline(X_test, models[t])
        predictions_table[f"y(T{t+1})"] = preds
        
    df_results = pd.DataFrame(predictions_table)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
