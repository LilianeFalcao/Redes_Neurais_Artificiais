import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(u):
    """Função de ativação sigmoide logística."""
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))

def sigmoid_derivative(y):
    """Derivada da função de ativação sigmoide em função da saída y = g(u)."""
    return y * (1.0 - y)

def forward_pass(x_sample, W1, W2):
    """
    Executa a propagação progressiva (Forward pass) para uma amostra.
    x_sample: vetor de entrada incluindo o bias x0=-1 (shape: 4,)
    W1: matriz de pesos camada escondida (shape: 10, 4)
    W2: matriz de pesos camada de saída (shape: 1, 11)
    """
    u1 = np.dot(W1, x_sample)              # potencial da camada oculta (10,)
    y1 = sigmoid(u1)                       # ativação da camada oculta (10,)
    y1_bias = np.insert(y1, 0, -1.0)       # adiciona bias da camada oculta y0 = -1 (11,)
    u2 = np.dot(W2, y1_bias)               # potencial do neurônio de saída (1,)
    y2 = sigmoid(u2)                       # saída final da rede (1,)
    return y1, y1_bias, y2[0]

def calculate_eqm(X_bias, d, W1, W2):
    """Calcula o Erro Quadrático Médio (EQM) para todo o conjunto."""
    N_samples = len(X_bias)
    sum_sq_error = 0.0
    for i in range(N_samples):
        _, _, y = forward_pass(X_bias[i], W1, W2)
        sum_sq_error += (d[i] - y) ** 2
    return sum_sq_error / N_samples

def train_mlp(X_bias, d, eta=0.1, epsilon=1e-6, max_epochs=20000):
    """
    Treina o Perceptron Multicamadas (3 -> 10 -> 1) com o algoritmo Backpropagation.
    Ajuste no modo padrão a padrão (estocástico/online).
    """
    # Inicializa matrizes de pesos com valores aleatórios uniformes entre 0 e 1
    # W1: 10 neurônios, 4 entradas (x0=-1, x1, x2, x3)
    # W2: 1 neurônio, 11 entradas (y0=-1, y1 ... y10)
    W1 = np.random.rand(10, 4)
    W2 = np.random.rand(1, 11)
    
    epochs = 0
    eqm_history = []
    
    current_eqm = calculate_eqm(X_bias, d, W1, W2)
    eqm_history.append(current_eqm)
    
    while epochs < max_epochs:
        # Apresentação padrão a padrão
        for i in range(len(X_bias)):
            x_i = X_bias[i]
            d_i = d[i]
            
            # Forward pass
            y1, y1_bias, y2 = forward_pass(x_i, W1, W2)
            
            # Cálculo do erro e gradiente da saída
            error = d_i - y2
            delta_out = error * sigmoid_derivative(y2)  # escalar
            
            # Retropropagação para a camada oculta (desconsiderando o bias no gradiente)
            # W2[0, 1:] corresponde aos pesos conectados às saídas y1
            delta_hidden = delta_out * W2[0, 1:] * sigmoid_derivative(y1)  # (10,)
            
            # Atualização dos pesos (Regra Delta Generalizada)
            W2 += eta * delta_out * y1_bias.reshape(1, -1)
            W1 += eta * delta_hidden.reshape(-1, 1) * x_i.reshape(1, -1)
            
        epochs += 1
        new_eqm = calculate_eqm(X_bias, d, W1, W2)
        eqm_history.append(new_eqm)
        
        # Critério de parada: variação do EQM menor que epsilon
        if abs(new_eqm - current_eqm) < epsilon:
            break
            
        current_eqm = new_eqm
        
    return W1, W2, epochs, new_eqm, eqm_history

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'data', 'conjunto_treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'conjunto_validacao.csv')
    
    # 1. Carregar conjunto de treinamento
    df_train = pd.read_csv(train_path)
    X_train = df_train[['x1', 'x2', 'x3']].values
    d_train = df_train['d'].values
    
    # Adicionar o bias x0 = -1
    X_train_bias = np.c_[np.full((len(X_train), 1), -1.0), X_train]
    
    print("=" * 70)
    print("ATIVIDADE: PERCEPTRON MULTICAMADAS (PMC) - LAB. INTELIGÊNCIA ARTIFICIAL")
    print("=" * 70)
    print(f"Número de padrões de treinamento: {len(X_train)}")
    print("Iniciando os 5 treinamentos (eta = 0.1, epsilon = 1e-6)...")
    print("-" * 70)
    
    trainings = []
    
    for t in range(5):
        # Reiniciar gerador para garantir pesos iniciais distintos
        np.random.seed(None)
        W1, W2, epochs, final_eqm, history = train_mlp(X_train_bias, d_train, eta=0.1, epsilon=1e-6)
        trainings.append({
            'id': t + 1,
            'name': f'T{t+1}',
            'W1': W1,
            'W2': W2,
            'epochs': epochs,
            'final_eqm': final_eqm,
            'history': history
        })
        print(f"Treinamento {t+1} (T{t+1}): EQM Final = {final_eqm:.6f} | Número de Épocas = {epochs}")
        
    # Tabela do Item 2
    print("\n" + "=" * 70)
    print("TABELA ITEM 2: RESULTADOS FINAIS DOS 5 TREINAMENTOS")
    print("=" * 70)
    df_item2 = pd.DataFrame({
        'Treinamento': [f"{t['id']}º ({t['name']})" for t in trainings],
        'Erro Quadrático Médio': [f"{t['final_eqm']:.6f}" for t in trainings],
        'Número de Épocas': [t['epochs'] for t in trainings]
    })
    print(df_item2.to_string(index=False))
    
    # Item 3: Identificar os dois treinamentos com maiores números de épocas
    sorted_by_epochs = sorted(trainings, key=lambda x: x['epochs'], reverse=True)
    top2 = sorted_by_epochs[:2]
    print("\n" + "=" * 70)
    print(f"ITEM 3: TREINAMENTOS COM MAIORES NÚMEROS DE ÉPOCAS")
    print(f"1º Maior: {top2[0]['name']} com {top2[0]['epochs']} épocas")
    print(f"2º Maior: {top2[1]['name']} com {top2[1]['epochs']} épocas")
    print("=" * 70)
    
    # Gerar gráficos lado a lado (não superpostos)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1
    axes[0].plot(top2[0]['history'], color='royalblue', linewidth=2)
    axes[0].set_title(f"Evolução do EQM - {top2[0]['name']} ({top2[0]['epochs']} épocas)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Número de Épocas", fontsize=11)
    axes[0].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico 2
    axes[1].plot(top2[1]['history'], color='darkorange', linewidth=2)
    axes[1].set_title(f"Evolução do EQM - {top2[1]['name']} ({top2[1]['epochs']} épocas)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Número de Épocas", fontsize=11)
    axes[1].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    chart_path = os.path.join(base_dir, 'graficos_eqm.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Gráficos salvos com sucesso em: {chart_path}")
    
    # Item 5: Validação com conjunto de teste
    df_val = pd.read_csv(test_path)
    X_val = df_val[['x1', 'x2', 'x3']].values
    d_val = df_val['d'].values
    X_val_bias = np.c_[np.full((len(X_val), 1), -1.0), X_val]
    
    predictions = {}
    for t in trainings:
        preds = []
        for i in range(len(X_val_bias)):
            _, _, y = forward_pass(X_val_bias[i], t['W1'], t['W2'])
            preds.append(y)
        predictions[t['name']] = np.array(preds)
        
    # Calcular erros relativos percentuais e variância para cada treinamento
    erros_rel_medios = {}
    variancias = {}
    for t in trainings:
        t_name = t['name']
        preds = predictions[t_name]
        # Erro relativo percentual: |d - y| / d * 100
        er_percent = np.abs(d_val - preds) / d_val * 100.0
        erros_rel_medios[t_name] = np.mean(er_percent)
        variancias[t_name] = np.var(er_percent)
        
    print("\n" + "=" * 70)
    print("TABELA ITEM 5: RESULTADOS DA VALIDAÇÃO (CONJUNTO DE TESTE)")
    print("=" * 70)
    
    table_data = {
        'Amostra': df_val['Amostra'].values,
        'x1': df_val['x1'].values,
        'x2': df_val['x2'].values,
        'x3': df_val['x3'].values,
        'd': df_val['d'].values,
    }
    for t in trainings:
        table_data[f"yrede({t['name']})"] = np.round(predictions[t['name']], 4)
        
    df_validation_table = pd.DataFrame(table_data)
    print(df_validation_table.to_string(index=False))
    
    print("-" * 70)
    print("MÉTRICAS DE ERRO NO CONJUNTO DE TESTE:")
    for t in trainings:
        t_name = t['name']
        print(f"{t_name}: Erro Relativo Médio = {erros_rel_medios[t_name]:.4f}% | Variância = {variancias[t_name]:.4f}%")
        
    # Item 6: Identificar melhor generalização
    best_t = min(trainings, key=lambda t: erros_rel_medios[t['name']])
    print("=" * 70)
    print(f"ITEM 6: MELHOR CONFIGURAÇÃO DE GENERALIZAÇÃO: {best_t['name']}")
    print(f"Apresentou o menor Erro Relativo Médio de {erros_rel_medios[best_t['name']]:.4f}% "
          f"com Variância de {variancias[best_t['name']]:.4f}%")
    print("=" * 70)
    
    return trainings, df_item2, top2, df_validation_table, erros_rel_medios, variancias, best_t

if __name__ == '__main__':
    main()
