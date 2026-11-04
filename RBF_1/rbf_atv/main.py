"""
CEFET-MG - Campus VIII - Varginha
Bacharelado em Sistemas de Informação
Disciplina: Laboratório de Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Trabalho: Rede Neural RBF (Radial Basis Function) para Detecção de Radiação
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(train_path, test_path):
    """Carrega os dados de treinamento e teste a partir dos arquivos CSV."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def train_kmeans(X, k=2, max_iter=300, random_state=42):
    """
    Treina a camada escondida através do algoritmo k-means.
    Utiliza apenas os padrões com presença de radiação (d = 1).
    """
    np.random.seed(random_state)
    # Inicialização determinística com os primeiros k padrões ou k aleatórios
    centers = X[:k].copy()
    
    for epoch in range(max_iter):
        # Distância Euclidiana de cada ponto para cada centro
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        
        new_centers = np.zeros_like(centers)
        for j in range(k):
            cluster_pts = X[labels == j]
            if len(cluster_pts) > 0:
                new_centers[j] = cluster_pts.mean(axis=0)
            else:
                new_centers[j] = centers[j]
                
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    # Ordenação dos clusters para manter padrão consistente (Cluster 1 com menor x1)
    if centers[0, 0] > centers[1, 0]:
        centers = centers[::-1]
        labels = 1 - labels

    return centers, labels


def compute_variances(X, labels, centers):
    """
    Calcula a variância de cada cluster:
    - Variância populacional (MSD): (1/N) * sum ||x - c||^2
    - Variância amostral: (1/(N-1)) * sum ||x - c||^2
    """
    k = len(centers)
    var_pop = np.zeros(k)
    var_sample = np.zeros(k)
    
    for j in range(k):
        pts = X[labels == j]
        sq_dists = np.sum((pts - centers[j]) ** 2, axis=1)
        var_pop[j] = np.mean(sq_dists)
        var_sample[j] = np.sum(sq_dists) / (len(pts) - 1)
        
    return var_pop, var_sample


def rbf_activation(X, centers, variances):
    """
    Calcula a ativação gaussiana para cada neurônio oculto:
    phi_k(x) = exp( - ||x - c_k||^2 / (2 * sigma_k^2) )
    com acréscimo do bias phi_0 = -1.0.
    """
    N = len(X)
    k = len(centers)
    Phi = np.zeros((N, k + 1))
    Phi[:, 0] = -1.0  # Bias x0 = -1
    
    for i in range(N):
        for j in range(k):
            dist_sq = np.sum((X[i] - centers[j]) ** 2)
            Phi[i, j + 1] = np.exp(-dist_sq / (2.0 * variances[j]))
            
    return Phi


def calculate_eqm(Phi, d, w):
    """Calcula o Erro Quadrático Médio (EQM)."""
    preds = Phi @ w
    return np.mean((d - preds) ** 2)


def train_output_layer(Phi, d, eta=0.01, epsilon=1e-7, max_epochs=50000, seed=42):
    """
    Treina o neurônio da camada de saída via Regra Delta Generalizada:
    w = w + eta * (d - y) * phi
    Parada quando |EQM(t) - EQM(t-1)| < epsilon.
    """
    np.random.seed(seed)
    w = np.zeros(Phi.shape[1])  # ou np.random.rand(Phi.shape[1])
    
    eqm_ant = calculate_eqm(Phi, d, w)
    history = [eqm_ant]
    
    for epoch in range(1, max_epochs + 1):
        # Atualização online (padrão a padrão)
        for i in range(len(d)):
            y = np.dot(w, Phi[i])
            erro = d[i] - y
            w += eta * erro * Phi[i]
            
        eqm = calculate_eqm(Phi, d, w)
        history.append(eqm)
        
        if abs(eqm - eqm_ant) < epsilon:
            break
            
        eqm_ant = eqm
        
    return w, epoch, eqm, history


def pos_process(y):
    """
    Função de pós-processamento utilizando a função sinal:
    ypós = 1 se y >= 0, senão -1.
    """
    return np.where(y >= 0.0, 1, -1)


def evaluate_test(df_test, centers, variances, weights):
    """Aplica a RBF treinada sobre o conjunto de teste e monta a tabela de validação."""
    X_test = df_test[['x1', 'x2']].values
    d_test = df_test['d'].values
    
    Phi_test = rbf_activation(X_test, centers, variances)
    y_raw = Phi_test @ weights
    y_pos = pos_process(y_raw)
    
    tabela = df_test[['amostra', 'x1', 'x2', 'd']].copy()
    tabela['y'] = np.round(y_raw, 4)
    tabela['y_pos'] = y_pos
    tabela['acerto'] = (tabela['y_pos'] == tabela['d'])
    
    acertos = tabela['acerto'].sum()
    taxa_acerto = (acertos / len(tabela)) * 100.0
    
    return tabela, taxa_acerto


def plot_charts(df_train, df_test, centers, variances, weights, history, output_path):
    """Gera visualização gráfica da fronteira de decisão e do EQM."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Superfície de decisão e dispersão das amostras
    ax1 = axes[0]
    
    # Grade para a fronteira de decisão
    x1_grid = np.linspace(-0.05, 1.05, 300)
    x2_grid = np.linspace(-0.05, 1.05, 300)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_pts = np.c_[X1.ravel(), X2.ravel()]
    
    Phi_grid = rbf_activation(grid_pts, centers, variances)
    Z = (Phi_grid @ weights).reshape(X1.shape)
    
    # Contorno preenchido da função contínua y
    cf = ax1.contourf(X1, X2, Z, levels=np.linspace(-1.5, 1.5, 13), cmap='RdBu_r', alpha=0.35)
    plt.colorbar(cf, ax=ax1, label='Saída contínua $y$')
    
    # Linha de decisão y = 0
    cs = ax1.contour(X1, X2, Z, levels=[0.0], colors='black', linewidths=2.5, linestyles='--')
    ax1.clabel(cs, fmt='y = 0 (Limiar)', fontsize=10)
    
    # Círculos indicando o raio de influência das gaussianas (1 desvio padrão)
    theta = np.linspace(0, 2*np.pi, 200)
    for j, (c, v) in enumerate(zip(centers, variances)):
        std = np.sqrt(v)
        ax1.plot(c[0] + std * np.cos(theta), c[1] + std * np.sin(theta), 
                 color='purple', linestyle=':', linewidth=1.8, 
                 label=f'Raio $\\sigma$ Cluster {j+1}' if j == 0 else "")
    
    # Padrões de treino
    tr_pos = df_train[df_train['d'] == 1]
    tr_neg = df_train[df_train['d'] == -1]
    ax1.scatter(tr_pos['x1'], tr_pos['x2'], c='blue', marker='o', s=45, label='Treino: Radiação ($d=+1$)', edgecolors='k')
    ax1.scatter(tr_neg['x1'], tr_neg['x2'], c='red', marker='x', s=45, label='Treino: Ausência ($d=-1$)', linewidths=1.5)
    
    # Centros dos clusters
    ax1.scatter(centers[:, 0], centers[:, 1], c='gold', marker='*', s=250, edgecolors='black', label='Centros K-means ($c_k$)', zorder=5)
    for j, c in enumerate(centers):
        ax1.annotate(f'C{j+1}', (c[0] + 0.02, c[1] + 0.02), fontweight='bold', fontsize=11, color='darkgreen')
        
    # Padrões de teste
    te_pos = df_test[df_test['d'] == 1]
    te_neg = df_test[df_test['d'] == -1]
    ax1.scatter(te_pos['x1'], te_pos['x2'], c='cyan', marker='s', s=80, label='Teste: Radiação ($d=+1$)', edgecolors='black')
    ax1.scatter(te_neg['x1'], te_neg['x2'], c='magenta', marker='^', s=80, label='Teste: Ausência ($d=-1$)', edgecolors='black')
    
    ax1.set_title('Superfície de Decisão da RBF e Distribuição dos Padrões', fontsize=12, fontweight='bold')
    ax1.set_xlabel('$x_1$ (Concentração 1)', fontsize=11)
    ax1.set_ylabel('$x_2$ (Concentração 2)', fontsize=11)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Gráfico 2: Curva de convergência do EQM
    ax2 = axes[1]
    ax2.plot(history, color='navy', linewidth=2)
    ax2.set_title(f'Convergência do Treinamento da Camada de Saída ({len(history)-1} épocas)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('Erro Quadrático Médio (EQM)', fontsize=11)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'data', 'treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'teste.csv')
    
    print("=" * 80)
    print("CEFET-MG - TRABALHO DE LAB. INTELIGÊNCIA ARTIFICIAL: REDE RBF")
    print("=" * 80)
    
    df_train, df_test = load_data(train_path, test_path)
    print(f"Conjunto de Treinamento carregado: {len(df_train)} amostras.")
    print(f"Conjunto de Teste carregado: {len(df_test)} amostras.")
    
    # -------------------------------------------------------------
    # ITEM 1: Treinamento da Camada Escondida via K-means
    # -------------------------------------------------------------
    pos_samples = df_train[df_train['d'] == 1][['x1', 'x2']].values
    print(f"\nPadrões com presença de radiação (d = 1) no treino: {len(pos_samples)}")
    
    centers, labels = train_kmeans(pos_samples, k=2, max_iter=300)
    var_pop, var_samp = compute_variances(pos_samples, labels, centers)
    
    print("\n" + "=" * 80)
    print("ITEM 1: CENTROS E VARIÂNCIAS DOS CLUSTERS (CAMADA ESCONDIDA)")
    print("=" * 80)
    tabela_item1 = pd.DataFrame({
        'Cluster': [1, 2],
        'Centro (x1, x2)': [f"({centers[0,0]:.4f}, {centers[0,1]:.4f})", 
                            f"({centers[1,0]:.4f}, {centers[1,1]:.4f})"],
        'Variância Populacional (MSD)': [f"{var_pop[0]:.6f}", f"{var_pop[1]:.6f}"],
        'Variância Amostral (N-1)': [f"{var_samp[0]:.6f}", f"{var_samp[1]:.6f}"],
        'Amostras': [(labels == 0).sum(), (labels == 1).sum()]
    })
    print(tabela_item1.to_string(index=False))
    
    # -------------------------------------------------------------
    # ITEM 2: Treinamento da Camada de Saída via Regra Delta Generalizada
    # -------------------------------------------------------------
    X_train = df_train[['x1', 'x2']].values
    d_train = df_train['d'].values
    
    # Usando a variância populacional (MSD: 1/N sum ||x-c||^2) padrão de RBF
    Phi_train = rbf_activation(X_train, centers, var_pop)
    
    eta = 0.01
    epsilon = 1e-7
    weights, epochs, final_eqm, history = train_output_layer(Phi_train, d_train, eta=eta, epsilon=epsilon)
    
    print("\n" + "=" * 80)
    print("ITEM 2: PESOS FINAIS DA CAMADA DE SAÍDA (REGRA DELTA GENERALIZADA)")
    print(f"Parâmetros: eta = {eta}, epsilon = {epsilon}")
    print(f"Convergência atingida em {epochs} épocas com EQM final = {final_eqm:.7f}")
    print("=" * 80)
    tabela_item2 = pd.DataFrame({
        'Peso': ['W2_1,0 (Bias)', 'W2_1,1', 'W2_1,2'],
        'Valor': [f"{weights[0]:.4f}", f"{weights[1]:.4f}", f"{weights[2]:.4f}"]
    })
    print(tabela_item2.to_string(index=False))
    
    # -------------------------------------------------------------
    # ITENS 3 e 4: Pós-processamento e Validação no Conjunto de Teste
    # -------------------------------------------------------------
    tabela_teste, taxa_acerto = evaluate_test(df_test, centers, var_pop, weights)
    
    print("\n" + "=" * 80)
    print("ITEM 4: TABELA DE VALIDAÇÃO (CONJUNTO DE TESTE)")
    print("=" * 80)
    print(tabela_teste[['amostra', 'x1', 'x2', 'd', 'y', 'y_pos', 'acerto']].to_string(index=False))
    print("-" * 80)
    print(f"Taxa de Acerto (%): {taxa_acerto:.2f}% ({tabela_teste['acerto'].sum()}/{len(tabela_teste)})")
    print("=" * 80)
    
    # Salvar tabela de validação em CSV
    csv_out = os.path.join(base_dir, 'tabela_item4.csv')
    tabela_teste.to_csv(csv_out, index=False)
    print(f"Tabela de validação exportada para: {csv_out}")
    
    # Gerar gráfico
    img_out = os.path.join(base_dir, 'grafico_rbf.png')
    plot_charts(df_train, df_test, centers, var_pop, weights, history, img_out)
    print(f"Gráfico visual gerado e salvo em: {img_out}")
    
    return centers, var_pop, var_samp, weights, epochs, final_eqm, tabela_teste, taxa_acerto


if __name__ == '__main__':
    main()
