"""
CEFET-MG - Campus VIII - Varginha
Bacharelado em Sistemas de Informação
Disciplina: Laboratório de Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Trabalho RBF 2: Rede Neural de Função de Base Radial (RBF)
Problema: Estimação da quantidade de gasolina (y) injetada por injeção eletrônica
Entradas: x1, x2, x3 | Saída: y (linear com bias)
Topologias:
  - Rede 1: N1 = 5 neurônios escondidos
  - Rede 2: N1 = 10 neurônios escondidos
  - Rede 3: N1 = 15 neurônios escondidos
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def load_data(base_dir):
    """Carrega os dados de treinamento (150 amostras) e teste (15 amostras)."""
    train_path = os.path.join(base_dir, 'data', 'treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'teste.csv')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def train_kmeans(X, k, max_iter=300, random_state=42):
    """
    Determina os centros dos clusters via algoritmo K-Means.
    Garante reprodutibilidade via random_state.
    """
    np.random.seed(random_state)
    indices = np.random.choice(len(X), size=k, replace=False)
    centers = X[indices].copy()

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centers[j]
            for j in range(k)
        ])

        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    # Ordenar centros por norma para consistência de apresentação
    order = np.argsort(np.linalg.norm(centers, axis=1))
    centers = centers[order]
    dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    return centers, labels


def compute_sigmas(centers, X, labels, method='haykin'):
    """
    Calcula a largura/dispersão (sigma) dos neurônios da camada oculta.
    Métodos disponíveis:
      - 'haykin': sigma global = d_max / sqrt(2 * K)  (Heurística canônica de Simon Haykin)
      - 'mean_dist': sigma global = média de todas as distâncias entre centros
      - 'cluster_var': sigma_j individual = desvio-padrão dos pontos de cada cluster
    """
    k = len(centers)
    D = squareform(pdist(centers))

    if method == 'haykin':
        dmax = D.max()
        sigma = dmax / np.sqrt(2.0 * k)
        return sigma
    elif method == 'mean_dist':
        # média das distâncias entre centros distintos
        mean_d = np.sum(D) / (k * (k - 1))
        return mean_d
    elif method == 'cluster_var':
        sigmas = np.zeros(k)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 1:
                sigmas[j] = np.sqrt(np.mean(np.sum((pts - centers[j]) ** 2, axis=1)))
            else:
                sigmas[j] = D.max() / np.sqrt(2.0 * k)
        return sigmas
    else:
        raise ValueError(f"Método de sigma desconhecido: {method}")


def compute_rbf_matrix(X, centers, sigma, bias_val=-1.0):
    """
    Calcula a matriz de ativação da camada oculta Phi, incluindo termo de bias.
    bias_val = -1.0 (convenção clássica do CEFET-MG / Haykin: phi_0 = -1).
    phi_j(x) = exp( - ||x - c_j||^2 / (2 * sigma_j^2) )
    """
    N = len(X)
    k = len(centers)
    Phi = np.zeros((N, k + 1))
    Phi[:, 0] = bias_val

    dists_sq = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    if np.isscalar(sigma):
        Phi[:, 1:] = np.exp(-dists_sq / (2.0 * (sigma ** 2)))
    else:
        Phi[:, 1:] = np.exp(-dists_sq / (2.0 * (sigma[np.newaxis, :] ** 2)))

    return Phi


def calculate_eqm(Phi, d, w):
    """Calcula o Erro Quadrático Médio (EQM) no conjunto."""
    y = Phi @ w
    return float(np.mean((d - y) ** 2))


def train_lms(Phi, d, eta=0.01, epsilon=1e-7, max_epochs=20000, seed=None):
    """
    Treina os pesos da camada de saída linear utilizando a Regra Delta / LMS online (padrão a padrão).
    Inicializa pesos aleatórios entre 0 e 1.
    Critério de parada: |EQM(t) - EQM(t-1)| <= epsilon ou max_epochs.
    """
    if seed is not None:
        np.random.seed(seed)

    # Pesos iniciais aleatórios uniformes entre 0 e 1
    w = np.random.rand(Phi.shape[1])
    w_initial = w.copy()

    eqm_ant = calculate_eqm(Phi, d, w)
    history = [eqm_ant]

    N = len(d)
    for epoch in range(1, max_epochs + 1):
        # Ajuste online padrão a padrão
        for i in range(N):
            y_i = np.dot(w, Phi[i])
            e_i = d[i] - y_i
            w += eta * e_i * Phi[i]

        eqm = calculate_eqm(Phi, d, w)
        history.append(eqm)

        if abs(eqm - eqm_ant) <= epsilon:
            break

        eqm_ant = eqm

    return w_initial, w, epoch, eqm, history


def run_experiment(base_dir, heuristic='haykin'):
    """Executa o experimento completo para as três topologias com 3 treinamentos cada."""
    df_train, df_test = load_data(base_dir)
    X_train = df_train[['x1', 'x2', 'x3']].values
    d_train = df_train['d'].values
    X_test = df_test[['x1', 'x2', 'x3']].values
    d_test = df_test['d'].values

    topologias = [
        {'id': 1, 'nome': 'Rede 1', 'N1': 5},
        {'id': 2, 'nome': 'Rede 2', 'N1': 10},
        {'id': 3, 'nome': 'Rede 3', 'N1': 15},
    ]

    # Sementes distintas e reprodutíveis para T1, T2 e T3
    seeds_treinos = [101, 202, 303]

    resultados = {}

    for topo in topologias:
        k = topo['N1']
        nome = topo['nome']
        centers, labels = train_kmeans(X_train, k, max_iter=300, random_state=42)
        sigma = compute_sigmas(centers, X_train, labels, method=heuristic)

        Phi_train = compute_rbf_matrix(X_train, centers, sigma, bias_val=-1.0)
        Phi_test = compute_rbf_matrix(X_test, centers, sigma, bias_val=-1.0)

        treinos = []
        for t_idx, seed in enumerate(seeds_treinos):
            t_name = f'T{t_idx + 1}'
            w_init, w_final, epochs, final_eqm, hist = train_lms(
                Phi_train, d_train, eta=0.01, epsilon=1e-7, max_epochs=20000, seed=seed
            )
            # Predições no conjunto de teste
            y_test = Phi_test @ w_final
            er_percent = np.abs(d_test - y_test) / d_test * 100.0
            erm = float(np.mean(er_percent))
            var_erm = float(np.var(er_percent))

            treinos.append({
                'name': t_name,
                'seed': seed,
                'w_init': w_init,
                'w_final': w_final,
                'epochs': epochs,
                'eqm': final_eqm,
                'history': hist,
                'y_test': y_test,
                'erm': erm,
                'var_erm': var_erm,
            })

        resultados[nome] = {
            'N1': k,
            'centers': centers,
            'sigma': sigma,
            'treinos': treinos,
        }

    return df_train, df_test, resultados


def print_and_save_tables(df_test, resultados, base_dir):
    """Gera e exibe as tabelas dos itens 2 e 3, exportando para CSV."""
    # ------------------ TABELA ITEM 2 ------------------
    rows_item2 = []
    for t_idx, t_name in enumerate(['1o (T1)', '2o (T2)', '3o (T3)']):
        row = {'Treinamento': t_name}
        for rede_nome in ['Rede 1', 'Rede 2', 'Rede 3']:
            t_data = resultados[rede_nome]['treinos'][t_idx]
            row[f'{rede_nome}_EQM'] = f"{t_data['eqm']:.6f}"
            row[f'{rede_nome}_Epocas'] = t_data['epochs']
        rows_item2.append(row)

    df_item2 = pd.DataFrame(rows_item2)
    print("\n" + "=" * 90)
    print("ITEM 2: REGISTRO DOS RESULTADOS DOS 3 TREINAMENTOS PARA AS 3 TOPOLOGIAS")
    print("=" * 90)
    print(df_item2.to_string(index=False))

    csv_item2_path = os.path.join(base_dir, 'tabela_item2.csv')
    df_item2.to_csv(csv_item2_path, index=False)
    print(f"\n[OK] Tabela do Item 2 salva em: {csv_item2_path}")

    # ------------------ TABELA ITEM 3 ------------------
    df_item3 = pd.DataFrame()
    df_item3['Amostra'] = [f"{int(x):02d}" for x in df_test['amostra'].values]
    df_item3['x1'] = df_test['x1'].values
    df_item3['x2'] = df_test['x2'].values
    df_item3['x3'] = df_test['x3'].values
    df_item3['d'] = df_test['d'].values

    for rede_nome in ['Rede 1', 'Rede 2', 'Rede 3']:
        for t_idx in range(3):
            t_data = resultados[rede_nome]['treinos'][t_idx]
            col_name = f"{rede_nome}_y({t_data['name']})"
            df_item3[col_name] = np.round(t_data['y_test'], 4)

    # Linha de Erro Relativo Médio (%)
    erm_row = {'Amostra': 'Erro Relativo Médio (%)', 'x1': '', 'x2': '', 'x3': '', 'd': ''}
    var_row = {'Amostra': 'Variância (%)', 'x1': '', 'x2': '', 'x3': '', 'd': ''}

    for rede_nome in ['Rede 1', 'Rede 2', 'Rede 3']:
        for t_idx in range(3):
            t_data = resultados[rede_nome]['treinos'][t_idx]
            col_name = f"{rede_nome}_y({t_data['name']})"
            erm_row[col_name] = f"{t_data['erm']:.2f}%"
            var_row[col_name] = f"{t_data['var_erm']:.4f}"

    df_item3_with_metrics = pd.concat([
        df_item3,
        pd.DataFrame([erm_row]),
        pd.DataFrame([var_row])
    ], ignore_index=True)

    print("\n" + "=" * 120)
    print("ITEM 3: VALIDAÇÃO NO CONJUNTO DE TESTE (15 AMOSTRAS)")
    print("=" * 120)
    print(df_item3_with_metrics.to_string(index=False))

    csv_item3_path = os.path.join(base_dir, 'tabela_item3.csv')
    df_item3_with_metrics.to_csv(csv_item3_path, index=False)
    print(f"\n[OK] Tabela do Item 3 salva em: {csv_item3_path}")

    return df_item2, df_item3_with_metrics


def plot_convergence(resultados, output_path):
    """
    Item 4: Traça a curva de aprendizado (EQM vs Épocas) para o melhor treinamento
    de cada uma das três topologias em 3 subplots não superpostos dispostos verticalmente.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (rede_nome, cor) in enumerate(zip(['Rede 1', 'Rede 2', 'Rede 3'], cores)):
        data = resultados[rede_nome]
        # Escolhe o melhor treino pelo menor ERM no teste (ou menor EQM final)
        melhor_t = min(data['treinos'], key=lambda t: (t['erm'], t['eqm']))
        hist = melhor_t['history']
        epocas = len(hist) - 1

        ax = axes[idx]
        ax.plot(range(len(hist)), hist, color=cor, linewidth=2,
                label=f"Melhor Treino ({melhor_t['name']}) | EQM final: {melhor_t['eqm']:.6f} | {melhor_t['epochs']} épocas")
        ax.set_title(f"{rede_nome} ($N_1 = {data['N1']}$ neurônios ocultos) - Convergência do EQM",
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Época de Treinamento', fontsize=11)
        ax.set_ylabel('EQM', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[OK] Gráfico do Item 4 salvo com sucesso em: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("CEFET-MG | LAB. INTELIGÊNCIA ARTIFICIAL - TRABALHO RBF 2")
    print("Aproximação de Quantidade de Gasolina em Injeção Eletrônica")
    print("=" * 80)

    # Execução principal com heurística canônica de Haykin: sigma = d_max / sqrt(2*K)
    print("\nExecutando treinamento com Heurística de Haykin (sigma = d_max / sqrt(2*K))...")
    df_train, df_test, resultados = run_experiment(base_dir, heuristic='haykin')

    # Imprimir e salvar tabelas
    df_item2, df_item3 = print_and_save_tables(df_test, resultados, base_dir)

    # Gráficos de convergência (Item 4)
    img_out = os.path.join(base_dir, 'graficos_convergencia.png')
    plot_convergence(resultados, img_out)

    # Análise das melhores topologias
    print("\n" + "=" * 90)
    print("RESUMO COMPARATIVO DE GENERALIZAÇÃO:")
    print("=" * 90)
    for rede_nome in ['Rede 1', 'Rede 2', 'Rede 3']:
        data = resultados[rede_nome]
        for t in data['treinos']:
            print(f"{rede_nome} - {t['name']}: Épocas={t['epochs']:4d} | EQM Treino={t['eqm']:.6f} | "
                  f"ERM Teste={t['erm']:.2f}% | Variância={t['var_erm']:.4f}")

    print("\n" + "=" * 90)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 90)


if __name__ == '__main__':
    main()
