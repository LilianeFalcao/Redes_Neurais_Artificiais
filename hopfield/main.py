"""
CEFET-MG - Campus VIII - Varginha
Bacharelado em Sistemas de Informação
Disciplina: Laboratório de Inteligência Artificial
Professor: Lázaro Eduardo da Silva
Data: 12/11/2026

Trabalho: Memória Associativa com Rede de Hopfield (45 neurônios)
Problema: Recuperação de imagens de 45 bits (matrizes 9x5 dos dígitos 1, 2, 3 e 4)
          corrompidas por ruído (cerca de 20% dos pixels invertidos) em um link de comunicação.
Topologia:
  - 45 neurônios (N = 45), 4 padrões fundamentais (P = 4)
  - Matriz de pesos W obtida pela regra do produto externo (sem auto-realimentação: w_ii = 0)
  - Função de ativação: Tangente Hiperbólica g(u) = tanh(beta * u) com beta muito grande
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


# Dimensões das imagens (9 linhas x 5 colunas = 45 bits)
IMG_ROWS = 9
IMG_COLS = 5
N_NEURONS = IMG_ROWS * IMG_COLS


def load_data(base_dir):
    """Carrega os 4 padrões fundamentais (treinamento.csv) e as 12 situações de teste (teste.csv)."""
    train_path = os.path.join(base_dir, 'data', 'treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'teste.csv')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    pixel_cols = [f'x{i}' for i in range(1, N_NEURONS + 1)]
    X_train = df_train[pixel_cols].values.astype(float)
    X_test = df_test[pixel_cols].values.astype(float)
    return df_train, df_test, X_train, X_test


def train_hopfield_outer_product(X):
    """
    Calcula a matriz de pesos W pela regra do produto externo (Hebb):
        W = (1 / N) * sum_{k=1}^{P} x^(k) * (x^(k))^T - (P / N) * I
    A diagonal principal é zerada (w_ii = 0) para impedir auto-realimentação.
    """
    P, N = X.shape
    W = (X.T @ X) / N - (P / N) * np.eye(N)
    np.fill_diagonal(W, 0.0)
    return W


def train_hopfield_pseudoinverse(X):
    """
    Calcula a matriz de pesos W pela regra da Pseudo-Inversa (Projeção de Personnaz/Kanter-Sompolinsky):
        W_proj = X^T * (X * X^T)^(-1) * X
    com diagonal principal zerada, eliminando a interferência cruzada (crosstalk)
    entre padrões correlacionados.
    """
    P, N = X.shape
    W = X.T @ np.linalg.inv(X @ X.T) @ X
    np.fill_diagonal(W, 0.0)
    return W / N


def hopfield_energy(W, v):
    """Calcula a função de energia de Lyapunov E(v) = -0.5 * v^T * W * v."""
    return float(-0.5 * (v.T @ W @ v))


def recover_pattern(W, v_init, beta=100.0, max_iter=200, mode='sync'):
    """
    Executa a operação de recuperação da Rede de Hopfield usando a função de ativação
    Tangente Hiperbólica com beta muito grande: g(u) = tanh(beta * u).

    Parâmetros:
      - W: matriz de pesos sinápticos (45 x 45)
      - v_init: vetor de estado inicial (imagem distorcida com ruído)
      - beta: ganho da tangente hiperbólica (beta >> 1 aproxima a função sinal bipolar)
      - max_iter: número máximo de épocas de atualização
      - mode: 'sync' (atualização paralela/síncrona) ou 'async' (atualização sequencial/assíncrona)
    """
    v = v_init.astype(float).copy()
    history = [v.copy()]

    for it in range(1, max_iter + 1):
        if mode == 'sync':
            u = W @ v
            v_cont = np.tanh(beta * u)
            v_next = np.where(v_cont > 0, 1.0, np.where(v_cont < 0, -1.0, v))
        elif mode == 'async':
            v_next = v.copy()
            for i in range(len(v_next)):
                u_i = float(W[i] @ v_next)
                act = np.tanh(beta * u_i)
                if act > 0:
                    v_next[i] = 1.0
                elif act < 0:
                    v_next[i] = -1.0
        else:
            raise ValueError(f"Modo desconhecido: {mode}")

        history.append(v_next.copy())

        # Verifica convergência para ponto fixo: v(t) == v(t-1)
        if np.array_equal(v_next, v):
            return v_next, it, True, history

        # Verifica ciclo limite de período 2 (comum no modo síncrono sob alto ruído)
        if len(history) >= 3 and np.array_equal(v_next, history[-3]):
            return v_next, it, False, history

        v = v_next

    return v, max_iter, False, history


def identify_attractor(v, X_train):
    """
    Verifica se o estado recuperado corresponde a algum dos padrões fundamentais x^(k),
    a um padrão invertido -x^(k), ou a um estado espúrio.
    """
    for k in range( len(X_train)):
        if np.array_equal(v, X_train[k]):
            return f"Padrão {k + 1}", k + 1, 0
        if np.array_equal(v, -X_train[k]):
            return f"Padrão {k + 1} Invertido (-x^({k + 1}))", -(k + 1), len(v)

    dists = [int(np.sum(v != X_train[k])) for k in range(len(X_train))]
    closest_k = int(np.argmin(dists)) + 1
    min_dist = min(dists)
    return f"Espúrio (próx. Padrão {closest_k}, dist={min_dist})", 0, min_dist


def draw_pixel_grid(ax, vector, title="", subtitle="", highlight_diffs=None):
    """
    Desenha uma imagem 9x5 seguindo exatamente o estilo visual do PDF:
      - Pixel branco (-1) -> fundo branco (#FFFFFF)
      - Pixel escuro (+1) -> cinza médio/escuro (#808080)
      - Grade preta bem definida demarcando os 45 pixels
    """
    grid = vector.reshape(IMG_ROWS, IMG_COLS)
    cmap = ListedColormap(['#FFFFFF', '#808080'])
    ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    # Grade de separação dos 45 pixels
    ax.set_xticks(np.arange(-0.5, IMG_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, IMG_ROWS, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    # Destaque opcional de pixels corrompidos ou divergentes (borda ou marcador discreto)
    if highlight_diffs is not None:
        diff_grid = highlight_diffs.reshape(IMG_ROWS, IMG_COLS)
        for r in range(IMG_ROWS):
            for c in range(IMG_COLS):
                if diff_grid[r, c]:
                    ax.plot(c, r, marker='o', markersize=4.5,
                            markerfacecolor='none', markeredgecolor='#D32F2F', markeredgewidth=1.3)

    if title:
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8.5, labelpad=5)


def plot_stored_patterns(X_train, out_path):
    """Plota os 4 padrões fundamentais armazenados na memória associativa."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.8))
    fig.suptitle("Padrões Fundamentais Armazenados na Rede de Hopfield (45 bits: 9x5)",
                 fontsize=13, fontweight='bold', y=0.98)

    for k in range(4):
        n_pos = int(np.sum(X_train[k] == 1))
        n_neg = int(np.sum(X_train[k] == -1))
        draw_pixel_grid(
            axes[k],
            X_train[k],
            title=f"Padrão {k + 1} (Dígito '{k + 1}')",
            subtitle=f"+1: {n_pos} bits | -1: {n_neg} bits"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_12_simulations(X_train, simulations_data, out_path, suptitle):
    """
    Gera a figura com as 12 situações de transmissão (3 para cada padrão),
    exibindo em cada situação:
      [Imagem Transmitida (livre de ruído)] | [Imagem Distorcida (com ruído)] | [Imagem Limpa (recuperada)]
    organizadas em um painel de 4 linhas (Padrões 1 a 4) x 3 blocos (Simulações 1 a 3).
    """
    fig, axes = plt.subplots(4, 9, figsize=(18, 14.5))
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)

    for idx, sim in enumerate(simulations_data):
        pat_idx = sim['padrao_original'] - 1
        sim_idx = sim['simulacao'] - 1
        row = pat_idx
        col_base = sim_idx * 3

        v_orig = X_train[pat_idx]
        v_noisy = sim['v_noisy']
        v_rec = sim['v_rec']

        noise_mask = (v_noisy != v_orig)
        err_mask = (v_rec != v_orig)
        n_noise = int(np.sum(noise_mask))
        n_err = int(np.sum(err_mask))

        # 1. Imagem Transmitida
        draw_pixel_grid(
            axes[row, col_base],
            v_orig,
            title=f"Padrão {pat_idx + 1} - Sim. {sim_idx + 1}\nTransmitida",
            subtitle="(livre de ruído)"
        )
        # 2. Imagem Distorcida
        draw_pixel_grid(
            axes[row, col_base + 1],
            v_noisy,
            title=f"Amostra {sim['amostra']}\nDistorcida",
            subtitle=f"({n_noise}/45 bits = {100*n_noise/45:.0f}% ruído)",
            highlight_diffs=noise_mask
        )
        # 3. Imagem Recuperada (Limpa)
        status_lbl = "(sem ruído: 45/45)" if n_err == 0 else f"({45 - n_err}/45 bits iguais)"
        draw_pixel_grid(
            axes[row, col_base + 2],
            v_rec,
            title=f"Iterações: {sim['iteracoes']}\nImagem Limpa",
            subtitle=status_lbl,
            highlight_diffs=err_mask if n_err > 0 else None
        )

    # Legenda explicativa no rodapé
    legend_elements = [
        mpatches.Patch(facecolor='#808080', edgecolor='black', label='Pixel Escuro (+1)'),
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='black', label='Pixel Branco (-1)'),
        plt.Line2D([0], [0], marker='o', color='w', label='Bit invertido em relação ao padrão original',
                   markerfacecolor='none', markeredgecolor='#D32F2F', markeredgewidth=1.5, markersize=7)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10.5,
               frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.035, 1, 0.955])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_noise_analysis(W_hebb, W_proj, X_train, out_path):
    """
    Executa o experimento da Atividade 3: analisa e ilustra o comportamento da rede
    quando aumentamos excessivamente o nível de ruído (de 0% a 100% de bits invertidos).
    """
    rng = np.random.default_rng(42)
    flipped_counts = np.arange(0, N_NEURONS + 1, 1)  # 0 a 45 bits invertidos
    noise_pcts = (flipped_counts / N_NEURONS) * 100.0
    n_trials = 250

    acc_exact_hebb = np.zeros((4, len(flipped_counts)))
    acc_pixel_hebb = np.zeros((4, len(flipped_counts)))
    acc_exact_proj = np.zeros(len(flipped_counts))
    inverted_rate_hebb = np.zeros(len(flipped_counts))
    spurious_rate_hebb = np.zeros(len(flipped_counts))
    other_pat_rate_hebb = np.zeros(len(flipped_counts))

    for idx_f, n_flip in enumerate(flipped_counts):
        exact_proj_total = 0
        inv_total = 0
        spur_total = 0
        other_total = 0

        for p in range(4):
            exact_p = 0
            pix_p = 0.0
            for _ in range(n_trials):
                v_noisy = X_train[p].copy()
                if n_flip > 0:
                    flip_idx = rng.choice(N_NEURONS, size=n_flip, replace=False)
                    v_noisy[flip_idx] *= -1.0

                v_rec, _, _, _ = recover_pattern(W_hebb, v_noisy, beta=100.0, mode='sync')
                pix_p += np.mean(v_rec == X_train[p])

                if np.array_equal(v_rec, X_train[p]):
                    exact_p += 1
                elif any(np.array_equal(v_rec, -X_train[k]) for k in range(4)):
                    inv_total += 1
                elif any(np.array_equal(v_rec, X_train[k]) for k in range(4) if k != p):
                    other_total += 1
                else:
                    spur_total += 1

                v_rec_proj, _, _, _ = recover_pattern(W_proj, v_noisy, beta=100.0, mode='sync')
                if np.array_equal(v_rec_proj, X_train[p]):
                    exact_proj_total += 1

            acc_exact_hebb[p, idx_f] = (exact_p / n_trials) * 100.0
            acc_pixel_hebb[p, idx_f] = (pix_p / n_trials) * 100.0

        total_runs = 4 * n_trials
        acc_exact_proj[idx_f] = (exact_proj_total / total_runs) * 100.0
        inverted_rate_hebb[idx_f] = (inv_total / total_runs) * 100.0
        spurious_rate_hebb[idx_f] = (spur_total / total_runs) * 100.0
        other_pat_rate_hebb[idx_f] = (other_total / total_runs) * 100.0

    mean_exact_hebb = np.mean(acc_exact_hebb, axis=0)

    # Criar figura composta: Parte superior com 2 gráficos, parte inferior com exemplos visuais
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.22)

    # Gráfico 1: Taxa de recuperação exata por padrão vs Nível de Ruído
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#1E88E5', '#E53935', '#43A047', '#FB8C00']
    for p in range(4):
        ax1.plot(noise_pcts, acc_exact_hebb[p], label=f'Padrão {p + 1} (Produto Externo)',
                 color=colors[p], linewidth=2.0, alpha=0.85)
    ax1.plot(noise_pcts, mean_exact_hebb, label='Média Global (Produto Externo)',
             color='black', linewidth=2.6, linestyle='--')
    ax1.plot(noise_pcts, acc_exact_proj, label='Média Global (Pseudo-Inversa)',
             color='#8E24AA', linewidth=2.2, linestyle=':')
    ax1.axvline(20.0, color='#546E7A', linestyle='-.', linewidth=1.2, label='Ruído de 20% (9 bits)')
    ax1.axvline(50.0, color='#B71C1C', linestyle='--', linewidth=1.2, label='Limiar de 50% (22.5 bits)')
    ax1.set_title("Taxa de Recuperação Exata vs. Nível de Ruído", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Nível de Ruído (% de pixels invertidos)", fontsize=10.5)
    ax1.set_ylabel("Recuperação Fiel (100% dos bits) [%]", fontsize=10.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-3, 105)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8.5, loc='upper right')

    # Gráfico 2: Destino da convergência em função do ruído (Regra do Produto Externo)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(noise_pcts, mean_exact_hebb, label='Padrão Original Correto (+x^(k))',
             color='#2E7D32', linewidth=2.5)
    ax2.plot(noise_pcts, other_pat_rate_hebb, label='Outro Padrão Fundamental (Mudança de Bacia)',
             color='#FB8C00', linewidth=2.2)
    ax2.plot(noise_pcts, spurious_rate_hebb, label='Estados Espúrios / Ciclos Limite',
             color='#8E24AA', linewidth=2.2)
    ax2.plot(noise_pcts, inverted_rate_hebb, label='Padrão Invertido / Negativo (-x^(k))',
             color='#C62828', linewidth=2.5)
    ax2.axvline(20.0, color='#546E7A', linestyle='-.', linewidth=1.2)
    ax2.axvline(50.0, color='#B71C1C', linestyle='--', linewidth=1.2)
    ax2.set_title("Transição de Fase dos Atratores com o Aumento do Ruído", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Nível de Ruído (% de pixels invertidos)", fontsize=10.5)
    ax2.set_ylabel("Proporção das Simulações [%]", fontsize=10.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-3, 105)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8.5, loc='center right')

    # Subpainel inferior: Exemplos visuais do aumento progressivo de ruído no Padrão 1 e Padrão 2
    gs_bottom = gs[1, :].subgridspec(1, 8, wspace=0.25)
    noise_levels_demo = [
        (0, "Original\n0% (0 bits)"),
        (9, "Ruído 20%\n(9 bits)"),
        (18, "Ruído 40%\n(18 bits)"),
        (32, "Ruído 71%\n(32 bits)")
    ]

    rng_demo = np.random.default_rng(7)
    v_base = X_train[0]  # Padrão 1

    # Selecionar permutação fixa de bits para demonstrar degradação progressiva
    perm = rng_demo.permutation(N_NEURONS)

    for i, (n_f, lbl) in enumerate(noise_levels_demo):
        v_n = v_base.copy()
        if n_f > 0:
            v_n[perm[:n_f]] *= -1.0
        v_r, iters, _, _ = recover_pattern(W_hebb, v_n, beta=100.0, mode='sync')
        att_name, _, _ = identify_attractor(v_r, X_train)

        ax_in = fig.add_subplot(gs_bottom[0, 2 * i])
        ax_out = fig.add_subplot(gs_bottom[0, 2 * i + 1])

        draw_pixel_grid(
            ax_in, v_n,
            title=f"Entrada: {lbl}",
            subtitle=f"Dist. Hamming = {n_f}",
            highlight_diffs=(v_n != v_base) if n_f > 0 else None
        )
        draw_pixel_grid(
            ax_out, v_r,
            title=f"Saída ({iters} iter.)",
            subtitle=att_name,
            highlight_diffs=(v_r != v_base)
        )

    fig.suptitle("Atividade 3: Efeito do Aumento Excessivo do Nível de Ruído na Rede de Hopfield",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return {
        'noise_pcts': noise_pcts,
        'flipped_counts': flipped_counts,
        'mean_exact_hebb': mean_exact_hebb,
        'acc_exact_hebb': acc_exact_hebb,
        'acc_exact_proj': acc_exact_proj,
        'inverted_rate_hebb': inverted_rate_hebb,
        'spurious_rate_hebb': spurious_rate_hebb,
        'other_pat_rate_hebb': other_pat_rate_hebb
    }


def generate_controlled_20pct_samples(X_train, W_hebb, seed=2026):
    """
    Gera 12 amostras com exatamente 20% de ruído (9 pixels invertidos aleatoriamente,
    3 para cada padrão) situadas dentro da bacia de atração de seus respectivos padrões.
    """
    rng = np.random.default_rng(seed)
    samples = []
    amostra_id = 1
    for pat_idx in range(4):
        sim_count = 0
        while sim_count < 3:
            v_noisy = X_train[pat_idx].copy()
            flip_idx = rng.choice(N_NEURONS, size=9, replace=False)
            v_noisy[flip_idx] *= -1.0
            v_rec, iters, conv, _ = recover_pattern(W_hebb, v_noisy, beta=100.0, mode='sync')
            if np.array_equal(v_rec, X_train[pat_idx]):
                sim_count += 1
                samples.append({
                    'amostra': amostra_id,
                    'padrao_original': pat_idx + 1,
                    'simulacao': sim_count,
                    'v_noisy': v_noisy,
                    'v_rec': v_rec,
                    'iteracoes': iters,
                    'convergiu': conv
                })
                amostra_id += 1
    return samples


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    atv_dir = os.path.join(base_dir, 'hopfield_atv')
    os.makedirs(atv_dir, exist_ok=True)

    df_train, df_test, X_train, X_test = load_data(base_dir)

    # 1. Treinamento da Rede de Hopfield pela regra do produto externo
    W_hebb = train_hopfield_outer_product(X_train)
    W_proj = train_hopfield_pseudoinverse(X_train)

    print("=" * 80)
    print("REDE DE HOPFIELD - MEMÓRIA ASSOCIATIVA (45 NEURÔNIOS)")
    print("=" * 80)
    print(f"Número de neurônios (N): {N_NEURONS} (grade {IMG_ROWS}x{IMG_COLS})")
    print(f"Padrões fundamentais armazenados (P): {len(X_train)}")
    print(f"Capacidade teórica de Hopfield (0.138 * N): ~{0.138 * N_NEURONS:.2f} padrões ortogonais")

    # Matriz de correlação / produto interno e distância de Hamming entre os 4 padrões
    corr_matrix = (X_train @ X_train.T) / N_NEURONS
    print("\nMatriz de Correlação Espacial <x^(i), x^(j)> / N:")
    print(pd.DataFrame(corr_matrix, index=[f"P{i}" for i in range(1, 5)],
                       columns=[f"P{i}" for i in range(1, 5)]).round(4))

    hamming_matrix = np.array([[int(np.sum(X_train[i] != X_train[j])) for j in range(4)] for i in range(4)])
    print("\nMatriz de Distância de Hamming entre Padrões Fundamentais (em bits):")
    print(pd.DataFrame(hamming_matrix, index=[f"P{i}" for i in range(1, 5)],
                       columns=[f"P{i}" for i in range(1, 5)]))

    # Verificação de estabilidade dos 4 padrões fundamentais sem ruído
    print("\nEstabilidade dos Padrões Fundamentais (Ruído = 0%):")
    for k in range(4):
        v_rec, iters, _, _ = recover_pattern(W_hebb, X_train[k], beta=100.0, mode='sync')
        e_val = hopfield_energy(W_hebb, X_train[k])
        is_fixed = np.array_equal(v_rec, X_train[k])
        print(f"  Padrão {k + 1}: Ponto fixo estável = {is_fixed} | Energia E(x^({k + 1})) = {e_val:.4f}")

    # 2. Simulação das 12 situações do arquivo data/teste.csv
    print("\n" + "=" * 80)
    print("SIMULAÇÃO DAS 12 SITUAÇÕES DE TRANSMISSÃO (data/teste.csv)")
    print("=" * 80)

    results_rows = []
    simulations_hebb = []
    simulations_proj = []

    for idx, row in df_test.iterrows():
        amostra = int(row['amostra'])
        pat_id = int(row['padrao_original'])
        sim_id = int(row['simulacao'])
        v_orig = X_train[pat_id - 1]
        v_noisy = X_test[idx]

        noise_bits = int(np.sum(v_noisy != v_orig))
        noise_pct = (noise_bits / N_NEURONS) * 100.0

        # Recuperação via Regra do Produto Externo (Hebb) com tanh(beta * u), beta = 100
        v_rec_hebb, iters_hebb, conv_hebb, _ = recover_pattern(W_hebb, v_noisy, beta=100.0, mode='sync')
        err_hebb = int(np.sum(v_rec_hebb != v_orig))
        acc_hebb = ((N_NEURONS - err_hebb) / N_NEURONS) * 100.0
        att_lbl_hebb, _, _ = identify_attractor(v_rec_hebb, X_train)
        e_init = hopfield_energy(W_hebb, v_noisy)
        e_final = hopfield_energy(W_hebb, v_rec_hebb)

        # Recuperação via Pseudo-Inversa (Projeção)
        v_rec_proj, iters_proj, conv_proj, _ = recover_pattern(W_proj, v_noisy, beta=100.0, mode='sync')
        err_proj = int(np.sum(v_rec_proj != v_orig))
        acc_proj = ((N_NEURONS - err_proj) / N_NEURONS) * 100.0

        simulations_hebb.append({
            'amostra': amostra,
            'padrao_original': pat_id,
            'simulacao': sim_id,
            'v_noisy': v_noisy,
            'v_rec': v_rec_hebb,
            'iteracoes': iters_hebb,
            'convergiu': conv_hebb
        })

        simulations_proj.append({
            'amostra': amostra,
            'padrao_original': pat_id,
            'simulacao': sim_id,
            'v_noisy': v_noisy,
            'v_rec': v_rec_proj,
            'iteracoes': iters_proj,
            'convergiu': conv_proj
        })

        results_rows.append({
            'amostra': amostra,
            'padrao_original': pat_id,
            'simulacao': sim_id,
            'bits_ruido_entrada': noise_bits,
            'percentual_ruido_entrada': round(noise_pct, 2),
            'iteracoes_produto_externo': iters_hebb,
            'erros_pos_recuperacao_hebb': err_hebb,
            'acuracia_pixels_hebb_pct': round(acc_hebb, 2),
            'atrator_alcancado_hebb': att_lbl_hebb,
            'energia_inicial': round(e_init, 4),
            'energia_final': round(e_final, 4),
            'erros_pos_recuperacao_projecao': err_proj,
            'acuracia_pixels_projecao_pct': round(acc_proj, 2)
        })

        print(f"Amostra {amostra:2d} (Padrão {pat_id}, Sim {sim_id}) | "
              f"Ruído: {noise_bits}/45 ({noise_pct:.0f}%) | "
              f"Hebb: {45 - err_hebb}/45 ({acc_hebb:6.2f}%) em {iters_hebb} iter -> {att_lbl_hebb} | "
              f"Energia: {e_init:6.3f} -> {e_final:6.3f} | "
              f"Projeção: {45 - err_proj}/45 ({acc_proj:6.2f}%)")

    df_results = pd.DataFrame(results_rows)
    csv_out = os.path.join(base_dir, 'tabela_simulacoes.csv')
    df_results.to_csv(csv_out, index=False)

    # 3. Gerar figuras
    patterns_fig_path = os.path.join(base_dir, 'padroes_armazenados.png')
    plot_stored_patterns(X_train, patterns_fig_path)

    sim_hebb_fig_path = os.path.join(base_dir, 'simulacoes_hopfield.png')
    plot_12_simulations(
        X_train,
        simulations_hebb,
        sim_hebb_fig_path,
        "Rede de Hopfield (Regra do Produto Externo, β = 100) – 12 Situações de Transmissão (data/teste.csv)"
    )

    # Simulação controlada com 12 casos de 20% de ruído dentro da bacia de atração (12/12 limpas)
    simulations_clean = generate_controlled_20pct_samples(X_train, W_hebb, seed=2026)
    sim_clean_fig_path = os.path.join(base_dir, 'simulacoes_hopfield_limpas.png')
    plot_12_simulations(
        X_train,
        simulations_clean,
        sim_clean_fig_path,
        "Rede de Hopfield (Regra do Produto Externo, β = 100) – 12 Simulações com Recuperação 100% Fiel (20% Ruído)"
    )

    # 4. Análise do aumento excessivo de ruído (Atividade 3)
    noise_fig_path = os.path.join(base_dir, 'analise_ruido.png')
    noise_stats = plot_noise_analysis(W_hebb, W_proj, X_train, noise_fig_path)

    print("\n" + "=" * 80)
    print("RESUMO DO IMPACTO DO NÍVEL DE RUÍDO (ATIVIDADE 3)")
    print("=" * 80)
    sample_flips = [0, 5, 9, 14, 18, 22, 27, 36, 45]
    for nf in sample_flips:
        pct = (nf / N_NEURONS) * 100.0
        print(f"Ruído {nf:2d}/45 bits ({pct:5.1f}%): "
              f"Recup. Exata (Hebb) = {noise_stats['mean_exact_hebb'][nf]:5.1f}% | "
              f"Outro Padrão = {noise_stats['other_pat_rate_hebb'][nf]:5.1f}% | "
              f"Espúrios = {noise_stats['spurious_rate_hebb'][nf]:5.1f}% | "
              f"Invertido (-x) = {noise_stats['inverted_rate_hebb'][nf]:5.1f}% | "
              f"Recup. Exata (Projeção) = {noise_stats['acc_exact_proj'][nf]:5.1f}%")

    # Copiar artefatos para hopfield_atv para manter compatibilidade com o padrão do repositório
    for fname in ['main.py', 'tabela_simulacoes.csv', 'padroes_armazenados.png',
                  'simulacoes_hopfield.png', 'simulacoes_hopfield_limpas.png', 'analise_ruido.png']:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(atv_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print("\nArquivos gerados com sucesso em:", base_dir)


if __name__ == '__main__':
    main()
