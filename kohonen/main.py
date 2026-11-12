"""
CEFET-MG - Campus VIII - Varginha
Bacharelado em Sistemas de Informação
Disciplina: Laboratório de Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Trabalho: Rede de Kohonen (Mapa Auto-Organizável - SOM)
Problema: Agrupamento de amostras de imperfeições no composto de borracha na fabricação de pneus
Entradas: x1, x2, x3
Topologia:
  - N1 = 16 neurônios dispostos em um grid topológico bidimensional 4x4
  - Taxa de aprendizado: eta = 0.001
  - Raio de vizinhança: R = 1
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_data(base_dir):
    """Carrega os conjuntos de treinamento (120 amostras) e de teste (12 amostras)."""
    train_path = os.path.join(base_dir, 'data', 'treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'teste.csv')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def build_grid_coordinates(rows=4, cols=4):
    """Retorna as coordenadas (linha, coluna) de cada neurônio 1..16 (0-indexado 0..15)."""
    return np.array([(r, c) for r in range(rows) for c in range(cols)])


def get_neighbors_list(coords, radius=1):
    """
    Retorna o dicionário de vizinhos (1-indexado) para cada neurônio no grid 4x4
    considerando distância topológica no grafo do diagrama esquemático (raio = 1).
    """
    n_neurons = len(coords)
    neighbors = {}
    for j in range(n_neurons):
        nbs = []
        for k in range(n_neurons):
            if j != k:
                dist_grid = abs(coords[j, 0] - coords[k, 0]) + abs(coords[j, 1] - coords[k, 1])
                if dist_grid <= radius:
                    nbs.append(k + 1)
        neighbors[j + 1] = nbs
    return neighbors


def train_kohonen(X, n_neurons=16, eta=0.001, radius=1, nb_factor=0.5,
                  max_epochs=2000, tol=1e-7, seed=42):
    """
    Treina a Rede de Kohonen (SOM) 4x4 utilizando a regra da Norma Euclidiana.
    - Neurônio vencedor j*: w_{j*}(t+1) = w_{j*}(t) + eta * (x - w_{j*}(t))
    - Neurônios vizinhos k em N(j*): w_k(t+1) = w_k(t) + (eta * nb_factor) * (x - w_k(t))
    """
    np.random.seed(seed)
    n_features = X.shape[1]
    W = np.random.uniform(0.0, 1.0, size=(n_neurons, n_features))
    W_init = W.copy()

    coords = build_grid_coordinates(4, 4)

    # Matriz de taxas de atualização H[bmu, k, 1] para vetorização rápida
    H = np.zeros((n_neurons, n_neurons, 1))
    for j in range(n_neurons):
        H[j, j, 0] = eta
        for k in range(n_neurons):
            if k != j:
                dist_grid = abs(coords[j, 0] - coords[k, 0]) + abs(coords[j, 1] - coords[k, 1])
                if dist_grid <= radius:
                    H[j, k, 0] = eta * nb_factor

    history_delta = []
    history_qe = []

    for epoch in range(1, max_epochs + 1):
        W_prev = W.copy()
        for x in X:
            dists_sq = np.sum((W - x) ** 2, axis=1)
            bmu = int(np.argmin(dists_sq))
            W += H[bmu] * (x - W)

        max_delta = float(np.max(np.abs(W - W_prev)))
        # Erro de quantização médio (média da distância euclidiana ao neurônio vencedor)
        dists_all = np.linalg.norm(X[:, np.newaxis, :] - W[np.newaxis, :, :], axis=2)
        qe = float(np.mean(np.min(dists_all, axis=1)))

        history_delta.append(max_delta)
        history_qe.append(qe)

        if max_delta < tol:
            break

    return W_init, W, epoch, history_delta, history_qe


def map_neurons_to_classes(W, df_train):
    """
    Mapeia os 16 neurônios do grid às classes A (1-20), B (21-60) e C (61-120):
    1. Pelo conjunto de neurônios vencedores ativos (BMUs das amostras de treinamento).
    2. Pela classe majoritária / centroide mais próximo (para rotular também neurônios de transição).
    """
    X_train = df_train[['x1', 'x2', 'x3']].values
    y_train = df_train['classe'].values

    # Neurônio vencedor (1 a 16) para cada amostra de treinamento
    dists = np.linalg.norm(X_train[:, np.newaxis, :] - W[np.newaxis, :, :], axis=2)
    bmus = np.argmin(dists, axis=1) + 1
    min_dists = np.min(dists, axis=1)

    df_mapped = df_train.copy()
    df_mapped['neuronio_vencedor'] = bmus
    df_mapped['distancia_euclidiana'] = np.round(min_dists, 4)

    # Centroides das classes A, B e C no conjunto de treinamento
    centroids = {
        cls: df_train[df_train['classe'] == cls][['x1', 'x2', 'x3']].mean().values
        for cls in ['A', 'B', 'C']
    }

    coords = build_grid_coordinates(4, 4)
    neighbors_dict = get_neighbors_list(coords, radius=1)

    neuron_rows = []
    for j in range(1, 17):
        w_j = W[j - 1]
        sub = df_mapped[df_mapped['neuronio_vencedor'] == j]
        count_a = int(np.sum(sub['classe'] == 'A'))
        count_b = int(np.sum(sub['classe'] == 'B'))
        count_c = int(np.sum(sub['classe'] == 'C'))
        total_hits = len(sub)

        # Distância aos centroides de cada classe
        dist_centroids = {cls: float(np.linalg.norm(w_j - centroids[cls])) for cls in ['A', 'B', 'C']}
        closest_cls = min(['A', 'B', 'C'], key=lambda c: dist_centroids[c])

        if total_hits > 0:
            counts = {'A': count_a, 'B': count_b, 'C': count_c}
            assigned_cls = max(counts, key=counts.get)
            status = 'Ativo (Vencedor BMU)'
        else:
            assigned_cls = closest_cls
            status = 'Transição (Fronteira)'

        r, c = coords[j - 1]
        neuron_rows.append({
            'Neuronio': j,
            'Linha_Grid': r + 1,
            'Coluna_Grid': c + 1,
            'Vizinhos_Raio1': str(neighbors_dict[j]),
            'w1': round(float(w_j[0]), 4),
            'w2': round(float(w_j[1]), 4),
            'w3': round(float(w_j[2]), 4),
            'Amostras_A': count_a,
            'Amostras_B': count_b,
            'Amostras_C': count_c,
            'Total_Amostras': total_hits,
            'Status': status,
            'Classe': assigned_cls,
            'Dist_Centroide_Classe': round(dist_centroids[assigned_cls], 4),
        })

    df_neurons = pd.DataFrame(neuron_rows)
    return df_mapped, df_neurons, centroids


def classify_test_samples(W, df_neurons, df_test):
    """Classifica as 12 amostras da tabela do Item 2 utilizando a Rede de Kohonen treinada."""
    X_test = df_test[['x1', 'x2', 'x3']].values
    neuron_to_class = dict(zip(df_neurons['Neuronio'], df_neurons['Classe']))

    rows = []
    for idx, row in df_test.iterrows():
        x = np.array([row['x1'], row['x2'], row['x3']])
        dists = np.linalg.norm(W - x, axis=1)
        bmu = int(np.argmin(dists)) + 1
        dist_bmu = float(dists[bmu - 1])
        cls = neuron_to_class[bmu]

        rows.append({
            'Amostra': int(row['amostra']),
            'x1': row['x1'],
            'x2': row['x2'],
            'x3': row['x3'],
            'Neuronio_Vencedor': bmu,
            'Distancia_BMU': round(dist_bmu, 4),
            'Classe': cls,
        })

    return pd.DataFrame(rows)


def plot_topological_grid_and_convergence(df_neurons, history_qe, history_delta, output_path):
    """
    Plota:
    (Esquerda) O Mapa Topológico 4x4 da Rede de Kohonen colorido por Classe (A, B, C),
               destacando neurônios vencedores ativos e de transição, seus pesos e número de ativações.
    (Direita)  A curva de convergência do Erro de Quantização Médio e variação máxima dos pesos.
    """
    fig, (ax_grid, ax_conv) = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={'width_ratios': [1.25, 1.0]})

    # Cores por classe
    class_colors = {
        'A': '#3b82f6',  # Azul
        'B': '#f59e0b',  # Laranja/Âmbar
        'C': '#10b981',  # Verde
    }
    class_colors_light = {
        'A': '#dbeafe',
        'B': '#fef3c7',
        'C': '#d1fae5',
    }

    coords = build_grid_coordinates(4, 4)

    # Desenhar arestas do grid 4x4 (raio de vizinhança = 1)
    for r in range(4):
        for c in range(4):
            y_pos = 3 - r
            x_pos = c
            if c < 3:
                ax_grid.plot([x_pos, x_pos + 1], [y_pos, y_pos], color='#475569', linewidth=2.2, zorder=1)
            if r < 3:
                ax_grid.plot([x_pos, x_pos], [y_pos, y_pos - 1], color='#475569', linewidth=2.2, zorder=1)

    # Desenhar cada neurônio no grid 4x4
    for _, row in df_neurons.iterrows():
        n_id = int(row['Neuronio'])
        r = int(row['Linha_Grid']) - 1
        c = int(row['Coluna_Grid']) - 1
        x_pos = c
        y_pos = 3 - r

        cls = row['Classe']
        hits = int(row['Total_Amostras'])
        is_active = hits > 0

        face_col = class_colors[cls] if is_active else class_colors_light[cls]
        edge_col = '#0f172a' if is_active else class_colors[cls]
        text_col = 'white' if is_active else '#0f172a'
        ls = 'solid' if is_active else 'dashed'
        lw = 2.5 if is_active else 2.0

        circle = plt.Circle((x_pos, y_pos), 0.36, facecolor=face_col, edgecolor=edge_col,
                            linewidth=lw, linestyle=ls, zorder=3)
        ax_grid.add_patch(circle)

        # Texto dentro do neurônio
        label_top = f"N{n_id} ({cls})"
        label_hits = f"{hits} am." if is_active else "fronteira"
        label_w = f"[{row['w1']:.2f}, {row['w2']:.2f}, {row['w3']:.2f}]"

        ax_grid.text(x_pos, y_pos + 0.11, label_top, ha='center', va='center',
                     fontsize=10.5, fontweight='bold', color=text_col, zorder=4)
        ax_grid.text(x_pos, y_pos - 0.04, label_hits, ha='center', va='center',
                     fontsize=8.5, fontweight='semibold', color=text_col, zorder=4)
        ax_grid.text(x_pos, y_pos - 0.18, label_w, ha='center', va='center',
                     fontsize=6.8, color=text_col, zorder=4)

    ax_grid.set_xlim(-0.6, 3.6)
    ax_grid.set_ylim(-0.6, 3.6)
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    ax_grid.set_title('Mapa Topológico 4x4 da Rede de Kohonen (Classes A, B e C)',
                      fontsize=13, fontweight='bold', pad=12)

    legend_patches = [
        mpatches.Patch(facecolor=class_colors['A'], edgecolor='#0f172a', label='Classe A (Ativo: N4, N8 | Fronteira: N7)'),
        mpatches.Patch(facecolor=class_colors['B'], edgecolor='#0f172a', label='Classe B (Ativo: N11, N15, N16 | Fronteira: N12, N14)'),
        mpatches.Patch(facecolor=class_colors['C'], edgecolor='#0f172a', label='Classe C (Ativo: N1, N2, N5, N6, N9, N10, N13 | Fronteira: N3)'),
    ]
    ax_grid.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.12),
                   fontsize=8.8, frameon=True, facecolor='#f8fafc')

    # Curva de convergência
    epochs_range = np.arange(1, len(history_qe) + 1)
    color_qe = '#2563eb'
    color_dl = '#dc2626'

    ax_conv.plot(epochs_range, history_qe, color=color_qe, linewidth=2.2, label='Erro de Quantização Médio (QE)')
    ax_conv.set_xlabel('Épocas de Treinamento', fontsize=11)
    ax_conv.set_ylabel('Erro de Quantização Médio (Distância Euclidiana)', color=color_qe, fontsize=10.5)
    ax_conv.tick_params(axis='y', labelcolor=color_qe)
    ax_conv.grid(True, linestyle='--', alpha=0.5)
    ax_conv.set_title('Convergência do Treinamento da Rede de Kohonen', fontsize=13, fontweight='bold', pad=12)

    ax_conv2 = ax_conv.twinx()
    ax_conv2.plot(epochs_range, history_delta, color=color_dl, linewidth=1.8, linestyle='--',
                  label=r'$\max |\Delta \mathbf{w}_j|$ por época')
    ax_conv2.set_yscale('log')
    ax_conv2.set_ylabel(r'Variação Máxima dos Pesos ($\max |\Delta \mathbf{w}_j|$)', color=color_dl, fontsize=10.5)
    ax_conv2.tick_params(axis='y', labelcolor=color_dl)

    lines1, labels1 = ax_conv.get_legend_handles_labels()
    lines2, labels2 = ax_conv2.get_legend_handles_labels()
    ax_conv.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_feature_space(df_train, df_test_res, W, df_neurons, output_path):
    """
    Plota no espaço tridimensional (x1, x2, x3):
    - As 120 amostras de treinamento (Classes A, B, C)
    - As 12 amostras de teste classificadas
    - Os 16 vetores de pesos sinápticos W_j conectados pela malha topológica 4x4
    """
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')

    class_colors = {'A': '#2563eb', 'B': '#d97706', 'C': '#059669'}

    for cls, label_desc in [('A', 'Treino Classe A (1-20)'),
                            ('B', 'Treino Classe B (21-60)'),
                            ('C', 'Treino Classe C (61-120)')]:
        sub = df_train[df_train['classe'] == cls]
        ax.scatter(sub['x1'], sub['x2'], sub['x3'], c=class_colors[cls],
                   s=36, alpha=0.55, label=label_desc)

    # Amostras de teste
    for cls in ['A', 'B', 'C']:
        sub_t = df_test_res[df_test_res['Classe'] == cls]
        ax.scatter(sub_t['x1'], sub_t['x2'], sub_t['x3'], c=class_colors[cls],
                   marker='^', s=110, edgecolors='black', linewidths=1.2,
                   label=f'Teste classificado como {cls}')

    # Conexões da grade 4x4 no espaço 3D
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            if c < 3:
                idx_r = r * 4 + (c + 1)
                ax.plot([W[idx, 0], W[idx_r, 0]],
                        [W[idx, 1], W[idx_r, 1]],
                        [W[idx, 2], W[idx_r, 2]],
                        color='#334155', linestyle='-', linewidth=1.2, alpha=0.7)
            if r < 3:
                idx_d = (r + 1) * 4 + c
                ax.plot([W[idx, 0], W[idx_d, 0]],
                        [W[idx, 1], W[idx_d, 1]],
                        [W[idx, 2], W[idx_d, 2]],
                        color='#334155', linestyle='-', linewidth=1.2, alpha=0.7)

    # Neurônios W_j
    ax.scatter(W[:, 0], W[:, 1], W[:, 2], c='#dc2626', marker='D', s=95,
               edgecolors='black', linewidths=1.2, zorder=10, label='Neurônios Kohonen (w_j)')

    for j in range(16):
        ax.text(W[j, 0] + 0.012, W[j, 1] + 0.012, W[j, 2] + 0.012,
                f"N{j+1}", fontsize=8.5, fontweight='bold', color='#7f1d1d')

    ax.set_xlabel('$x_1$', fontsize=11, fontweight='bold')
    ax.set_ylabel('$x_2$', fontsize=11, fontweight='bold')
    ax.set_zlabel('$x_3$', fontsize=11, fontweight='bold')
    ax.set_title('Espaço Tridimensional de Entradas $(x_1, x_2, x_3)$ e Grade Topológica 4x4 de Kohonen',
                 fontsize=12.5, fontweight='bold', pad=14)
    ax.view_init(elev=24, azim=42)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 95)
    print("CEFET-MG | LAB. INTELIGÊNCIA ARTIFICIAL - TRABALHO REDE DE KOHONEN (SOM 4x4)")
    print("Agrupamento de Amostras Imperfeitas no Processo de Fabricação de Pneus")
    print("=" * 95)

    df_train, df_test = load_data(base_dir)
    X_train = df_train[['x1', 'x2', 'x3']].values

    # Treinamento principal (Algoritmo canônico de Kohonen: eta = 0.001, raio = 1, vizinhos com eta/2)
    W_init, W, epochs, hist_delta, hist_qe = train_kohonen(
        X_train, n_neurons=16, eta=0.001, radius=1, nb_factor=0.5, max_epochs=2000, tol=1e-7, seed=42
    )

    df_mapped, df_neurons, centroids = map_neurons_to_classes(W, df_train)
    df_test_res = classify_test_samples(W, df_neurons, df_test)

    print(f"\n[OK] Treinamento concluído em {epochs} épocas (Erro de Quantização Médio final = {hist_qe[-1]:.4f}).")

    print("\n" + "=" * 95)
    print("QUESTÃO 1: MAPEAMENTO DOS 16 NEURÔNIOS DO GRID 4x4 PARA AS CLASSES A, B e C")
    print("=" * 95)
    print(df_neurons[['Neuronio', 'Linha_Grid', 'Coluna_Grid', 'w1', 'w2', 'w3',
                      'Amostras_A', 'Amostras_B', 'Amostras_C', 'Total_Amostras',
                      'Status', 'Classe']].to_string(index=False))

    print("\nResumo dos Conjuntos de Neurônios por Classe:")
    for cls in ['A', 'B', 'C']:
        ativos = df_neurons[(df_neurons['Classe'] == cls) & (df_neurons['Total_Amostras'] > 0)]['Neuronio'].tolist()
        todos = df_neurons[df_neurons['Classe'] == cls]['Neuronio'].tolist()
        fronteira = [n for n in todos if n not in ativos]
        print(f"  - Classe {cls}:")
        print(f"      * Neurônios vencedores ativos (BMUs diretos no treino): {ativos}")
        print(f"      * Região completa no grid 4x4 (incluindo transição {fronteira}): {todos}")

    print("\nRepresentação Visual do Grid 4x4 (Neurônio -> Classe [Amostras Ativadas]):")
    print("-" * 65)
    for r in range(1, 5):
        row_cells = []
        for c in range(1, 5):
            n_row = df_neurons[(df_neurons['Linha_Grid'] == r) & (df_neurons['Coluna_Grid'] == c)].iloc[0]
            n_id = int(n_row['Neuronio'])
            cls = n_row['Classe']
            hits = int(n_row['Total_Amostras'])
            tag = f"{hits:2d} am" if hits > 0 else "front"
            row_cells.append(f"N{n_id:02d}:{cls} ({tag})")
        print("  |  ".join(row_cells))
    print("-" * 65)

    print("\n" + "=" * 95)
    print("QUESTÃO 2: CLASSIFICAÇÃO DAS 12 AMOSTRAS DE TESTE")
    print("=" * 95)
    print(df_test_res.to_string(index=False))

    # Salvar CSVs
    csv_q1_path = os.path.join(base_dir, 'tabela_item1_neuronios.csv')
    csv_q2_path = os.path.join(base_dir, 'tabela_item2_teste.csv')
    df_neurons.to_csv(csv_q1_path, index=False)
    df_test_res.to_csv(csv_q2_path, index=False)
    print(f"\n[OK] Tabela da Questão 1 salva em: {csv_q1_path}")
    print(f"[OK] Tabela da Questão 2 salva em: {csv_q2_path}")

    # Gerar gráficos
    grid_img_path = os.path.join(base_dir, 'mapa_topologico_kohonen.png')
    space_img_path = os.path.join(base_dir, 'espaco_3d_kohonen.png')
    plot_topological_grid_and_convergence(df_neurons, hist_qe, hist_delta, grid_img_path)
    plot_3d_feature_space(df_train, df_test_res, W, df_neurons, space_img_path)
    print(f"[OK] Gráfico do Mapa Topológico e Convergência salvo em: {grid_img_path}")
    print(f"[OK] Gráfico do Espaço 3D salvo em: {space_img_path}")


if __name__ == '__main__':
    main()

