"""
CEFET-MG - Campus VIII - Varginha
Bacharelado em Sistemas de Informação
Disciplina: Laboratório de Inteligência Artificial
Professor: Lázaro Eduardo da Silva

Trabalho: Rede Neural LVQ-1 (Learning Vector Quantization 1)
Problema: Classificação de perfis de demanda de potência elétrica a partir das
          medições nas primeiras horas do dia (7h às 12h).
Entradas: x = [x1 (7h), x2 (8h), x3 (9h), x4 (10h), x5 (11h), x6 (12h)]^T
Classes: 4 perfis de demanda (Classes 1, 2, 3 e 4)
Parâmetro especificado: taxa de aprendizagem gamma = 0.05
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_COLS = ['x1_7h', 'x2_8h', 'x3_9h', 'x4_10h', 'x5_11h', 'x6_12h']
HOUR_LABELS = ['7h', '8h', '9h', '10h', '11h', '12h']


def load_data(base_dir):
    """Carrega os conjuntos de treinamento (16 dias) e teste (8 dias)."""
    train_path = os.path.join(base_dir, 'data', 'treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'teste.csv')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def train_lvq1(X, y, gamma0=0.05, n_prototypes_per_class=1, decay=False,
               max_epochs=500, tol=1e-6):
    """
    Treina uma rede LVQ-1 (Learning Vector Quantization 1).

    Inicialização canônica: seleciona os primeiros `n_prototypes_per_class` vetores
    de treinamento de cada classe como protótipos (codebook vectors) iniciais.

    Regra de atualização (LVQ-1):
      - Competição: j* = argmin_j ||x - w_j||_2
      - Se classe(j*) == T (acerto -> atração):
            w_{j*}(t+1) = w_{j*}(t) + gamma * (x - w_{j*}(t))
      - Se classe(j*) != T (erro -> repulsão):
            w_{j*}(t+1) = w_{j*}(t) - gamma * (x - w_{j*}(t))
    """
    classes = np.sort(np.unique(y))
    W_list = []
    W_classes_list = []
    init_sample_indices = []

    for c in classes:
        idxs = np.where(y == c)[0]
        chosen = idxs[:n_prototypes_per_class]
        for idx in chosen:
            W_list.append(X[idx].astype(float).copy())
            W_classes_list.append(int(c))
            init_sample_indices.append(int(idx) + 1)  # 1-indexado (número da amostra)

    W = np.array(W_list, dtype=float)
    W_init = W.copy()
    W_classes = np.array(W_classes_list, dtype=int)

    history_delta = []
    history_qe = []
    history_acc = []
    history_gamma = []

    for epoch in range(1, max_epochs + 1):
        W_prev = W.copy()
        gamma = gamma0 * (1.0 - (epoch - 1) / max_epochs) if decay else gamma0

        for x, target in zip(X, y):
            dists = np.linalg.norm(W - x, axis=1)
            winner = int(np.argmin(dists))

            if W_classes[winner] == target:
                W[winner] = W[winner] + gamma * (x - W[winner])
            else:
                W[winner] = W[winner] - gamma * (x - W[winner])

        max_delta = float(np.max(np.abs(W - W_prev)))

        # Métricas ao fim da época sobre o conjunto de treinamento
        dists_all = np.linalg.norm(X[:, np.newaxis, :] - W[np.newaxis, :, :], axis=2)
        winners = np.argmin(dists_all, axis=1)
        preds = W_classes[winners]
        qe = float(np.mean(np.min(dists_all, axis=1)))
        acc = float(np.mean(preds == y) * 100.0)

        history_delta.append(max_delta)
        history_qe.append(qe)
        history_acc.append(acc)
        history_gamma.append(gamma)

        if max_delta < tol:
            break

    return {
        'W_init': W_init,
        'W': W,
        'W_classes': W_classes,
        'init_samples': init_sample_indices,
        'epochs': epoch,
        'history_delta': history_delta,
        'history_qe': history_qe,
        'history_acc': history_acc,
        'history_gamma': history_gamma,
    }


def predict_lvq1(X, W, W_classes):
    """Classifica um conjunto de amostras usando os vetores protótipos treinados."""
    dists = np.linalg.norm(X[:, np.newaxis, :] - W[np.newaxis, :, :], axis=2)
    winners = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(X)), winners]
    preds = W_classes[winners]
    return preds, winners, min_dists, dists


def build_prototypes_table(model, df_train):
    """Constrói tabela resumo comparando protótipos LVQ-1 com os centroides das classes."""
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['classe'].values
    W = model['W']
    W_classes = model['W_classes']

    _, winners, _, _ = predict_lvq1(X_train, W, W_classes)

    rows = []
    for j in range(len(W)):
        cls = int(W_classes[j])
        centroid = X_train[y_train == cls].mean(axis=0)
        dev = float(np.linalg.norm(W[j] - centroid))
        hits = int(np.sum(winners == j))
        row = {
            'Neuronio': j + 1,
            'Classe': cls,
            'Amostra_Inicial': model['init_samples'][j],
            'w1_7h': round(W[j, 0], 4),
            'w2_8h': round(W[j, 1], 4),
            'w3_9h': round(W[j, 2], 4),
            'w4_10h': round(W[j, 3], 4),
            'w5_11h': round(W[j, 4], 4),
            'w6_12h': round(W[j, 5], 4),
            'Dist_Centroide': round(dev, 4),
            'Amostras_Treino_Ativadas': hits,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_test_table(model, df_test):
    """Classifica os 8 dias de teste e gera tabela detalhada com distâncias."""
    X_test = df_test[FEATURE_COLS].values
    preds, winners, min_dists, all_dists = predict_lvq1(
        X_test, model['W'], model['W_classes']
    )

    rows = []
    for i in range(len(df_test)):
        sorted_d = np.sort(all_dists[i])
        margin = float(sorted_d[1] - sorted_d[0])
        row = {
            'Dia': int(df_test.loc[i, 'dia']),
            '7_horas': round(float(X_test[i, 0]), 4),
            '8_horas': round(float(X_test[i, 1]), 4),
            '9_horas': round(float(X_test[i, 2]), 4),
            '10_horas': round(float(X_test[i, 3]), 4),
            '11_horas': round(float(X_test[i, 4]), 4),
            '12_horas': round(float(X_test[i, 5]), 4),
            'd_C1': round(float(all_dists[i, 0]), 4),
            'd_C2': round(float(all_dists[i, 1]), 4),
            'd_C3': round(float(all_dists[i, 2]), 4),
            'd_C4': round(float(all_dists[i, 3]), 4),
            'Dist_Minima': round(float(min_dists[i]), 4),
            'Margem_Separacao': round(margin, 4),
            'Classe': int(preds[i]),
        }
        rows.append(row)

    return pd.DataFrame(rows), all_dists


def plot_power_profiles(df_train, df_test_res, model, output_path):
    """
    Plota um grid 2x2 comparando, para cada uma das 4 classes de demanda:
      - Curvas das 4 amostras de treinamento;
      - Vetor protótipo (pesos sinápticos) aprendido pela LVQ-1;
      - Curvas dos dias de teste classificados naquela classe.
    """
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['classe'].values
    W = model['W']
    W_classes = model['W_classes']

    colors = {
        1: '#2563eb',  # Azul
        2: '#ea580c',  # Laranja
        3: '#16a34a',  # Verde
        4: '#9333ea',  # Roxo
    }
    test_colors = ['#dc2626', '#0f172a']
    test_markers = ['s', '^']

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=True, sharey=True)
    axes = axes.flatten()
    x_hours = np.arange(len(HOUR_LABELS))

    for idx, cls in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        c_hex = colors[cls]

        # Amostras de treinamento da classe
        tr_samples = X_train[y_train == cls]
        for s_i, curve in enumerate(tr_samples):
            label = 'Amostras de treinamento (4 dias)' if s_i == 0 else None
            ax.plot(
                x_hours, curve, color=c_hex, alpha=0.35, linewidth=1.6,
                linestyle='-', marker='o', markersize=4, label=label
            )

        # Vetor protótipo aprendido pela LVQ-1
        proto_idx = int(np.where(W_classes == cls)[0][0])
        ax.plot(
            x_hours, W[proto_idx], color=c_hex, linewidth=3.0,
            marker='D', markersize=7, markeredgecolor='white', markeredgewidth=0.8,
            zorder=5, label=f'Protótipo LVQ-1 ($w_{cls}$)'
        )

        # Amostras de teste classificadas nessa classe
        df_sub = df_test_res[df_test_res['Classe'] == cls].reset_index(drop=True)
        for t_i, row in df_sub.iterrows():
            t_curve = [
                row['7_horas'], row['8_horas'], row['9_horas'],
                row['10_horas'], row['11_horas'], row['12_horas']
            ]
            dia = int(row['Dia'])
            dist_min = row['Dist_Minima']
            ax.plot(
                x_hours, t_curve,
                color=test_colors[t_i % len(test_colors)],
                linestyle='--', linewidth=2.0,
                marker=test_markers[t_i % len(test_markers)],
                markersize=6.5, zorder=6,
                label=f'Teste – Dia {dia} ($d={dist_min:.4f}$)'
            )

        ax.set_title(f'Perfil de Demanda – Classe {cls}', fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks(x_hours)
        ax.set_xticklabels(HOUR_LABELS, fontsize=10)
        ax.set_ylim(-0.3, 7.6)
        ax.grid(True, linestyle=':', alpha=0.55)
        ax.legend(loc='upper left', fontsize=8.5, framealpha=0.92)
        if idx in (0, 2):
            ax.set_ylabel('Potência Elétrica (p.u. / escala medida)', fontsize=10.5)
        if idx in (2, 3):
            ax.set_xlabel('Horário da Medição', fontsize=10.5)

    fig.suptitle(
        'Rede Neural LVQ-1 ($\\gamma = 0.05$) – Perfis de Potência por Classe: Treinamento, Protótipos e Dias de Teste',
        fontsize=13.5, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_convergence_and_distances(model, df_test_res, all_dists, output_path):
    """
    Gera figura com dois painéis:
      - Esquerda: Convergência do treinamento (max |Delta w| em escala log e QE médio).
      - Direita: Matriz de distâncias euclidianas dos 8 dias de teste aos 4 protótipos LVQ-1.
    """
    epochs_axis = np.arange(1, model['epochs'] + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={'width_ratios': [1.05, 1.0]})

    # Painel esquerdo: convergência
    color_delta = '#1d4ed8'
    color_qe = '#dc2626'

    l1 = ax1.plot(
        epochs_axis, model['history_delta'], color=color_delta, linewidth=2.2,
        label=r'$\max |\Delta w_{ji}|$ por época'
    )
    ax1.set_yscale('log')
    ax1.set_xlabel('Época de Treinamento', fontsize=11)
    ax1.set_ylabel(r'Variação Máxima dos Pesos ($\max |\Delta w|$)', color=color_delta, fontsize=10.5)
    ax1.tick_params(axis='y', labelcolor=color_delta)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(
        epochs_axis, model['history_qe'], color=color_qe, linewidth=2.0, linestyle='--',
        label='Erro de Quantização Médio (QE)'
    )
    ax1_twin.set_ylabel('Erro de Quantização Médio (QE)', color=color_qe, fontsize=10.5)
    ax1_twin.tick_params(axis='y', labelcolor=color_qe)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9, framealpha=0.92)
    ax1.set_title(
        f'Convergência da Rede LVQ-1 ($\\gamma = 0.05$, {model["epochs"]} épocas)',
        fontsize=11.5, fontweight='bold', pad=10
    )

    # Painel direito: mapa de calor das distâncias dos dias de teste
    im = ax2.imshow(all_dists, cmap='YlGnBu_r', aspect='auto', vmin=0.0, vmax=np.max(all_dists))
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Distância Euclidiana $\\|\\mathbf{x} - \\mathbf{w}_c\\|_2$', fontsize=10)

    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(['Protótipo 1\n(Classe 1)', 'Protótipo 2\n(Classe 2)',
                         'Protótipo 3\n(Classe 3)', 'Protótipo 4\n(Classe 4)'], fontsize=9.5)
    ax2.set_yticks(np.arange(len(df_test_res)))
    ax2.set_yticklabels([f'Dia {d}' for d in df_test_res['Dia']], fontsize=10)
    ax2.set_title(
        'Distâncias Euclidianas – Dias de Teste vs. Protótipos LVQ-1',
        fontsize=11.5, fontweight='bold', pad=10
    )

    # Anotação numérica em cada célula com destaque no vencedor
    for i in range(all_dists.shape[0]):
        best_j = int(np.argmin(all_dists[i]))
        for j in range(all_dists.shape[1]):
            val = all_dists[i, j]
            if j == best_j:
                ax2.text(
                    j, i, f'{val:.4f}\n(C{j+1} ★)',
                    ha='center', va='center', fontsize=8.8, fontweight='bold', color='white'
                )
                rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                     fill=False, edgecolor='#facc15', linewidth=2.4)
                ax2.add_patch(rect)
            else:
                txt_color = 'white' if val < 3.5 else '#0f172a'
                ax2.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=8.5, color=txt_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print('=' * 100)
    print('CEFET-MG | LAB. INTELIGÊNCIA ARTIFICIAL - TRABALHO REDE NEURAL LVQ-1')
    print('Classificação de Perfis de Demanda de Potência Elétrica (7h às 12h) | gamma = 0.05')
    print('=' * 100)

    df_train, df_test = load_data(base_dir)
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train['classe'].values
    X_test = df_test[FEATURE_COLS].values

    # 1. Treinamento principal LVQ-1 com gamma = 0.05 (constante) e 1 protótipo por classe
    model = train_lvq1(
        X_train, y_train, gamma0=0.05, n_prototypes_per_class=1,
        decay=False, max_epochs=500, tol=1e-6
    )

    df_proto = build_prototypes_table(model, df_train)
    df_test_res, all_dists = build_test_table(model, df_test)

    print(f"\n[OK] Treinamento LVQ-1 concluído em {model['epochs']} épocas.")
    print(f"     - Variação máxima final dos pesos: {model['history_delta'][-1]:.2e}")
    print(f"     - Erro de Quantização Médio (QE) no treino: {model['history_qe'][-1]:.4f}")
    print(f"     - Acurácia no conjunto de treinamento (16 amostras): {model['history_acc'][-1]:.1f}%")

    print('\n' + '=' * 100)
    print('VETORES PROTÓTIPOS FINAIS (PESOS SINÁPTICOS DA CAMADA COMPETITIVA LVQ-1)')
    print('=' * 100)
    print(df_proto.to_string(index=False))

    print('\nDistâncias Euclidianas entre os Protótipos Finais:')
    W = model['W']
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            d = float(np.linalg.norm(W[i] - W[j]))
            print(f'  d(w_{i+1}, w_{j+1}) = {d:.4f}')

    print('\n' + '=' * 100)
    print('CLASSIFICAÇÃO DOS DADOS DE TESTE (8 DIAS)')
    print('=' * 100)
    print(df_test_res.to_string(index=False))

    # 2. Verificação de robustez (comparação com decaimento linear de gamma e 2 protótipos/classe)
    print('\n' + '=' * 100)
    print('VERIFICAÇÃO DE ROBUSTEZ (VARIAÇÕES DE CONFIGURAÇÃO COM gamma_0 = 0.05)')
    print('=' * 100)
    variants = [
        ('LVQ-1 padrão (gamma=0.05 fixo, 4 protótipos)', dict(gamma0=0.05, n_prototypes_per_class=1, decay=False, max_epochs=500)),
        ('LVQ-1 com decaimento linear (gamma_0=0.05 -> 0, 4 protótipos)', dict(gamma0=0.05, n_prototypes_per_class=1, decay=True, max_epochs=200)),
        ('LVQ-1 estendida (gamma=0.05 fixo, 8 protótipos / 2 por classe)', dict(gamma0=0.05, n_prototypes_per_class=2, decay=False, max_epochs=500)),
    ]
    for name, kwargs in variants:
        m_var = train_lvq1(X_train, y_train, tol=1e-6, **kwargs)
        preds_var, _, _, _ = predict_lvq1(X_test, m_var['W'], m_var['W_classes'])
        print(f"  - {name}:")
        print(f"      Épocas = {m_var['epochs']:3d} | QE treino = {m_var['history_qe'][-1]:.4f} | "
              f"Acurácia treino = {m_var['history_acc'][-1]:.1f}% | Predições teste = {preds_var.tolist()}")

    # 3. Salvar CSVs
    csv_proto_path = os.path.join(base_dir, 'tabela_prototipos_lvq1.csv')
    csv_test_path = os.path.join(base_dir, 'tabela_teste_lvq1.csv')
    df_proto.to_csv(csv_proto_path, index=False)
    df_test_res.to_csv(csv_test_path, index=False)
    print(f'\n[OK] Tabela de protótipos salva em: {csv_proto_path}')
    print(f'[OK] Tabela de classificação de teste salva em: {csv_test_path}')

    # 4. Gerar gráficos
    profiles_img_path = os.path.join(base_dir, 'perfis_potencia_lvq1.png')
    conv_img_path = os.path.join(base_dir, 'convergencia_e_distancias_lvq1.png')
    plot_power_profiles(df_train, df_test_res, model, profiles_img_path)
    plot_convergence_and_distances(model, df_test_res, all_dists, conv_img_path)
    print(f'[OK] Gráfico de perfis de potência salvo em: {profiles_img_path}')
    print(f'[OK] Gráfico de convergência e distâncias salvo em: {conv_img_path}')


if __name__ == '__main__':
    main()

