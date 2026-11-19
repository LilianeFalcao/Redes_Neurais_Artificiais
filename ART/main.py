"""
CEFET-MG - Campus VIII - Varginha
Bacharelado em Sistemas de Informação
Disciplina: Laboratório de Inteligência Artificial
Professor: Lázaro Eduardo da Silva
Data: 19/11/2026

Trabalho: Rede Neural ART-1 (Adaptive Resonance Theory 1)
Problema: Classificação e agrupamento de 10 situações do comportamento de um
          processo industrial a partir de 16 variáveis binárias de status
          (x1 a x16), visando provável diagnóstico para eventual manutenção.
Parâmetros de vigilância avaliados: rho in {0.5, 0.8, 0.9, 0.99}
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


FEATURE_COLS = [f'x{i}' for i in range(1, 17)]
RHOS_TARGET = [0.5, 0.8, 0.9, 0.99]


def load_data(base_dir):
    """Carrega os dados das 10 situações industriais (16 variáveis binárias)."""
    train_path = os.path.join(base_dir, 'data', 'treinamento.csv')
    test_path = os.path.join(base_dir, 'data', 'teste.csv')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


class ART1:
    """
    Implementação canônica da Rede Neural ART-1 (Carpenter & Grossberg, 1987;
    Fausett, 1994; Silva, Spatti & Flauzino, 2010).

    Arquitetura:
      - Camada de comparação/entrada F1: n = 16 neurônios (entradas binárias).
      - Camada de reconhecimento/saída F2: m neurônios competitivos (classes).
      - Pesos top-down (F2 -> F1): t_{ji}(0) = 1 (protótipos/gabaritos binários).
      - Pesos bottom-up (F1 -> F2): b_{ij}(0) = 1 / (1 + n), satisfazendo
        0 < b_{ij}(0) < L / (L - 1 + n) para o parâmetro padrão L = 2.

    Regras de operação (aprendizagem rápida):
      1. Competição (camada F2): y_j = sum_i b_{ij} * s_i, escolhendo o neurônio
         não inibido J com maior ativação y_J (desempate pelo menor índice j).
      2. Teste de vigilância: x = s AND t_J; r = ||x||_1 / ||s||_1.
         - Se r < rho: ocorre reset (neurônio J é inibido para a amostra atual)
           e busca-se o próximo candidato em F2.
         - Se r >= rho: ocorre ressonância e os pesos do neurônio J são atualizados:
               t_{Ji}^{novo} = s_i * t_{Ji}^{antigo}
               b_{iJ}^{novo} = (L * t_{Ji}^{novo}) / (L - 1 + ||t_J^{novo}||_1)
      3. Critério de parada: treinamento em épocas sucessivas até convergência
         (nenhuma alteração nas atribuições de classe nem nos protótipos T).
    """

    def __init__(self, n_features, max_clusters=10, rho=0.8, L=2.0, max_epochs=100):
        self.n_features = int(n_features)
        self.max_clusters = int(max_clusters)
        self.rho = float(rho)
        self.L = float(L)
        self.max_epochs = int(max_epochs)
        self.b0 = 1.0 / (1.0 + self.n_features)

        # Pesos iniciais
        self.T = np.ones((self.max_clusters, self.n_features), dtype=int)
        self.B = np.full((self.n_features, self.max_clusters), self.b0, dtype=float)
        self.committed = np.zeros(self.max_clusters, dtype=bool)

        # Histórico e métricas do treinamento
        self.epochs_ = 0
        self.total_resets_ = 0
        self.epoch_history_ = []
        self.assignments_ = None
        self.match_ratios_ = None

    def _select_winner(self, s, norm_s, inhibited):
        """
        Seleciona o neurônio vencedor J em F2 dentre os não inibidos.
        Cálculo explícito por coluna para evitar que diferenças numéricas de
        ponto flutuante (SIMD/BLAS) quebrem empates exatos entre neurônios livres.
        """
        best_j = -1
        best_y = -np.inf
        for j in range(self.max_clusters):
            if inhibited[j]:
                continue
            if not self.committed[j]:
                yj = self.b0 * norm_s
            else:
                yj = float(np.sum(s * self.B[:, j]))
            if yj > best_y + 1e-12:
                best_y = yj
                best_j = j
        return best_j, best_y

    def fit(self, X):
        """Treina a rede ART-1 até estabilização completa dos agrupamentos."""
        X = np.asarray(X, dtype=int)
        n_samples = X.shape[0]

        # Reinicialização limpa
        self.T = np.ones((self.max_clusters, self.n_features), dtype=int)
        self.B = np.full((self.n_features, self.max_clusters), self.b0, dtype=float)
        self.committed = np.zeros(self.max_clusters, dtype=bool)
        self.total_resets_ = 0
        self.epoch_history_ = []

        prev_assign = None
        for epoch in range(1, self.max_epochs + 1):
            self.epochs_ = epoch
            assign = np.full(n_samples, -1, dtype=int)
            ratios = np.zeros(n_samples, dtype=float)
            T_prev = self.T.copy()
            epoch_resets = 0

            for idx in range(n_samples):
                s = X[idx]
                norm_s = int(np.sum(s))
                inhibited = np.zeros(self.max_clusters, dtype=bool)

                while True:
                    j, _ = self._select_winner(s, norm_s, inhibited)
                    if j == -1:
                        raise RuntimeError(
                            'Todos os neurônios de F2 foram inibidos. '
                            'Aumente max_clusters.'
                        )
                    x_and = s * self.T[j]
                    norm_x = int(np.sum(x_and))
                    r = norm_x / norm_s if norm_s > 0 else 1.0

                    if r >= self.rho - 1e-12:
                        # Ressonância: atualização dos pesos top-down e bottom-up
                        self.T[j] = x_and
                        self.B[:, j] = (self.L * x_and) / (self.L - 1.0 + norm_x)
                        self.committed[j] = True
                        assign[idx] = j
                        ratios[idx] = r
                        break
                    else:
                        # Reset: inibe o neurônio j para a amostra atual
                        inhibited[j] = True
                        epoch_resets += 1

            self.total_resets_ += epoch_resets
            self.epoch_history_.append({
                'epoch': epoch,
                'assignments': assign.copy(),
                'resets': epoch_resets,
                'active_neurons': sorted(set(int(c) for c in assign)),
                'committed_count': int(np.sum(self.committed)),
            })

            if (prev_assign is not None
                    and np.array_equal(assign, prev_assign)
                    and np.array_equal(self.T, T_prev)):
                self.assignments_ = assign
                self.match_ratios_ = ratios
                break
            prev_assign = assign.copy()
        else:
            self.assignments_ = prev_assign
            self.match_ratios_ = ratios

        return self

    def predict(self, X):
        """
        Classifica um conjunto de entradas usando os pesos já treinados
        (sem alterar os pesos sinápticos).
        """
        X = np.asarray(X, dtype=int)
        n_samples = X.shape[0]
        preds = np.full(n_samples, -1, dtype=int)
        ratios = np.zeros(n_samples, dtype=float)
        activations = np.zeros(n_samples, dtype=float)

        for idx in range(n_samples):
            s = X[idx]
            norm_s = int(np.sum(s))
            inhibited = np.zeros(self.max_clusters, dtype=bool)
            while True:
                j, yj = self._select_winner(s, norm_s, inhibited)
                if j == -1:
                    break
                x_and = s * self.T[j]
                r = int(np.sum(x_and)) / norm_s if norm_s > 0 else 1.0
                if r >= self.rho - 1e-12:
                    preds[idx] = j
                    ratios[idx] = r
                    activations[idx] = yj
                    break
                inhibited[j] = True

        return preds, ratios, activations

    def get_active_clusters(self, situations):
        """
        Retorna os agrupamentos ativos ordenados pela menor situação do grupo,
        com rótulo sequencial (Classe 1, Classe 2, ...) e o índice real do
        neurônio em F2 (1-indexado).
        """
        raw_groups = {}
        for sit_id, neuron_idx in zip(situations, self.assignments_):
            raw_groups.setdefault(int(neuron_idx), []).append(int(sit_id))

        # Ordena os grupos ativos pela primeira situação pertencente ao grupo
        ordered = sorted(raw_groups.items(), key=lambda kv: min(kv[1]))
        result = []
        for class_num, (neuron_idx, members) in enumerate(ordered, start=1):
            proto = self.T[neuron_idx].copy()
            result.append({
                'classe': class_num,
                'neuronio_f2': neuron_idx + 1,  # 1-indexado
                'situacoes': sorted(members),
                'prototipo': proto,
                'bits_ativos_prototipo': int(np.sum(proto)),
            })
        return result


def compute_pairwise_match_matrix(X):
    """
    Calcula a matriz de similaridade assimétrica da ART-1:
        R[i, j] = ||s_i AND s_j||_1 / ||s_i||_1
    que representa a taxa de correspondência de s_i frente a um protótipo igual a s_j.
    """
    n_samples = X.shape[0]
    R = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        norm_i = int(np.sum(X[i]))
        for j in range(n_samples):
            R[i, j] = int(np.sum(X[i] * X[j])) / norm_i
    return R


def plot_clusters_overview(X, situations, results_by_rho, output_path):
    """
    Gera painel 2x2 mostrando, para cada grau de vigilância (rho = 0.5, 0.8, 0.9, 0.99),
    as situações agrupadas em cada classe ativa, seus bits x1..x16 e o protótipo
    top-down (t_j) aprendido por cada classe.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5))
    axes = axes.ravel()

    palette = [
        '#1d4ed8', '#059669', '#d97706', '#dc2626',
        '#7c3aed', '#0891b2', '#db2777', '#4f46e5',
    ]

    for ax, rho in zip(axes, RHOS_TARGET):
        info = results_by_rho[rho]
        clusters = info['clusters']
        n_active = len(clusters)

        # Monta matriz visual: para cada classe, linhas das situações + linha do protótipo
        rows_data = []
        row_labels = []
        row_is_proto = []
        row_class_idx = []
        group_boundaries = []

        cursor = 0
        for c_idx, cl in enumerate(clusters):
            start = cursor
            for sit in cl['situacoes']:
                rows_data.append(X[sit - 1])
                row_labels.append(f'Sit. {sit:2d}  (C{cl["classe"]})')
                row_is_proto.append(False)
                row_class_idx.append(c_idx)
                cursor += 1
            # Linha do protótipo top-down da classe
            rows_data.append(cl['prototipo'])
            row_labels.append(f'Prot. C{cl["classe"]} (N{cl["neuronio_f2"]})')
            row_is_proto.append(True)
            row_class_idx.append(c_idx)
            cursor += 1
            group_boundaries.append((start, cursor - 1, cl))

        mat = np.array(rows_data, dtype=int)
        n_rows, n_cols = mat.shape

        # Renderiza célula por célula com código de cor por classe (situação vs protótipo)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

        for r_i in range(n_rows):
            c_color = palette[row_class_idx[r_i] % len(palette)]
            is_proto = row_is_proto[r_i]
            for c_j in range(n_cols):
                val = mat[r_i, c_j]
                if is_proto:
                    face = c_color if val == 1 else '#f1f5f9'
                    edge = '#0f172a' if val == 1 else '#cbd5e1'
                    alpha = 0.95 if val == 1 else 1.0
                    txt_color = 'white' if val == 1 else '#94a3b8'
                else:
                    face = c_color if val == 1 else '#ffffff'
                    edge = '#cbd5e1'
                    alpha = 0.52 if val == 1 else 1.0
                    txt_color = '#0f172a' if val == 1 else '#cbd5e1'

                rect = plt.Rectangle(
                    (c_j - 0.46, r_i - 0.44), 0.92, 0.88,
                    facecolor=face, edgecolor=edge,
                    linewidth=0.9 if is_proto else 0.5, alpha=alpha
                )
                ax.add_patch(rect)
                ax.text(
                    c_j, r_i, str(val), ha='center', va='center',
                    fontsize=7.3 if n_rows > 15 else 8.0,
                    fontweight='bold' if (is_proto and val == 1) else 'normal',
                    color=txt_color
                )

        # Caixas delimitando cada classe ativa
        for start, end, cl in group_boundaries:
            c_color = palette[(cl['classe'] - 1) % len(palette)]
            box = plt.Rectangle(
                (-0.49, start - 0.48), n_cols - 0.02, (end - start) + 0.96,
                fill=False, edgecolor=c_color, linewidth=1.8
            )
            ax.add_patch(box)

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([f'$x_{{{k}}}$' for k in range(1, 17)], fontsize=8.5)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7.6 if n_rows > 15 else 8.4)
        for tick, is_proto in zip(ax.get_yticklabels(), row_is_proto):
            if is_proto:
                tick.set_fontweight('bold')
                tick.set_color('#0f172a')

        ax.set_title(
            f'Vigilância $\\rho = {rho}$  |  {n_active} Classes Ativas  '
            f'(Convergência: {info["epochs"]} épocas)',
            fontsize=11.5, fontweight='bold', pad=10, color='#0f172a'
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = [
        Patch(facecolor='#1d4ed8', alpha=0.52, edgecolor='#cbd5e1', label='Bit = 1 (Situação)'),
        Patch(facecolor='#1d4ed8', alpha=0.95, edgecolor='#0f172a', label='Bit = 1 (Protótipo top-down $t_j$)'),
        Patch(facecolor='#ffffff', edgecolor='#cbd5e1', label='Bit = 0'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=3,
        frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=10,
        bbox_to_anchor=(0.5, 0.005)
    )
    fig.suptitle(
        'Rede Neural ART-1 — Agrupamento das 10 Situações Industriais e Protótipos por Grau de Vigilância ($\\rho$)',
        fontsize=13.5, fontweight='bold', y=0.985, color='#0f172a'
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_vigilance_and_similarity(X, situations, results_by_rho, output_path):
    """
    Gera gráfico analítico em dois painéis:
      - Esquerda: Curva de sensibilidade do número de classes ativas em função de rho in [0.10, 1.00].
      - Direita: Matriz 10x10 de similaridade ART-1 (||s_i AND s_j||_1 / ||s_i||_1).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={'width_ratios': [1.05, 1.15]})

    # Painel 1: Varredura de rho
    rhos_grid = np.round(np.linspace(0.10, 1.00, 91), 2)
    active_counts = []
    for r_val in rhos_grid:
        model = ART1(n_features=X.shape[1], max_clusters=len(X), rho=float(r_val), L=2.0)
        model.fit(X)
        active_counts.append(len(model.get_active_clusters(situations)))

    ax1.step(rhos_grid, active_counts, where='post', color='#2563eb', linewidth=2.5,
             label='Classes ativas (ART-1, $L = 2$)')
    ax1.fill_between(rhos_grid, active_counts, step='post', color='#3b82f6', alpha=0.12)

    # Destaca os 4 pontos pedidos no enunciado
    offsets = {
        0.5: (0.04, -0.80),
        0.8: (-0.23, 0.25),
        0.9: (-0.25, -0.15),
        0.99: (-0.22, 0.42),
    }
    for rho in RHOS_TARGET:
        k = len(results_by_rho[rho]['clusters'])
        ax1.scatter([rho], [k], s=110, color='#dc2626', edgecolor='white', linewidth=1.8, zorder=5)
        dx, dy = offsets[rho]
        ax1.annotate(
            f'$\\rho={rho}$\n({k} classes)',
            xy=(rho, k), xytext=(rho + dx, k + dy),
            fontsize=9.2, fontweight='bold', color='#991b1b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef2f2', edgecolor='#fca5a5', lw=1.0),
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.2)
        )

    ax1.set_xlim(0.08, 1.03)
    ax1.set_ylim(0.5, 9.2)
    ax1.set_yticks(range(1, 10))
    ax1.set_xlabel('Parâmetro de Vigilância ($\\rho$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Número de Classes Ativas', fontsize=11, fontweight='bold')
    ax1.set_title('Granularidade dos Agrupamentos vs. Vigilância $\\rho$',
                  fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9.5, frameon=True)

    # Painel 2: Matriz de similaridade ART-1 entre as 10 situações
    R = compute_pairwise_match_matrix(X)
    im = ax2.imshow(R, cmap='YlGnBu', vmin=0.3, vmax=1.0, aspect='auto')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('$r(s_i, s_j) = \\|s_i \\cap s_j\\|_1 / \\|s_i\\|_1$', fontsize=10)

    ax2.set_xticks(np.arange(len(situations)))
    ax2.set_xticklabels([f'S{s}' for s in situations], fontsize=9.5, fontweight='bold')
    ax2.set_yticks(np.arange(len(situations)))
    ax2.set_yticklabels([f'S{s} ($\\|s\\|_1$={int(X[s-1].sum())})' for s in situations], fontsize=9)
    ax2.set_xlabel('Protótipo de Referência ($s_j$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Entrada Apresentada ($s_i$)', fontsize=11, fontweight='bold')
    ax2.set_title('Matriz de Correspondência ART-1 entre Situações',
                  fontsize=12, fontweight='bold', pad=10)

    # Anotação numérica + destaque nos pares naturais (r >= 0.80 fora da diagonal)
    for i in range(len(situations)):
        for j in range(len(situations)):
            val = R[i, j]
            txt_color = 'white' if val >= 0.72 else '#0f172a'
            is_pair = (i != j) and ((i % 5) == (j % 5))
            ax2.text(
                j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8.2, fontweight='bold' if (is_pair or i == j) else 'normal',
                color=txt_color
            )
            if is_pair:
                rect = plt.Rectangle(
                    (j - 0.47, i - 0.47), 0.94, 0.94,
                    fill=False, edgecolor='#facc15', linewidth=2.0
                )
                ax2.add_patch(rect)

    fig.suptitle(
        'Análise de Sensibilidade da Vigilância e Estrutura de Similaridade das Situações Industriais',
        fontsize=13, fontweight='bold', y=1.01, color='#0f172a'
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print('=' * 100)
    print('CEFET-MG | LAB. INTELIGÊNCIA ARTIFICIAL - TRABALHO REDE NEURAL ART-1')
    print('Agrupamento de Situações de um Processo Industrial (10 Situações x 16 Variáveis Binárias)')
    print('=' * 100)

    df_train, df_test = load_data(base_dir)
    X_train = df_train[FEATURE_COLS].values.astype(int)
    situations = df_train['situacao'].values.astype(int)
    X_test = df_test[FEATURE_COLS].values.astype(int)

    print(f'\n[INFO] Dados carregados: {X_train.shape[0]} situações x {X_train.shape[1]} variáveis binárias.')
    print(f'[INFO] Parâmetros canônicos ART-1: L = 2.0 | t_ji(0) = 1 | b_ij(0) = 1/(1+16) = {1/17:.6f}')
    print(f'[INFO] Conjunto de teste idêntico ao de treinamento: {np.array_equal(X_train, X_test)}')

    # Identifica padrões idênticos nos dados
    print('\n[INFO] Situações com vetores binários idênticos:')
    for i in range(len(X_train)):
        for j in range(i + 1, len(X_train)):
            if np.array_equal(X_train[i], X_train[j]):
                print(f'       - Situação {situations[i]} == Situação {situations[j]} (||s||_1 = {X_train[i].sum()})')

    results_by_rho = {}
    summary_rows = []
    classification_rows = []

    for rho in RHOS_TARGET:
        model = ART1(n_features=X_train.shape[1], max_clusters=len(X_train), rho=rho, L=2.0, max_epochs=100)
        model.fit(X_train)
        preds_test, ratios_test, _ = model.predict(X_test)
        assert np.array_equal(preds_test, model.assignments_), 'Divergência entre treino estável e inferência!'

        clusters = model.get_active_clusters(situations)
        neuron_to_class = {cl['neuronio_f2'] - 1: cl['classe'] for cl in clusters}
        committed_neurons = [int(j + 1) for j in range(model.max_clusters) if model.committed[j]]
        active_neurons = [cl['neuronio_f2'] for cl in clusters]
        emptied_neurons = sorted(set(committed_neurons) - set(active_neurons))

        results_by_rho[rho] = {
            'model': model,
            'clusters': clusters,
            'epochs': model.epochs_,
            'total_resets': model.total_resets_,
            'emptied_neurons': emptied_neurons,
        }

        print('\n' + '=' * 100)
        print(f'SIMULAÇÃO ART-1 — GRAU DE VIGILÂNCIA rho = {rho}')
        print('=' * 100)
        print(f'  • Classes ativas: {len(clusters)}')
        print(f'  • Convergência:   {model.epochs_} épocas | Total de resets: {model.total_resets_}')
        if emptied_neurons:
            print(f'  • Observação:     Neurônio(s) F2 comprometido(s) na época 1 e esvaziado(s) após '
                  f'migração na convergência: {emptied_neurons}')

        print('\n  Agrupamentos e Protótipos Top-Down (t_j):')
        groups_str_parts = []
        for cl in clusters:
            proto_str = ''.join(str(b) for b in cl['prototipo'])
            sits_str = ', '.join(f'S{s}' for s in cl['situacoes'])
            groups_str_parts.append(f"C{cl['classe']}: {{{', '.join(str(s) for s in cl['situacoes'])}}}")
            print(f"    - Classe {cl['classe']} (Neurônio F2 = {cl['neuronio_f2']:2d}) | "
                  f"Situações: [{sits_str:18s}] | "
                  f"Protótipo t_j = [{proto_str}] (||t_j||_1 = {cl['bits_ativos_prototipo']:2d})")

        print('\n  Detalhamento por Situação:')
        for idx, sit in enumerate(situations):
            neuron_idx = int(model.assignments_[idx])
            cls_num = neuron_to_class[neuron_idx]
            norm_s = int(np.sum(X_train[idx]))
            proto = model.T[neuron_idx]
            norm_and = int(np.sum(X_train[idx] * proto))
            r_val = float(ratios_test[idx])
            proto_str = ''.join(str(b) for b in proto)
            print(f'    Situação {sit:2d} (||s||_1={norm_s:2d}) -> Classe {cls_num} '
                  f'(F2={neuron_idx+1:2d}) | ||s∩t||_1={norm_and:2d} | '
                  f'r = {norm_and:2d}/{norm_s:2d} = {r_val:.4f} >= {rho}')

            classification_rows.append({
                'rho': rho,
                'situacao': int(sit),
                'norma_entrada': norm_s,
                'classe_ativa': cls_num,
                'neuronio_f2': neuron_idx + 1,
                'bits_comuns_prototipo': norm_and,
                'similaridade_r': round(r_val, 4),
                'prototipo_top_down': proto_str,
            })

        summary_rows.append({
            'rho': rho,
            'classes_ativas': len(clusters),
            'epocas_convergencia': model.epochs_,
            'total_resets': model.total_resets_,
            'neuronios_f2_ativos': str(active_neurons),
            'neuronios_esvaziados': str(emptied_neurons),
            'agrupamentos': ' | '.join(groups_str_parts),
        })

    # Verificação de robustez variando o parâmetro L > 1
    print('\n' + '=' * 100)
    print('VERIFICAÇÃO DE ROBUSTEZ (SENSIBILIDADE AO PARÂMETRO L)')
    print('=' * 100)
    for L_test in [1.5, 2.0, 3.0, 5.0]:
        line_parts = []
        for rho in RHOS_TARGET:
            m_test = ART1(n_features=X_train.shape[1], max_clusters=len(X_train), rho=rho, L=L_test)
            m_test.fit(X_train)
            cls_test = [cl['situacoes'] for cl in m_test.get_active_clusters(situations)]
            line_parts.append(f'rho={rho}: {len(cls_test)} cls')
        print(f'  L = {L_test:3.1f} -> ' + ' | '.join(line_parts))

    # Salva tabelas CSV
    df_summary = pd.DataFrame(summary_rows)
    df_classif = pd.DataFrame(classification_rows)
    summary_csv_path = os.path.join(base_dir, 'tabela_resumo_art1.csv')
    classif_csv_path = os.path.join(base_dir, 'tabela_classificacao_art1.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    df_classif.to_csv(classif_csv_path, index=False)

    print('\n' + '=' * 100)
    print('RESUMO COMPARATIVO FINAL')
    print('=' * 100)
    print(df_summary[['rho', 'classes_ativas', 'epocas_convergencia', 'agrupamentos']].to_string(index=False))

    # Gera visualizações
    clusters_img_path = os.path.join(base_dir, 'agrupamentos_art1.png')
    analysis_img_path = os.path.join(base_dir, 'analise_vigilancia_art1.png')
    plot_clusters_overview(X_train, situations, results_by_rho, clusters_img_path)
    plot_vigilance_and_similarity(X_train, situations, results_by_rho, analysis_img_path)

    print(f'\n[OK] Tabela resumo salva em:        {summary_csv_path}')
    print(f'[OK] Tabela classificação salva em: {classif_csv_path}')
    print(f'[OK] Gráfico agrupamentos salvo em: {clusters_img_path}')
    print(f'[OK] Gráfico análise salvo em:      {analysis_img_path}')


if __name__ == '__main__':
    main()
