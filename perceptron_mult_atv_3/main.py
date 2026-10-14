"""
PMC2 - Perceptron Multicamadas (4 -> 15 -> 3) para classificação de conservantes.
Itens:
 1. Backpropagation padrão   (eta=0.1, epsilon=1e-6, sigmoide, pesos iniciais U[0,1])
 2. Backpropagation c/ momentum (mesmos pesos iniciais, alpha=0.9)
    -> gráficos EQM x época (lado a lado) + tempo de processamento
 3. Pós-processamento (arredondamento simétrico: y >= 0.5 -> 1, senão 0)
 4. Validação no conjunto de teste e taxa de acerto (%)
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Parâmetros do enunciado ----------------
N_ENTRADAS = 4
N_OCULTOS = 15
N_SAIDAS = 3
ETA = 0.1
ALPHA = 0.9
EPSILON = 1e-6
MAX_EPOCAS = 100000  # trava de segurança
SEED = 42            # reprodutibilidade (pesos iniciais idênticos nos dois treinos)

COLS_X = ['x1', 'x2', 'x3', 'x4']
COLS_D = ['d1', 'd2', 'd3']


def sigmoid(u):
    """Função de ativação logística."""
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def sigmoid_derivative(y):
    """Derivada da logística em função da saída y = g(u)."""
    return y * (1.0 - y)


def carregar(caminho):
    """Lê o CSV e retorna X (com bias -1 na 1ª coluna), D e o DataFrame."""
    df = pd.read_csv(caminho)
    X = df[COLS_X].values.astype(float)
    X_bias = np.c_[np.full((len(X), 1), -1.0), X]
    D = df[COLS_D].values.astype(float)
    return X_bias, D, df


def inicializar_pesos():
    """W1: (15, 5) e W2: (3, 16), valores aleatórios entre 0 e 1."""
    W1 = np.random.rand(N_OCULTOS, N_ENTRADAS + 1)
    W2 = np.random.rand(N_SAIDAS, N_OCULTOS + 1)
    return W1, W2


def forward(x, W1, W2):
    """Propagação adiante para uma amostra (x já contém o bias)."""
    y1 = sigmoid(W1 @ x)                  # (15,)
    y1_bias = np.insert(y1, 0, -1.0)      # (16,)
    y2 = sigmoid(W2 @ y1_bias)            # (3,)
    return y1, y1_bias, y2


def calcular_eqm(X, D, W1, W2):
    """EQM = (1/p) * soma_k 0.5 * soma_j (d_j - y_j)^2"""
    total = 0.0
    for x, d in zip(X, D):
        _, _, y = forward(x, W1, W2)
        total += 0.5 * np.sum((d - y) ** 2)
    return total / len(X)


def treinar(X, D, W1_ini, W2_ini, eta=ETA, alpha=0.0, epsilon=EPSILON, max_epocas=MAX_EPOCAS):
    """
    Backpropagation online. alpha=0 -> padrão; alpha>0 -> com momentum.
    Retorna pesos finais, épocas, histórico do EQM e tempo (s).
    """
    W1, W2 = W1_ini.copy(), W2_ini.copy()
    W1_ant, W2_ant = W1.copy(), W2.copy()  # pesos na iteração anterior (momentum)

    eqm_ant = calcular_eqm(X, D, W1, W2)
    historico = [eqm_ant]
    epocas = 0
    inicio = time.perf_counter()

    while epocas < max_epocas:
        for x, d in zip(X, D):
            y1, y1_bias, y2 = forward(x, W1, W2)

            delta2 = (d - y2) * sigmoid_derivative(y2)                  # (3,)
            delta1 = (W2[:, 1:].T @ delta2) * sigmoid_derivative(y1)    # (15,)

            W1_atual, W2_atual = W1.copy(), W2.copy()
            W2 = W2 + alpha * (W2 - W2_ant) + eta * np.outer(delta2, y1_bias)
            W1 = W1 + alpha * (W1 - W1_ant) + eta * np.outer(delta1, x)
            W1_ant, W2_ant = W1_atual, W2_atual

        epocas += 1
        eqm = calcular_eqm(X, D, W1, W2)
        historico.append(eqm)
        if abs(eqm - eqm_ant) <= epsilon:
            break
        eqm_ant = eqm

    tempo = time.perf_counter() - inicio
    return W1, W2, epocas, historico, tempo


def pos_processar(y):
    """Arredondamento simétrico: 1 se y >= 0.5, senão 0."""
    return (y >= 0.5).astype(int)


def plotar(hist_padrao, hist_mom, caminho):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist, titulo, cor in [
        (axes[0], hist_padrao, 'Backpropagation padrão', 'royalblue'),
        (axes[1], hist_mom, 'Backpropagation com momentum', 'darkorange'),
    ]:
        ax.plot(hist, color=cor, linewidth=2)
        ax.set_title(f'{titulo} ({len(hist) - 1} épocas)', fontweight='bold')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('EQM')
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()


def validar(nome, X, D, df, W1, W2):
    Y = np.array([forward(x, W1, W2)[2] for x in X])
    Y_pos = pos_processar(Y)
    acertos = np.all(Y_pos == D.astype(int), axis=1)
    taxa = acertos.mean() * 100

    tabela = df[['Amostra'] + COLS_X + COLS_D].copy()
    for j in range(N_SAIDAS):
        tabela[f'y{j+1}'] = np.round(Y[:, j], 4)
    for j in range(N_SAIDAS):
        tabela[f'y{j+1}_pos'] = Y_pos[:, j]
    tabela['acerto'] = np.where(acertos, 'OK', 'X')

    print('\n' + '=' * 90)
    print(f'ITEM 4 - VALIDAÇÃO ({nome})')
    print('=' * 90)
    print(tabela.to_string(index=False))
    print(f'Taxa de acerto: {taxa:.2f}% ({acertos.sum()}/{len(acertos)})')
    return taxa


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    X_tr, D_tr, _ = carregar(os.path.join(base, 'data', 'treinamento.csv'))
    X_te, D_te, df_te = carregar(os.path.join(base, 'data', 'teste.csv'))

    np.random.seed(SEED)
    W1_ini, W2_ini = inicializar_pesos()

    print('=' * 90)
    print(f'PMC {N_ENTRADAS}-{N_OCULTOS}-{N_SAIDAS} | eta={ETA} | epsilon={EPSILON} | '
          f'amostras treino={len(X_tr)} | teste={len(X_te)}')
    print('=' * 90)

    # Item 1
    W1_p, W2_p, ep_p, hist_p, t_p = treinar(X_tr, D_tr, W1_ini, W2_ini, alpha=0.0)
    print(f'ITEM 1 - Padrão  : épocas={ep_p:6d} | EQM final={hist_p[-1]:.6f} | tempo={t_p:.2f}s')

    # Item 2
    W1_m, W2_m, ep_m, hist_m, t_m = treinar(X_tr, D_tr, W1_ini, W2_ini, alpha=ALPHA)
    print(f'ITEM 2 - Momentum: épocas={ep_m:6d} | EQM final={hist_m[-1]:.6f} | tempo={t_m:.2f}s')

    grafico = os.path.join(base, 'graficos_eqm.png')
    plotar(hist_p, hist_m, grafico)
    print(f'Gráficos salvos em: {grafico}')

    # Itens 3 e 4
    validar('Backpropagation padrão', X_te, D_te, df_te, W1_p, W2_p)
    validar('Backpropagation com momentum', X_te, D_te, df_te, W1_m, W2_m)


if __name__ == '__main__':
    main()
