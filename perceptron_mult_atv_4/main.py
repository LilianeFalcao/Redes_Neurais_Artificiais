"""
PMC3 - Perceptron Multicamadas com topologia Time Delay (TDNN) para previsão de série temporal.
Topologias:  Rede 1 (p=5,  N1=10) | Rede 2 (p=10, N1=15) | Rede 3 (p=15, N1=25)
Treino: backpropagation com momentum, sigmoide, eta=0.1, alpha=0.8, epsilon=0.5e-6,
        pesos iniciais U[0,1] distintos em cada treinamento (3 por rede).
Validação: previsão RECURSIVA para t=101..120 (previsões realimentam a janela).
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Parâmetros do enunciado ----------------
TOPOLOGIAS = [
    {'nome': 'Rede 1', 'p': 5, 'n1': 10},
    {'nome': 'Rede 2', 'p': 10, 'n1': 15},
    {'nome': 'Rede 3', 'p': 15, 'n1': 25},
]
N_TREINOS = 3
ETA = 0.1
ALPHA = 0.8
EPSILON = 0.5e-6
MAX_EPOCAS = 20000  # trava de segurança


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def sigmoid_derivative(y):
    return y * (1.0 - y)


def carregar_serie(caminho):
    return pd.read_csv(caminho)['f(t)'].values.astype(float)


def montar_janelas(serie, p):
    """
    Cada padrão: entrada [-1, x(t-1), x(t-2), ..., x(t-p)] e alvo f(t).
    Gera len(serie) - p padrões.
    """
    X, D = [], []
    for t in range(p, len(serie)):
        janela = serie[t - p:t][::-1]          # x(t-1) primeiro
        X.append(np.insert(janela, 0, -1.0))
        D.append(serie[t])
    return np.array(X), np.array(D)


def forward(x, W1, W2):
    y1 = sigmoid(W1 @ x)
    y1_bias = np.insert(y1, 0, -1.0)
    y2 = sigmoid(W2 @ y1_bias)[0]
    return y1, y1_bias, y2


def calcular_eqm(X, D, W1, W2):
    erros = [0.5 * (d - forward(x, W1, W2)[2]) ** 2 for x, d in zip(X, D)]
    return float(np.mean(erros))


def treinar(X, D, n1, eta=ETA, alpha=ALPHA, epsilon=EPSILON, max_epocas=MAX_EPOCAS):
    """Backpropagation com momentum (ajuste online)."""
    p = X.shape[1] - 1
    np.random.seed(None)                       # pesos iniciais distintos a cada treino
    W1 = np.random.rand(n1, p + 1)
    W2 = np.random.rand(1, n1 + 1)
    W1_ant, W2_ant = W1.copy(), W2.copy()

    eqm_ant = calcular_eqm(X, D, W1, W2)
    historico = [eqm_ant]
    epocas = 0
    inicio = time.perf_counter()

    while epocas < max_epocas:
        for x, d in zip(X, D):
            y1, y1_bias, y2 = forward(x, W1, W2)
            delta2 = (d - y2) * sigmoid_derivative(y2)
            delta1 = W2[0, 1:] * delta2 * sigmoid_derivative(y1)

            W1_atual, W2_atual = W1.copy(), W2.copy()
            W2 = W2 + alpha * (W2 - W2_ant) + eta * delta2 * y1_bias.reshape(1, -1)
            W1 = W1 + alpha * (W1 - W1_ant) + eta * np.outer(delta1, x)
            W1_ant, W2_ant = W1_atual, W2_atual

        epocas += 1
        eqm = calcular_eqm(X, D, W1, W2)
        historico.append(eqm)
        if abs(eqm - eqm_ant) <= epsilon:
            break
        eqm_ant = eqm

    return {'W1': W1, 'W2': W2, 'epocas': epocas, 'eqm': historico[-1],
            'historico': historico, 'tempo': time.perf_counter() - inicio}


def prever_recursivo(serie_treino, n_passos, p, W1, W2):
    """Previsão recursiva: cada previsão entra na janela do passo seguinte."""
    historico = list(serie_treino)
    previsoes = []
    for _ in range(n_passos):
        janela = np.array(historico[-p:][::-1])
        y = forward(np.insert(janela, 0, -1.0), W1, W2)[2]
        previsoes.append(y)
        historico.append(y)
    return np.array(previsoes)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    serie = carregar_serie(os.path.join(base, 'data', 'treinamento.csv'))   # t=1..100
    desejado = carregar_serie(os.path.join(base, 'data', 'teste.csv'))      # t=101..120
    t_teste = np.arange(len(serie) + 1, len(serie) + len(desejado) + 1)

    print('=' * 80)
    print(f'PMC3 - TDNN | eta={ETA} | alpha={ALPHA} | epsilon={EPSILON} | validação recursiva')
    print('=' * 80)

    # ---------------- Item 1: 3 treinamentos por rede ----------------
    resultados = {}
    for topo in TOPOLOGIAS:
        X, D = montar_janelas(serie, topo['p'])
        resultados[topo['nome']] = []
        for k in range(N_TREINOS):
            r = treinar(X, D, topo['n1'])
            r['nome'] = f'T{k + 1}'
            # ---------------- Item 3: validação recursiva ----------------
            y = prever_recursivo(serie, len(desejado), topo['p'], r['W1'], r['W2'])
            er = np.abs(desejado - y) / desejado * 100.0
            r.update({'y': y, 'erro_rel': er.mean(), 'variancia': er.var()})
            resultados[topo['nome']].append(r)
            print(f"{topo['nome']} (p={topo['p']:2d}, N1={topo['n1']:2d}) {r['nome']}: "
                  f"EQM={r['eqm']:.6f} | épocas={r['epocas']:5d} | tempo={r['tempo']:.1f}s | "
                  f"ERM={r['erro_rel']:.2f}% | var={r['variancia']:.2f}")

    nomes = [t['nome'] for t in TOPOLOGIAS]

    # ---------------- Item 2: tabela EQM / Épocas ----------------
    tab2 = pd.DataFrame({'Treinamento': [f'{k + 1}º (T{k + 1})' for k in range(N_TREINOS)]})
    for n in nomes:
        tab2[f'{n} EQM'] = [f"{r['eqm']:.6f}" for r in resultados[n]]
        tab2[f'{n} Épocas'] = [r['epocas'] for r in resultados[n]]
    print('\n' + '=' * 80 + '\nITEM 2 - RESULTADOS DOS TREINAMENTOS\n' + '=' * 80)
    print(tab2.to_string(index=False))

    # ---------------- Item 3: tabela de validação ----------------
    tab3 = pd.DataFrame({'Amostra': [f't = {t}' for t in t_teste], 'f(t)': desejado})
    for n in nomes:
        for r in resultados[n]:
            tab3[f"{n} ({r['nome']})"] = np.round(r['y'], 4)
    rodape = pd.DataFrame([
        ['Erro Relativo Médio (%)', ''] + [f"{r['erro_rel']:.2f}" for n in nomes for r in resultados[n]],
        ['Variância (%)', ''] + [f"{r['variancia']:.2f}" for n in nomes for r in resultados[n]],
    ], columns=tab3.columns)
    print('\n' + '=' * 80 + '\nITEM 3 - VALIDAÇÃO RECURSIVA (t=101..120)\n' + '=' * 80)
    print(pd.concat([tab3.astype(str), rodape]).to_string(index=False))
    tab2.to_csv(os.path.join(base, 'tabela_item2.csv'), index=False)
    pd.concat([tab3.astype(str), rodape]).to_csv(os.path.join(base, 'tabela_item3.csv'), index=False)

    # Melhor treinamento de cada rede: menor erro relativo médio na validação
    melhores = {n: min(resultados[n], key=lambda r: r['erro_rel']) for n in nomes}

    # ---------------- Item 4: EQM x épocas ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, n, cor in zip(axes, nomes, ['royalblue', 'darkorange', 'seagreen']):
        r = melhores[n]
        ax.plot(r['historico'], color=cor, linewidth=2)
        ax.set_title(f"{n} - {r['nome']} ({r['epocas']} épocas)", fontweight='bold')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('EQM')
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'item4_eqm.png'), dpi=300)
    plt.close()

    # ---------------- Item 5: desejado x estimado ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, n in zip(axes, nomes):
        r = melhores[n]
        ax.plot(t_teste, desejado, 'o-', color='black', label='Desejado')
        ax.plot(t_teste, r['y'], 's--', color='crimson', label='Estimado (recursivo)')
        ax.set_title(f"{n} - {r['nome']} (ERM={r['erro_rel']:.1f}%)", fontweight='bold')
        ax.set_xlabel('t')
        ax.set_ylabel('f(t)')
        ax.set_xticks(t_teste[::2])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base, 'item5_previsao.png'), dpi=300)
    plt.close()

    # ---------------- Item 6: configuração mais adequada ----------------
    melhor_rede = min(nomes, key=lambda n: melhores[n]['erro_rel'])
    r = melhores[melhor_rede]
    print('\n' + '=' * 80)
    for n in nomes:
        print(f"Melhor de {n}: {melhores[n]['nome']} | ERM={melhores[n]['erro_rel']:.2f}% | "
              f"var={melhores[n]['variancia']:.2f}")
    print(f"ITEM 6 - Configuração mais adequada: {melhor_rede} com {r['nome']} "
          f"(ERM={r['erro_rel']:.2f}%, variância={r['variancia']:.2f})")
    print('Gráficos: item4_eqm.png, item5_previsao.png | Tabelas: tabela_item2.csv, tabela_item3.csv')
    print('=' * 80)


if __name__ == '__main__':
    main()
