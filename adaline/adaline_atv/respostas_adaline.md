# Respostas da Atividade: ADALINE

Este documento contém as respostas para a atividade da rede neural ADALINE, treinada usando a Regra Delta com taxa de aprendizado $\eta = 0.0025$ e precisão $\epsilon = 10^{-6}$.

## 1 e 2. Resultados dos 5 Treinamentos

Abaixo estão registrados os resultados de 5 execuções com inicialização de pesos aleatórios entre 0 e 1.

- **Treinamento 1 (T1):**
  - **Pesos Iniciais:** `w0=0.5374, w1=0.1806, w2=0.2911, w3=0.8560, w4=0.9838`
  - **Pesos Finais:** `w0=-1.8131, w1=1.3129, w2=1.6424, w3=-0.4277, w4=-1.1778`
  - **Número de Épocas:** 937

- **Treinamento 2 (T2):**
  - **Pesos Iniciais:** `w0=0.8963, w1=0.1762, w2=0.5467, w3=0.0928, w4=0.8110`
  - **Pesos Finais:** `w0=-1.8131, w1=1.3129, w2=1.6423, w3=-0.4278, w4=-1.1778`
  - **Número de Épocas:** 920

- **Treinamento 3 (T3):**
  - **Pesos Iniciais:** `w0=0.4176, w1=0.6484, w2=0.3337, w3=0.2308, w4=0.3253`
  - **Pesos Finais:** `w0=-1.8132, w1=1.3129, w2=1.6423, w3=-0.4278, w4=-1.1778`
  - **Número de Épocas:** 894

- **Treinamento 4 (T4):**
  - **Pesos Iniciais:** `w0=0.6250, w1=0.5989, w2=0.5988, w3=0.8354, w4=0.6686`
  - **Pesos Finais:** `w0=-1.8131, w1=1.3129, w2=1.6424, w3=-0.4276, w4=-1.1778`
  - **Número de Épocas:** 925

- **Treinamento 5 (T5):**
  - **Pesos Iniciais:** `w0=0.4134, w1=0.2512, w2=0.4267, w3=0.0409, w4=0.7300`
  - **Pesos Finais:** `w0=-1.8131, w1=1.3128, w2=1.6423, w3=-0.4278, w4=-1.1778`
  - **Número de Épocas:** 895

---

## 3. Gráficos de Erro Quadrático Médio (EQM)

O gráfico plotando o declínio do Erro Quadrático Médio em relação ao número de épocas para os treinamentos 1 e 2 foi gerado separadamente pelo código.
Você pode visualizar a curva de aprendizado na imagem correspondente na pasta do projeto: `grafico_eqm.png`.

---

## 4. Resultados da Classificação (Conjunto de Teste)

A tabela abaixo exibe o resultado da classificação para os 15 sinais de teste. A predição segue a seguinte convenção:
* **-1** = Comando para a **Válvula A**
* **+1** = Comando para a **Válvula B**

| Amostra | y(T1) | y(T2) | y(T3) | y(T4) | y(T5) |
|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|    1    |   -1  |   -1  |   -1  |   -1  |   -1  |
|    2    |   -1  |   -1  |   -1  |   -1  |   -1  |
|    3    |    1  |    1  |    1  |    1  |    1  |
|    4    |   -1  |   -1  |   -1  |   -1  |   -1  |
|    5    |   -1  |   -1  |   -1  |   -1  |   -1  |
|    6    |    1  |    1  |    1  |    1  |    1  |
|    7    |    1  |    1  |    1  |    1  |    1  |
|    8    |    1  |    1  |    1  |    1  |    1  |
|    9    |    1  |    1  |    1  |    1  |    1  |
|   10    |   -1  |   -1  |   -1  |   -1  |   -1  |
|   11    |   -1  |   -1  |   -1  |   -1  |   -1  |
|   12    |    1  |    1  |    1  |    1  |    1  |
|   13    |   -1  |   -1  |   -1  |   -1  |   -1  |
|   14    |   -1  |   -1  |   -1  |   -1  |   -1  |
|   15    |    1  |    1  |    1  |    1  |    1  |

---

## 5. Análise da Convergência (Pesos Inalterados vs Épocas Variáveis)

**Pergunta (Item 5):** Embora o número de épocas de cada treinamento realizado no item 2 seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.

**Resposta:**
Diferente do Perceptron Clássico, que busca qualquer hiperplano separador válido (resultando em pesos finais diferentes para cada inicialização), a rede ADALINE ajusta seus pesos baseada no **Erro Quadrático Médio (EQM) calculado antes da função de ativação**. 

Matematicamente, a superfície de erro quadrático associada à Regra Delta tem o formato de um **paraboloide multidimensional** que possui um **único ponto de mínimo global**. O algoritmo de gradiente descendente garante que, independentemente do ponto de partida (pesos iniciais aleatórios), a rede sempre "escorregará" em direção ao fundo dessa única bacia de erro.

A variação no número de épocas ocorre unicamente porque alguns vetores de inicialização aleatória nascem mais "distantes" ou em encostas mais íngremes/suaves do mínimo global, demandando assim mais ou menos passos de ajuste (épocas) para atravessar a superfície e atingir o critério de parada ($\Delta EQM < 10^{-6}$). Contudo, como só existe um único "fundo", os pesos finais convergem invariavelmente para o mesmo local.
