# Respostas da Atividade: Perceptron

Este documento contém as respostas para a atividade proposta, com base na execução do algoritmo do Perceptron.

## 1. Resultados dos 5 Treinamentos

Abaixo estão os resultados de 5 execuções independentes do algoritmo de treinamento do Perceptron. Como os pesos iniciais são gerados de forma **aleatória** (sem seed fixa), o número de épocas e os pesos finais de convergência variam a cada treinamento.

*Exemplo de execução gerada pelo código:*

- **Treinamento 1:**
  - **Pesos Iniciais:** `w0=0.9434, w1=0.1194, w2=0.4064, w3=0.4284`
  - **Pesos Finais:** ` w0=-3.1566, w1=1.5914, w2=2.5326, w3=-0.7493`
  - **Número de Épocas:** 390

- **Treinamento 2:**
  - **Pesos Iniciais:** `w0=0.9827, w1=0.0775, w2=0.2225, w3=0.7837`
  - **Pesos Finais:** `w0=-3.0573, w1=1.5506, w2=2.4587, w3=-0.7289`
  - **Número de Épocas:** 430

- **Treinamento 3:**
  - **Pesos Iniciais:** `w0=0.6280, w1=0.5477, w2=0.8247, w3=0.7888`
  - **Pesos Finais:** `w0=-3.0720, w1=1.5655, w2=2.4570, w3=-0.7317`
  - **Número de Épocas:** 415

- **Treinamento 4:**
  - **Pesos Iniciais:** `w0=0.0772, w1=0.7345, w2=0.7019, w3=0.2213`
  - **Pesos Finais:** `w0=-3.1028, w1=1.5744, w2=2.5039, w3=-0.7395`
  - **Número de Épocas:** 406

- **Treinamento 5:**
  - **Pesos Iniciais:** `w0=0.3913, w1=0.4041, w2=0.9894, w3=0.4619`
  - **Pesos Finais:** `w0=-3.0887, w1=1.5672, w2=2.4816, w3=-0.7367`
  - **Número de Épocas:** 432

*(Nota: Você pode rodar o script novamente para obter novos valores caso seja necessário documentar outra execução).*

---

## 2. Análise sobre a convergência (Épocas e Pesos)

**Por que o número de épocas e o vetor de pesos finais variam em cada treinamento?**

Isso ocorre porque os pesos iniciais (`w0, w1, w2, w3`) são inicializados com **valores aleatórios** em cada execução (`np.random.rand`). Dependendo do ponto de partida (pesos iniciais) no hiperplano de separação:
1. O algoritmo do Perceptron precisará de mais ou menos ajustes (épocas) para encontrar uma superfície de decisão que separe os dados corretamente.
2. Como o problema é linearmente separável e existem infinitos planos que podem separar as classes, cada inicialização aleatória fará com que o algoritmo convirja para uma **superfície de separação diferente** (embora todas sejam válidas e não possuam erros de classificação nos dados de treino). Por isso os pesos finais variam.

---

## 3. Resultados da Classificação do Conjunto de Teste

A tabela abaixo mostra a classificação das amostras do arquivo de teste (`oleo_teste.csv`) utilizando os modelos treinados nas 5 execuções (T1 a T5):

| Amostra | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:-------:|:------:|:------:|:------:|:------:|:------:|
|    1    |   -1   |   -1   |   -1   |   -1   |   -1   |
|    2    |    1   |    1   |    1   |    1   |    1   |
|    3    |    1   |    1   |    1   |    1   |    1   |
|    4    |    1   |    1   |    1   |    1   |    1   |
|    5    |    1   |    1   |    1   |    1   |    1   |
|    6    |    1   |    1   |    1   |    1   |    1   |
|    7    |   -1   |   -1   |   -1   |   -1   |   -1   |
|    8    |    1   |    1   |    1   |    1   |    1   |
|    9    |   -1   |   -1   |   -1   |   -1   |   -1   |
|   10    |   -1   |   -1   |   -1   |   -1   |   -1   |

**Observação:**
Apesar dos vetores de pesos finais serem diferentes para cada treinamento (T1 a T5), podemos notar que a classificação para o conjunto de testes foi **exatamente a mesma** em todos eles. Isso confirma que os diferentes planos de separação obtidos pelo Perceptron conseguiram generalizar perfeitamente os dados de teste da mesma forma.
