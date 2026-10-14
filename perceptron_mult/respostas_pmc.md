# Atividade: Perceptron Multicamadas (PMC) - Backpropagation

**Centro Federal de Educação Tecnológica de Minas Gerais – CEFET-MG**  
**Campus VIII – Varginha**  
**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 08/10/2026  
**Tema:** Demonstração do Algoritmo Backpropagation para Topologia PMC com Conexões Diretas  

---

## 1. Enunciado e Descrição do Problema

Visando uma aplicação específica, a equipe de engenheiros e cientistas de uma instituição projetou uma rede neural artificial do tipo perceptron multicamadas (PMC), constituída de três camadas, cuja topologia está ilustrada na figura a seguir.

### Informações referentes à topologia da rede:
- **Camada de Entrada (1ª camada):** Constituída de $N$ sinais de entrada $(X_1, X_2, \dots, X_N)$.
- **Camada Neural Escondida (2ª camada):** Constituída de $N_1$ neurônios.
- **Camada Neural de Saída (3ª camada):** Constituída de apenas $1$ neurônio.
- **Conjunto de Padrões de Treinamento:** Constituído de $P$ padrões.

### Objetivo:
Demonstrar de forma detalhada a **sequência** e o **ajuste das matrizes de pesos** entre cada uma das camadas quando se utiliza o algoritmo de aprendizagem **Backpropagation**, adotando a seguinte convenção:
- **Matriz $W_1$:** Matriz de pesos entre a 1ª e a 2ª camada (Entrada $\to$ Escondida).
- **Matriz $W_2$:** Matriz de pesos entre a 2ª e a 3ª camada (Escondida $\to$ Saída).
- **Matriz $W_3$:** Matriz de pesos entre a 1ª e a 3ª camada (Entrada $\to$ Saída / Conexão Direta).

---

## 2. Diagrama da Topologia e Convenções Notacionais

A característica diferencial desta arquitetura reside na presença da matriz **$W_3$**, que introduz conexões diretas (*shortcut / bypass connections*) da camada de entrada diretamente para o neurônio de saída, além das conexões convencionais intermediadas pela camada oculta ($W_1$ e $W_2$).

```
+-----------------------------------------------------------------------------+
|                     TOPOLOGIA DA REDE NEURAL ARTIFICIAL                     |
+-----------------------------------------------------------------------------+

   Camada 1 (Entrada)           Camada 2 (Escondida)        Camada 3 (Saída)
      [N sinais]                   [N1 neurônios]             [1 neurônio]

         +----+                         +----+
         | X1 |-----+----( W1 )-------->|  1 |----+
         +----+     |                   +----+    |
                    |                             +---( W2 )---+
         +----+     |                   +----+    |            |
         | X2 |-----+----( W1 )-------->|  2 |----+            v
         +----+     |                   +----+               +---+
           :        |                     :                  | 1 |----> y (Saída)
           :        |                   +----+               +---+
         +----+     +----( W1 )-------->| N1 |----+            ^
         | XN |                         +----+    |            |
         +----+                                   +------------+
           |                                                   |
           +-----------------------( W3 )----------------------+
                    (Conexões Diretas Entrada -> Saída)
```

### Definição das Dimensões:

Para um dado padrão de treinamento $p \in \{1, 2, \dots, P\}$:
- **Vetor de entrada:**
  $$X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix} \in \mathbb{R}^{N \times 1}$$
- **Matriz de pesos $W_1$ (1ª $\to$ 2ª camada):**
  $$W_1 = \begin{bmatrix} 
  w_{1, 11} & w_{1, 12} & \dots & w_{1, 1N} \\
  w_{1, 21} & w_{1, 22} & \dots & w_{1, 2N} \\
  \vdots & \vdots & \ddots & \vdots \\
  w_{1, N_1 1} & w_{1, N_1 2} & \dots & w_{1, N_1 N}
  \end{bmatrix} \in \mathbb{R}^{N_1 \times N}$$
  *(onde $w_{1, ij}$ pondera a conexão da entrada $x_j$ para o neurônio escondido $i$)*

- **Matriz de pesos $W_2$ (2ª $\to$ 3ª camada):**
  $$W_2 = \begin{bmatrix} w_{2, 11} & w_{2, 12} & \dots & w_{2, 1N_1} \end{bmatrix} \in \mathbb{R}^{1 \times N_1}$$
  *(onde $w_{2, 1i}$ pondera a conexão do neurônio escondido $i$ para o neurônio de saída)*

- **Matriz de pesos $W_3$ (1ª $\to$ 3ª camada):**
  $$W_3 = \begin{bmatrix} w_{3, 11} & w_{3, 12} & \dots & w_{3, 1N} \end{bmatrix} \in \mathbb{R}^{1 \times N}$$
  *(onde $w_{3, 1j}$ pondera a conexão direta da entrada $x_j$ para o neurônio de saída)*

- **Saída desejada (alvo):** $d \in \mathbb{R}$
- **Funções de ativação:** $g_1(\cdot)$ na camada escondida e $g_2(\cdot)$ na camada de saída, com suas respectivas derivadas $g_1'(\cdot)$ e $g_2'(\cdot)$.

---

## 3. Passo Forward (Propagação para a Frente)

Ao apresentar um padrão de entrada $X$:

### 3.1. Processamento na Camada Escondida (2ª Camada)
O potencial de ativação induzido no $i$-ésimo neurônio escondido ($i = 1, \dots, N_1$) é:
$$u_i^{(1)} = \sum_{j=1}^{N} w_{1, ij} x_j$$
Em notação matricial:
$$u^{(1)} = W_1 X \in \mathbb{R}^{N_1 \times 1}$$

O sinal de saída produzido pelo $i$-ésimo neurônio escondido é dado pela aplicação da função de ativação $g_1(\cdot)$:
$$y_i^{(1)} = g_1(u_i^{(1)})$$
Em notação vetorial:
$$y^{(1)} = g_1(u^{(1)}) = \begin{bmatrix} y_1^{(1)} \\ y_2^{(1)} \\ \vdots \\ y_{N_1}^{(1)} \end{bmatrix} \in \mathbb{R}^{N_1 \times 1}$$

### 3.2. Processamento na Camada de Saída (3ª Camada)
O neurônio de saída recebe contribuições simultâneas de duas fontes:
1. Das saídas dos neurônios da camada escondida $y^{(1)}$, ponderadas pelos pesos de $W_2$.
2. Das entradas originais da rede $X$, ponderadas diretamente pelos pesos de $W_3$.

Assim, o potencial de ativação induzido no neurônio de saída $u$ é dado por:
$$u = \sum_{i=1}^{N_1} w_{2, 1i} y_i^{(1)} + \sum_{j=1}^{N} w_{3, 1j} x_j$$
Em notação matricial:
$$u = W_2 y^{(1)} + W_3 X \in \mathbb{R}^{1 \times 1}$$

O sinal de saída final da rede é:
$$y = g_2(u) \in \mathbb{R}$$

### 3.3. Cálculo do Erro e Função Custo
O erro instantâneo para o padrão apresentado é:
$$e = d - y$$
A energia do erro instantâneo (função de custo de mínimos quadrados) é dada por:
$$E = \frac{1}{2} e^2 = \frac{1}{2} (d - y)^2$$

---

## 4. Passo Backward (Retropropagação e Dedução do Ajuste de Pesos)

Pelo método do gradiente descendente, a regra geral de atualização de qualquer peso sináptico $w$ com taxa de aprendizagem $\eta > 0$ é:
$$\Delta w = -\eta \frac{\partial E}{\partial w}, \qquad w^{(novo)} = w + \Delta w$$

Aplicando a **Regra da Cadeia** do Cálculo Diferencial:

### 4.1. Gradiente Local do Neurônio de Saída ($\delta_{saida}$)
Calcula-se a derivada de $E$ em relação ao potencial induzido $u$:
$$\frac{\partial E}{\partial u} = \frac{\partial E}{\partial e} \cdot \frac{\partial e}{\partial y} \cdot \frac{\partial y}{\partial u}$$

Calculando cada termo individualmente:
- $\frac{\partial E}{\partial e} = \frac{\partial}{\partial e} \left( \frac{1}{2} e^2 \right) = e$
- $\frac{\partial e}{\partial y} = \frac{\partial}{\partial y} (d - y) = -1$
- $\frac{\partial y}{\partial u} = g_2'(u)$

Substituindo os termos:
$$\frac{\partial E}{\partial u} = e \cdot (-1) \cdot g_2'(u) = -(d - y) g_2'(u)$$

Define-se o **gradiente local de saída ($\delta_{saida}$)** como o negativo dessa derivada:
$$\delta_{saida} \equiv -\frac{\partial E}{\partial u} = (d - y) g_2'(u) = e \cdot g_2'(u)$$

---

### 4.2. Ajuste da Matriz $W_2$ (Pesos entre a 2ª e a 3ª camada)
Para o peso $w_{2, 1i}$, que conecta a saída do neurônio oculto $y_i^{(1)}$ ao neurônio de saída:
$$\frac{\partial E}{\partial w_{2, 1i}} = \frac{\partial E}{\partial u} \cdot \frac{\partial u}{\partial w_{2, 1i}}$$

Como $u = \sum_{k=1}^{N_1} w_{2, 1k} y_k^{(1)} + \sum_{j=1}^{N} w_{3, 1j} x_j$, a derivada em relação a $w_{2, 1i}$ é:
$$\frac{\partial u}{\partial w_{2, 1i}} = y_i^{(1)}$$

Portanto:
$$\frac{\partial E}{\partial w_{2, 1i}} = -\delta_{saida} \cdot y_i^{(1)}$$

O termo de correção do peso é:
$$\Delta w_{2, 1i} = -\eta \frac{\partial E}{\partial w_{2, 1i}} = \eta \cdot \delta_{saida} \cdot y_i^{(1)}$$

**Formulação Matricial para $W_2$:**
$$\Delta W_2 = \eta \cdot \delta_{saida} \cdot (y^{(1)})^T$$
$$W_2^{(novo)} = W_2 + \Delta W_2$$
*(Dimensão: $1 \times 1 \cdot 1 \times N_1 = 1 \times N_1$)*

---

### 4.3. Ajuste da Matriz $W_3$ (Pesos diretos entre a 1ª e a 3ª camada)
Para o peso $w_{3, 1j}$, que conecta diretamente o sinal de entrada $x_j$ ao neurônio de saída:
$$\frac{\partial E}{\partial w_{3, 1j}} = \frac{\partial E}{\partial u} \cdot \frac{\partial u}{\partial w_{3, 1j}}$$

Como $u = \sum_{k=1}^{N_1} w_{2, 1k} y_k^{(1)} + \sum_{l=1}^{N} w_{3, 1l} x_l$, a derivada em relação a $w_{3, 1j}$ é:
$$\frac{\partial u}{\partial w_{3, 1j}} = x_j$$

Portanto:
$$\frac{\partial E}{\partial w_{3, 1j}} = -\delta_{saida} \cdot x_j$$

O termo de correção do peso é:
$$\Delta w_{3, 1j} = -\eta \frac{\partial E}{\partial w_{3, 1j}} = \eta \cdot \delta_{saida} \cdot x_j$$

**Formulação Matricial para $W_3$:**
$$\Delta W_3 = \eta \cdot \delta_{saida} \cdot X^T$$
$$W_3^{(novo)} = W_3 + \Delta W_3$$
*(Dimensão: $1 \times 1 \cdot 1 \times N = 1 \times N$)*

> **Observação Fundamental:** Devido à conexão direta, os pesos $W_3$ são atualizados utilizando diretamente o gradiente local da saída $\delta_{saida}$ multiplicado pelos sinais de entrada $x_j$, de forma análoga à regra delta de uma rede de camada única, sem necessidade de retropropagação intermediária.

---

### 4.4. Ajuste da Matriz $W_1$ (Pesos entre a 1ª e a 2ª camada)
Para os pesos $w_{1, ij}$, que conectam a entrada $x_j$ ao neurônio escondido $i$, aplica-se novamente a regra da cadeia:
$$\frac{\partial E}{\partial w_{1, ij}} = \frac{\partial E}{\partial u_i^{(1)}} \cdot \frac{\partial u_i^{(1)}}{\partial w_{1, ij}}$$

Como $u_i^{(1)} = \sum_{l=1}^{N} w_{1, il} x_l$, temos que:
$$\frac{\partial u_i^{(1)}}{\partial w_{1, ij}} = x_j$$

Resta determinar $\frac{\partial E}{\partial u_i^{(1)}}$. O potencial $u_i^{(1)}$ influencia a energia de erro $E$ através de sua saída $y_i^{(1)}$, que por sua vez afeta o potencial de saída $u$ através do peso $w_{2, 1i}$:
$$\frac{\partial E}{\partial u_i^{(1)}} = \frac{\partial E}{\partial u} \cdot \frac{\partial u}{\partial y_i^{(1)}} \cdot \frac{\partial y_i^{(1)}}{\partial u_i^{(1)}}$$

Sabendo que:
- $\frac{\partial E}{\partial u} = -\delta_{saida}$
- $\frac{\partial u}{\partial y_i^{(1)}} = w_{2, 1i}$
- $\frac{\partial y_i^{(1)}}{\partial u_i^{(1)}} = g_1'(u_i^{(1)})$

Obtém-se:
$$\frac{\partial E}{\partial u_i^{(1)}} = (-\delta_{saida}) \cdot w_{2, 1i} \cdot g_1'(u_i^{(1)})$$

Define-se o **gradiente local do neurônio escondido $i$ ($\delta_i^{(1)}$)** como:
$$\delta_i^{(1)} \equiv -\frac{\partial E}{\partial u_i^{(1)}} = \delta_{saida} \cdot w_{2, 1i} \cdot g_1'(u_i^{(1)})$$

Em termos vetoriais para toda a camada oculta:
$$\delta^{(1)} = \left( W_2^T \delta_{saida} \right) \odot g_1'(u^{(1)}) \in \mathbb{R}^{N_1 \times 1}$$
*(onde $\odot$ denota o produto elemento a elemento / Hadamard)*.

Assim:
$$\frac{\partial E}{\partial w_{1, ij}} = -\delta_i^{(1)} \cdot x_j$$

A regra de ajuste para cada elemento de $W_1$ é:
$$\Delta w_{1, ij} = -\eta \frac{\partial E}{\partial w_{1, ij}} = \eta \cdot \delta_i^{(1)} \cdot x_j$$

**Formulação Matricial para $W_1$:**
$$\Delta W_1 = \eta \cdot \delta^{(1)} \cdot X^T$$
$$W_1^{(novo)} = W_1 + \Delta W_1$$
*(Dimensão: $N_1 \times 1 \cdot 1 \times N = N_1 \times N$)*

---

## 5. Sequência Completa do Algoritmo Backpropagation

Apresenta-se a seguir o fluxo operacional sequencial detalhado para execução do treinamento da rede:

```
+--------------------------------------------------------------------------+
|                  FLUXO SEQUENCIAL DO BACKPROPAGATION                     |
+--------------------------------------------------------------------------+

               [ Passo 1: Inicialização dos Parâmetros e Pesos ]
                                     |
                                     v
+------------->[ Passo 2: Início da Época de Treinamento ]
|                                    |
|                                    v
|              [ Passo 3: Apresentar Padrão p (X(p), d(p)) ]
|                                    |
|                                    v
|              [ Passo 4: Forward Pass ]
|                4.1: u(1) = W1 * X
|                4.2: y(1) = g1(u(1))
|                4.3: u    = W2 * y(1) + W3 * X
|                4.4: y    = g2(u)
|                4.5: e    = d - y
|                                    |
|                                    v
|              [ Passo 5: Backward Pass ]
|                5.1: delta_saida = e * g2'(u)
|                5.2: delta(1)    = (W2^T * delta_saida) (x) g1'(u(1))
|                5.3: Ajuste W2: W2 = W2 + eta * delta_saida * (y(1))^T
|                5.4: Ajuste W3: W3 = W3 + eta * delta_saida * X^T
|                5.5: Ajuste W1: W1 = W1 + eta * delta(1) * X^T
|                                    |
|                                    v
|              [ Todos os P padrões foram apresentados? ]
|                     |                              |
|                    Não                            Sim
|                     |                              |
|                     +                              v
|                                        [ Passo 6: Cálculo do EQM ]
|                                          EQM = (1/P) * sum( 0.5 * e^2 )
|                                                    |
|                                                    v
|                                        [ Critério de Parada Atendido? ]
|                                          |                          |
|                                         Não                        Sim
+------------------------------------------+                          |
                                                                      v
                                                               [ Fim: Rede Treinada ]
```

### Detalhamento dos Passos:

1. **Passo 1: Inicialização**
   - Inicializar as matrizes $W_1$, $W_2$ e $W_3$ com números aleatórios pequenos (ex.: distribuídos uniformemente no intervalo $[-0.5, +0.5]$ ou $[0, 1]$).
   - Definir a taxa de aprendizagem $\eta > 0$ (ex.: $\eta = 0.01$ ou $\eta = 0.1$).
   - Definir a precisão desejada $\epsilon$ e o número máximo de épocas ($Épocas_{max}$).
   - Inicializar o contador de épocas: $época = 0$.

2. **Passo 2: Início do Ciclo de Época**
   - Reinicializar a variável de erro acumulado da época: $E_{acumulado} = 0$.

3. **Passo 3: Loop sobre os $P$ Padrões de Treinamento**
   - Para cada padrão $p = 1, 2, \dots, P$:
     - **3.1. Apresentação:** Injetar o vetor $X(p)$ na entrada e selecionar $d(p)$ como saída desejada.
     - **3.2. Propagação Direta (Forward):**
       - Calcular o potencial da camada escondida: $u^{(1)} = W_1 X(p)$.
       - Calcular o sinal de saída da camada escondida: $y^{(1)} = g_1(u^{(1)})$.
       - Calcular o potencial do neurônio de saída combinando a camada intermediária e a conexão direta:
         $$u = W_2 y^{(1)} + W_3 X(p)$$
       - Calcular a resposta final da rede: $y = g_2(u)$.
       - Calcular o erro instantâneo: $e = d(p) - y$.
       - Acumular a energia do erro: $E_{acumulado} \leftarrow E_{acumulado} + \frac{1}{2} e^2$.
     - **3.3. Retropropagação do Erro (Backward):**
       - Calcular o gradiente local de saída:
         $$\delta_{saida} = e \cdot g_2'(u)$$
       - Calcular o vetor de gradientes locais da camada escondida:
         $$\delta^{(1)} = \left( W_2^T \delta_{saida} \right) \odot g_1'(u^{(1)})$$
     - **3.4. Atualização das Matrizes de Pesos (Modo Padrão a Padrão / Estocástico):**
       - Atualizar $W_2$:
         $$W_2 \leftarrow W_2 + \eta \cdot \delta_{saida} \cdot (y^{(1)})^T$$
       - Atualizar $W_3$:
         $$W_3 \leftarrow W_3 + \eta \cdot \delta_{saida} \cdot (X(p))^T$$
       - Atualizar $W_1$:
         $$W_1 \leftarrow W_1 + \eta \cdot \delta^{(1)} \cdot (X(p))^T$$

4. **Passo 4: Cálculo do Erro Quadrático Médio (EQM)**
   - Ao final da apresentação dos $P$ padrões, calcular:
     $$EQM(época) = \frac{1}{P} E_{acumulado} = \frac{1}{P} \sum_{p=1}^{P} \frac{1}{2} (d(p) - y(p))^2$$

5. **Passo 5: Verificação do Critério de Parada**
   - Incrementar o contador: $época \leftarrow época + 1$.
   - Se $|EQM(época) - EQM(época - 1)| < \epsilon$ ou se $época \ge Épocas_{max}$, encerrar o algoritmo.
   - Caso contrário, retornar ao **Passo 2** para uma nova época.

---

## 6. Considerações Adicionais: Bias e Funções de Ativação

### 6.1. Incorporação do Bias (Limiar de Ativação)
Na prática da disciplina (conforme convenções adotadas no Perceptron e ADALINE com entrada fixa $x_0 = -1$):
- Para a camada escondida, inclui-se uma entrada fictícia $x_0 = -1$ e uma coluna de pesos de limiar em $W_1$.
- Para a camada de saída, inclui-se um limiar fixo $x_0 = -1$ (ou $y_0^{(1)} = -1$).
- Nesse caso, a dedução do ajuste do limiar segue rigorosamente a mesma regra de gradiente local:
  $$\Delta w_{bias} = \eta \cdot \delta \cdot (-1)$$

### 6.2. Caso da Função Sigmoide Logística
Se a função de ativação utilizada for a sigmoide padrão:
$$g(u) = \frac{1}{1 + e^{-u}}$$
Sua derivada possui a propriedade notável:
$$g'(u) = g(u) \cdot (1 - g(u)) = y \cdot (1 - y)$$

Substituindo nas equações de ajuste obtidas:
- **Gradiente de saída:**
  $$\delta_{saida} = (d - y) \cdot y \cdot (1 - y)$$
- **Gradiente escondido para cada neurônio $i$:**
  $$\delta_i^{(1)} = \delta_{saida} \cdot w_{2, 1i} \cdot y_i^{(1)} \cdot (1 - y_i^{(1)})$$

---

## 7. Conclusão

A demonstração comprova que a inclusão das conexões diretas através da matriz $W_3$ preserva a estrutura clássica do algoritmo Backpropagation:
1. A matriz $W_2$ é ajustada pela interação do gradiente da saída com as ativações da camada oculta.
2. A matriz $W_3$ é ajustada diretamente pelo produto do gradiente de saída com os sinais da camada de entrada.
3. A matriz $W_1$ é ajustada propagando-se o gradiente da saída exclusivamente através das conexões de $W_2$ (uma vez que $W_3$ não passa pela camada intermediária).
