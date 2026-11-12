# Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)
## Campus VIII – Varginha | Bacharelado em Sistemas de Informação
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 12/11/2026  
**Trabalho:** Rede de Kohonen (Mapa Auto-Organizável – SOM) Aplicada ao Agrupamento de Imperfeições na Fabricação de Pneus  

---

## Visão Geral do Problema e Arquitetura da Rede

No processo industrial de fabricação de pneus, o composto de borracha pode apresentar imperfeições que inviabilizam sua utilização. Foram coletadas **120 amostras de treinamento** (disponíveis no Apêndice e em [`data/treinamento.csv`](file:///home/alunos/Downloads/linn/kohonen/data/treinamento.csv)), cada uma descrita por três grandezas de processo $\{x_1, x_2, x_3\}$, além de **12 amostras de operação/teste** ([`data/teste.csv`](file:///home/alunos/Downloads/linn/kohonen/data/teste.csv)).

Para detectar similaridades intrínsecas e agrupar as amostras imperfeitas em classes, emprega-se uma **Rede de Kohonen (*Self-Organizing Map* – SOM)** com os seguintes parâmetros especificados no enunciado:
- **Entradas:** $\mathbf{x} = [x_1, x_2, x_3]^T \in \mathbb{R}^3$.
- **Número de neurônios competitivos:** $N_1 = 16$.
- **Topologia do Grid:** Bidimensional $4 \times 4$, numerados de $1$ a $16$:
  $$\begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix}$$
- **Taxa de aprendizado:** $\eta = 0.001$.
- **Raio de vizinhança topológica:** $R = 1$ (neurônios diretamente conectados por uma aresta no grid esquemático $4 \times 4$).
- **Critério de competição (*Best Matching Unit* – BMU):** Menor distância euclidiana entre a amostra $\mathbf{x}$ e o vetor de pesos $\mathbf{w}_j = [w_{j1}, w_{j2}, w_{j3}]^T$:
  $$j^* = \arg\min_{j \in \{1, \dots, 16\}} \|\mathbf{x} - \mathbf{w}_j\| = \arg\min_{j \in \{1, \dots, 16\}} \sqrt{\sum_{i=1}^{3} (x_i - w_{ji})^2}$$
- **Regra de adaptação sináptica (Norma Euclidiana):**
  - Para o neurônio vencedor $j^*$:
    $$\mathbf{w}_{j^*}(t+1) = \mathbf{w}_{j^*}(t) + \eta \left(\mathbf{x} - \mathbf{w}_{j^*}(t)\right)$$
  - Para os neurônios pertencentes à vizinhança topológica de raio $1$ ($k \in \mathcal{N}_1(j^*)$):
    $$\mathbf{w}_{k}(t+1) = \mathbf{w}_{k}(t) + \frac{\eta}{2} \left(\mathbf{x} - \mathbf{w}_{k}(t)\right)$$

A análise das 120 amostras de treinamento divide o conjunto em três classes com características físico-químicas bem definidas:
- **Classe A (amostras 1 a 20 – 20 amostras):** Centroide médio em $\bar{\mathbf{x}}_A = (0.2557,\; 0.2518,\; 0.1887)$.
- **Classe B (amostras 21 a 60 – 40 amostras):** Centroide médio em $\bar{\mathbf{x}}_B = (0.7493,\; 0.2521,\; 0.7517)$.
- **Classe C (amostras 61 a 120 – 60 amostras):** Centroide médio em $\bar{\mathbf{x}}_C = (0.5172,\; 0.7478,\; 0.5025)$.

---

## Questão 1 – Conjuntos de Neurônios no Grid Relativos às Classes A, B e C

> **Enunciado:** *Indique quem são os conjuntos de neurônios representados no grid que fornecem respostas relativas às classes A, B e C.*

Após a convergência do treinamento da Rede de Kohonen (1372 épocas até estabilização com $\max |\Delta \mathbf{w}_j| < 10^{-7}$ e erro de quantização médio final $QE = 0.0685$), os neurônios do grid $4 \times 4$ auto-organizaram-se em três regiões topologicamente conexas e sem qualquer mistura entre classes:

### 1.1. Resumo dos Conjuntos de Neurônios por Classe

| Classe | Faixa de Amostras (Treino) | Neurônios Vencedores Ativos (BMUs Diretos) | Neurônios de Fronteira / Transição | Conjunto Completo da Região no Grid $4 \times 4$ |
| :---: | :---: | :---: | :---: | :---: |
| **Classe A** | $1 \text{ a } 20$ ($20$ amostras) | **$\{4,\; 8\}$** | $\{7\}$ | **$\{4,\; 7,\; 8\}$** |
| **Classe B** | $21 \text{ a } 60$ ($40$ amostras) | **$\{11,\; 15,\; 16\}$** | $\{12,\; 14\}$ | **$\{11,\; 12,\; 14,\; 15,\; 16\}$** |
| **Classe C** | $61 \text{ a } 120$ ($60$ amostras) | **$\{1,\; 2,\; 5,\; 6,\; 9,\; 10,\; 13\}$** | $\{3\}$ | **$\{1,\; 2,\; 3,\; 5,\; 6,\; 9,\; 10,\; 13\}$** |

> **Observação sobre a proporção de neurônios no mapa:**  
> A distribuição de neurônios no grid $4 \times 4$ reflete fielmente a densidade amostral de cada classe no conjunto de treinamento:
> - **Classe A ($20/120 = 16.7\%$ dos dados):** representada por $2$ neurônios ativos ($\{4, 8\}$, com $10$ amostras cada) no canto superior/médio direito.
> - **Classe B ($40/120 = 33.3\%$ dos dados):** representada por $3$ neurônios ativos ($\{11, 15, 16\}$, com $3$, $19$ e $18$ amostras) no canto inferior direito.
> - **Classe C ($60/120 = 50.0\%$ dos dados):** representada por $7$ neurônios ativos ($\{1, 2, 5, 6, 9, 10, 13\}$, totalizando as $60$ amostras) nas colunas à esquerda e topo do mapa.
> - Os neurônios $\{3, 7, 12, 14\}$ atuam como **neurônios de fronteira (transição topológica)** entre os três agrupamentos, sendo classificados pela menor distância euclidiana aos respectivos centroides como pertencentes às regiões de $C$ ($\{3\}$), $A$ ($\{7\}$) e $B$ ($\{12, 14\}$).

### 1.2. Diagrama Esquemático do Grid Topológico $4 \times 4$ Pós-Treinamento

```text
   [ N01: Classe C ] ─── [ N02: Classe C ] ─── [ N03: Classe C*] ─── [ N04: Classe A ]
      (10 amostras)         (12 amostras)          (fronteira)          (10 amostras)
           │                     │                      │                     │
   [ N05: Classe C ] ─── [ N06: Classe C ] ─── [ N07: Classe A*] ─── [ N08: Classe A ]
       (9 amostras)          (5 amostras)          (fronteira)          (10 amostras)
           │                     │                      │                     │
   [ N09: Classe C ] ─── [ N10: Classe C ] ─── [ N11: Classe B ] ─── [ N12: Classe B*]
       (7 amostras)          (9 amostras)          (3 amostras)          (fronteira)
           │                     │                      │                     │
   [ N13: Classe C ] ─── [ N14: Classe B*] ─── [ N15: Classe B ] ─── [ N16: Classe B ]
       (8 amostras)          (fronteira)          (19 amostras)         (18 amostras)
```

### 1.3. Tabela Completa dos 16 Neurônios do Grid (Pesos Sinápticos Finais e Ativações)

| Neurônio | Posição $(L, C)$ | Vizinhos ($R=1$) | $w_{j1}$ | $w_{j2}$ | $w_{j3}$ | Amostras A | Amostras B | Amostras C | Papel no Mapa | Classe Associada |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $(1, 1)$ | $\{2, 5\}$ | 0.5129 | 0.7931 | 0.5368 | 0 | 0 | 10 | Ativo (BMU) | **C** |
| **2** | $(1, 2)$ | $\{1, 3, 6\}$ | 0.5369 | 0.7560 | 0.5555 | 0 | 0 | 12 | Ativo (BMU) | **C** |
| **3** | $(1, 3)$ | $\{2, 4, 7\}$ | 0.4188 | 0.5000 | 0.4136 | 0 | 0 | 0 | Fronteira | **C** |
| **4** | $(1, 4)$ | $\{3, 8\}$ | 0.2621 | 0.2397 | 0.2003 | 10 | 0 | 0 | Ativo (BMU) | **A** |
| **5** | $(2, 1)$ | $\{1, 6, 9\}$ | 0.5012 | 0.7976 | 0.4848 | 0 | 0 | 9 | Ativo (BMU) | **C** |
| **6** | $(2, 2)$ | $\{2, 5, 7, 10\}$ | 0.5335 | 0.7449 | 0.5045 | 0 | 0 | 5 | Ativo (BMU) | **C** |
| **7** | $(2, 3)$ | $\{3, 6, 8, 11\}$ | 0.4006 | 0.4219 | 0.3383 | 0 | 0 | 0 | Fronteira | **A** |
| **8** | $(2, 4)$ | $\{4, 7, 12\}$ | 0.2494 | 0.2640 | 0.1771 | 10 | 0 | 0 | Ativo (BMU) | **A** |
| **9** | $(3, 1)$ | $\{5, 10, 13\}$ | 0.5064 | 0.7374 | 0.4578 | 0 | 0 | 7 | Ativo (BMU) | **C** |
| **10** | $(3, 2)$ | $\{6, 9, 11, 14\}$ | 0.5663 | 0.6690 | 0.4771 | 0 | 0 | 9 | Ativo (BMU) | **C** |
| **11** | $(3, 3)$ | $\{7, 10, 12, 15\}$ | 0.6690 | 0.3608 | 0.6525 | 0 | 3 | 0 | Ativo (BMU) | **B** |
| **12** | $(3, 4)$ | $\{8, 11, 16\}$ | 0.6043 | 0.2843 | 0.5693 | 0 | 0 | 0 | Fronteira | **B** |
| **13** | $(4, 1)$ | $\{9, 14\}$ | 0.4748 | 0.7122 | 0.4785 | 0 | 0 | 8 | Ativo (BMU) | **C** |
| **14** | $(4, 2)$ | $\{10, 13, 15\}$ | 0.6197 | 0.4415 | 0.6138 | 0 | 0 | 0 | Fronteira | **B** |
| **15** | $(4, 3)$ | $\{11, 14, 16\}$ | 0.7391 | 0.2413 | 0.7467 | 0 | 19 | 0 | Ativo (BMU) | **B** |
| **16** | $(4, 4)$ | $\{12, 15\}$ | 0.7686 | 0.2550 | 0.7646 | 0 | 18 | 0 | Ativo (BMU) | **B** |

---

## Questão 2 – Classificação das Amostras de Teste

> **Enunciado:** *Para as amostras da tabela abaixo indique a que classes as mesmas pertencem.*

Cada uma das 12 amostras da tabela foi apresentada à Rede de Kohonen treinada, determinando-se o neurônio vencedor $j^* = \arg\min_j \|\mathbf{x} - \mathbf{w}_j\|$ e a respectiva classe associada:

| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio Vencedor ($j^*$) | Distância Euclidiana $\|\mathbf{x} - \mathbf{w}_{j^*}\|$ | **Classe** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.2471 | 0.1778 | 0.2905 | Neurônio **4** | 0.1104 | **A** |
| **2** | 0.8240 | 0.2223 | 0.7041 | Neurônio **16** | 0.0883 | **B** |
| **3** | 0.4960 | 0.7231 | 0.5866 | Neurônio **2** | 0.0610 | **C** |
| **4** | 0.2923 | 0.2041 | 0.2234 | Neurônio **4** | 0.0521 | **A** |
| **5** | 0.8118 | 0.2668 | 0.7484 | Neurônio **16** | 0.0477 | **B** |
| **6** | 0.4837 | 0.8200 | 0.4792 | Neurônio **5** | 0.0290 | **C** |
| **7** | 0.3248 | 0.2629 | 0.2375 | Neurônio **4** | 0.0765 | **A** |
| **8** | 0.7209 | 0.2116 | 0.7821 | Neurônio **15** | 0.0497 | **B** |
| **9** | 0.5259 | 0.6522 | 0.5957 | Neurônio **2** | 0.1118 | **C** |
| **10** | 0.2075 | 0.1669 | 0.1745 | Neurônio **4** | 0.0946 | **A** |
| **11** | 0.7830 | 0.3171 | 0.7888 | Neurônio **16** | 0.0682 | **B** |
| **12** | 0.5393 | 0.7510 | 0.5682 | Neurônio **2** | 0.0138 | **C** |

**Resumo da Classificação (Questão 2):**
- **Classe A:** Amostras **1, 4, 7 e 10** (todas ativam o neurônio 4 da região da Classe A).
- **Classe B:** Amostras **2, 5, 8 e 11** (ativam os neurônios 15 e 16 da região da Classe B).
- **Classe C:** Amostras **3, 6, 9 e 12** (ativam os neurônios 2 e 5 da região da Classe C).

---

## Questão 3 – Demonstração Matemática da Regra de Alteração de Pesos ("Norma Euclidiana")

> **Enunciado:** *Demonstrar que a regra de alteração de pesos "Norma Euclidiana" para um padrão $\mathbf{x}$ é obtida a partir da minimização da função erro quadrático:*
> $$E = \frac{1}{2} \sum_{i} (w_{ji} - x_i)^2$$
> *onde $j$ é o índice do neurônio vencedor.*

### Demonstração:

Seja $\mathbf{x} = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$ o vetor de entrada apresentado à rede e $\mathbf{w}_j = [w_{j1}, w_{j2}, \dots, w_{jn}]^T \in \mathbb{R}^n$ o vetor de pesos sinápticos associado ao neurônio vencedor $j$.

**Passo 1 – Definição da função custo (erro quadrático de quantização):**  
A metade do quadrado da distância euclidiana entre o vetor de pesos $\mathbf{w}_j$ e o padrão de entrada $\mathbf{x}$ define o funcional de energia/erro quadrático instantâneo:

$$E(\mathbf{w}_j) = \frac{1}{2} \|\mathbf{w}_j - \mathbf{x}\|^2 = \frac{1}{2} \sum_{i=1}^{n} (w_{ji} - x_i)^2$$

**Passo 2 – Aplicação do Método do Gradiente Descendente (*Steepest Descent*):**  
Para minimizar iterativamente a função erro quadrático $E(\mathbf{w}_j)$ em relação aos pesos sinápticos $w_{ji}$, o ajuste $\Delta w_{ji}$ deve ocorrer na direção oposta ao gradiente de $E$ em relação a $w_{ji}$, ponderado pela taxa de aprendizado $\eta > 0$:

$$\Delta w_{ji} = -\eta \frac{\partial E}{\partial w_{ji}}$$

**Passo 3 – Cálculo da derivada parcial $\frac{\partial E}{\partial w_{ji}}$:**  
Derivando a expressão de $E$ parcialmente em relação à componente específica $w_{ji}$:

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} \sum_{k=1}^{n} (w_{jk} - x_k)^2 \right] = \frac{1}{2} \sum_{k=1}^{n} \frac{\partial}{\partial w_{ji}} (w_{jk} - x_k)^2$$

Como os pesos sinápticos $w_{jk}$ são variáveis independentes entre si ($\frac{\partial w_{jk}}{\partial w_{ji}} = \delta_{ki}$, onde $\delta_{ki}$ é o delta de Kronecker) e as entradas $x_k$ são constantes para o padrão apresentado, todos os termos do somatório com $k \neq i$ possuem derivada nula. Resta apenas o termo $k = i$:

$$\frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \cdot 2 (w_{ji} - x_i)^{2-1} \cdot \frac{\partial (w_{ji} - x_i)}{\partial w_{ji}} = (w_{ji} - x_i) \cdot 1 = w_{ji} - x_i$$

**Passo 4 – Obtenção da variação dos pesos $\Delta w_{ji}$:**  
Substituindo o gradiente obtido no Passo 3 na equação do gradiente descendente do Passo 2:

$$\Delta w_{ji} = -\eta (w_{ji} - x_i) = \eta (x_i - w_{ji})$$

**Passo 5 – Equação de atualização sináptica discreta no tempo:**  
Somando o incremento $\Delta w_{ji}$ ao peso atual $w_{ji}(t)$ na iteração $t$, obtém-se a equação de atualização para cada sinapse $i \in \{1, \dots, n\}$ do neurônio vencedor $j$:

$$w_{ji}(t+1) = w_{ji}(t) + \Delta w_{ji} \implies \boxed{w_{ji}(t+1) = w_{ji}(t) + \eta \left(x_i - w_{ji}(t)\right)}$$

Ou, de forma equivalente em notação vetorial para o neurônio vencedor $j$:

$$\boxed{\mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \eta \left(\mathbf{x} - \mathbf{w}_j(t)\right)}$$

**Interpretação Geométrica:**  
O vetor diferença $(\mathbf{x} - \mathbf{w}_j(t))$ aponta exatamente da posição atual do vetor de pesos $\mathbf{w}_j(t)$ para o padrão de entrada $\mathbf{x}$. Multiplicar esse vetor por $\eta \in (0, 1)$ desloca o vetor de pesos do neurônio vencedor (e, por extensão cooperativa, de sua vizinhança topológica) na direção do padrão $\mathbf{x}$, reduzindo a distância euclidiana $\|\mathbf{w}_j - \mathbf{x}\|$. $\blacksquare$

---

## Arquivos e Recursos Gerados

- **Código-fonte da resolução:** [`main.py`](file:///home/alunos/Downloads/linn/kohonen/main.py)
- **Tabela de neurônios e mapeamento do Grid 4x4 (Questão 1):** [`tabela_item1_neuronios.csv`](file:///home/alunos/Downloads/linn/kohonen/tabela_item1_neuronios.csv)
- **Tabela de amostras de teste classificadas (Questão 2):** [`tabela_item2_teste.csv`](file:///home/alunos/Downloads/linn/kohonen/tabela_item2_teste.csv)
- **Gráfico do Mapa Topológico 4x4 e Curva de Convergência:** [`mapa_topologico_kohonen.png`](file:///home/alunos/Downloads/linn/kohonen/mapa_topologico_kohonen.png)
- **Gráfico Tridimensional das Amostras e Malha de Kohonen:** [`espaco_3d_kohonen.png`](file:///home/alunos/Downloads/linn/kohonen/espaco_3d_kohonen.png)

