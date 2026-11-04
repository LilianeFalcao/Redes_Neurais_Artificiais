# Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)
## Campus VIII – Varginha | Bacharelado em Sistemas de Informação
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 04/11/2026  
**Trabalho:** Rede Neural de Função de Base Radial (RBF) para Detecção de Radiação Nuclear  

---

## 1. Visão Geral do Problema e Topologia da Rede

O objetivo deste trabalho é projetar, treinar e validar uma rede neural de Função de Base Radial (**RBF - *Radial Basis Function***) para verificar a presença de radiação em compostos nucleares com base em duas variáveis de entrada ($x_1$ e $x_2$), representativas das concentrações de grandezas físico-químicas.

A base de dados é composta por **50 amostras** no total:
- **Conjunto de Treinamento:** 40 amostras (disponíveis no Apêndice e em `data/treinamento.csv`).
- **Conjunto de Teste:** 10 amostras (disponíveis no enunciado e em `data/teste.csv`).

A padronização das classes para a saída desejada $d$ é:
- **Presença de Radiação:** $d = +1$
- **Ausência de Radiação:** $d = -1$

### Topologia Adotada:
- **Camada de Entrada:** 2 variáveis ($x_1, x_2$).
- **Camada Oculta (Intermediária):** 2 neurônios com funções de ativação radiais gaussianas ($\phi_1, \phi_2$).
- **Camada de Saída:** 1 neurônio linear com bias $x_0 = -1$ e pesos $W2 = [W2_{1,0}, W2_{1,1}, W2_{1,2}]$.

---

## Atividade 1 – Treinamento da Camada Escondida (K-means)

Conforme a instrução do enunciado:
> *"Em se tratando de um problema de classificação de padrões, compute os centros dos dois clusters levando-se em consideração apenas aqueles padrões com presença de radiação."*

Dentre as 40 amostras de treinamento, exatamente **19 amostras** possuem presença de radiação ($d = +1$). O algoritmo **$k$-means** foi executado para agrupar esses 19 pontos em $k = 2$ clusters.

### Formulação Matemática:
1. **Centróides:**
   $$\mathbf{c}_j = \frac{1}{N_j} \sum_{\mathbf{x}_i \in S_j} \mathbf{x}_i, \quad j \in \{1, 2\}$$

2. **Variância do Cluster (Dispersão / Largura da Gaussiana):**
   - **Variância Populacional (MSD - *Mean Squared Distance*):**  
     $$\sigma_j^2 = \frac{1}{N_j} \sum_{\mathbf{x}_i \in S_j} \|\mathbf{x}_i - \mathbf{c}_j\|^2 = \operatorname{Var}(x_1) + \operatorname{Var}(x_2)$$
   - **Variância Amostral ($N-1$):**  
     $$s_j^2 = \frac{1}{N_j - 1} \sum_{\mathbf{x}_i \in S_j} \|\mathbf{x}_i - \mathbf{c}_j\|^2$$

O agrupamento convergiu de maneira estável e unívoca (100% de convergência para os mesmos grupos em repetidas execuções), particionando os dados em:
- **Cluster 1:** 6 amostras (região superior-esquerda, $x_2$ mais elevado).
- **Cluster 2:** 13 amostras (região central-inferior, $x_2$ mais baixo).

### Tabela Preenchida (Item 1):

| Cluster | Centro $(x_1, x_2)$ | Variância ($\sigma^2$) [MSD / Populacional] | Variância Amostral ($s^2$) | Nº Amostras |
| :---: | :---: | :---: | :---: | :---: |
| **1** | **(0.1648, 0.6121)** | **0.029806** | 0.035767 | 6 |
| **2** | **(0.3990, 0.1571)** | **0.038460** | 0.041665 | 13 |

---

## Atividade 2 – Treinamento da Camada de Saída (Regra Delta Generalizada)

Com os centros e as larguras definidos na camada escondida, a resposta de cada neurônio oculto para uma entrada $\mathbf{x} = [x_1, x_2]^T$ é calculada pela função de base radial gaussiana:

$$\phi_1(\mathbf{x}) = \exp\left( - \frac{\|\mathbf{x} - \mathbf{c}_1\|^2}{2\sigma_1^2} \right)$$

$$\phi_2(\mathbf{x}) = \exp\left( - \frac{\|\mathbf{x} - \mathbf{c}_2\|^2}{2\sigma_2^2} \right)$$

O neurônio da camada de saída combina linearmente essas ativações com o termo de bias ($\phi_0 = x_0 = -1$):

$$y = \sum_{j=0}^{2} W2_{1,j} \cdot \phi_j = -W2_{1,0} + W2_{1,1} \cdot \phi_1(\mathbf{x}) + W2_{1,2} \cdot \phi_2(\mathbf{x})$$

### Parâmetros de Treinamento:
- **Taxa de aprendizado:** $\eta = 0.01$
- **Precisão de parada:** $\varepsilon = 10^{-7}$ ($|EQM(t) - EQM(t-1)| \le \varepsilon$)
- **Modo de ajuste:** Padrão a padrão (modo online) utilizando a Regra Delta:
  $$\Delta W2_{1,j} = \eta \cdot (d_k - y_k) \cdot \phi_{k,j}$$

### Convergência:
- **Número de épocas até convergência:** **350 épocas**
- **Erro Quadrático Médio Final ($EQM$):** **0.234239**

### Tabela Preenchida (Item 2):

| Peso | Valor |
| :---: | :---: |
| **$W2_{1,0}$** (Bias) | **1.0026** |
| **$W2_{1,1}$** | **2.3780** |
| **$W2_{1,2}$** | **2.6977** |

> **Nota teórica sobre o Bias:**  
> Como a entrada de bias é padronizada como $\phi_0 = -1.0$, o produto $W2_{1,0} \cdot \phi_0 = 1.0026 \cdot (-1) = -1.0026$. Isso significa que quando um ponto está distante de ambos os centros de radiação ($\phi_1 \approx 0$ e $\phi_2 \approx 0$), o potencial gerado é $y \approx -1.0026 \approx -1$, correspondendo naturalmente à **Ausência de Radiação** ($d = -1$). Quando o composto se aproxima de um dos clusters, a gaussiana correspondente se ativa ($\phi \to 1$), somando um valor positivo ponderado ($+2.3780$ ou $+2.6977$) que eleva a saída acima de zero ($y > 0$), indicando **Presença de Radiação** ($d = +1$). Caso a convenção de entrada de bias adotada seja $\phi_0 = +1.0$, o peso correspondente converge simetricamente para $W2_{1,0} = -1.0026$.

---

## Atividade 3 – Pós-Processamento da Saída

Como a camada de saída fornece valores reais contínuos ($y \in \mathbb{R}$), para a tarefa de classificação de padrões implementa-se a função degrau bipolar (função sinal com limiar em zero) exclusivamente no pós-processamento:

$$y^{\text{pós}} = \begin{cases} +1, & \text{se } y \ge 0 \\ -1, & \text{se } y < 0 \end{cases}$$

Onde:
- $y^{\text{pós}} = +1 \implies$ **Presença de Radiação**.
- $y^{\text{pós}} = -1 \implies$ **Ausência de Radiação**.

---

## Atividade 4 – Validação no Conjunto de Teste

A rede RBF treinada foi avaliada nas 10 amostras do conjunto de teste fornecido.

### Tabela de Validação Preenchida (Item 4):

| Amostra | $x_1$ | $x_2$ | $d$ | $y$ (Real) | $y^{\text{pós}}$ (Inteiro) | Resultado |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.8705 | 0.9329 | **-1** | -1.0025 | **-1** | Correto |
| **2** | 0.0388 | 0.2703 | **1** | -0.3231 | **-1** | Incorreto |
| **3** | 0.8236 | 0.4458 | **-1** | -0.9140 | **-1** | Correto |
| **4** | 0.7075 | 0.1502 | **1** | -0.2201 | **-1** | Incorreto |
| **5** | 0.9587 | 0.8663 | **-1** | -1.0026 | **-1** | Correto |
| **6** | 0.6115 | 0.9365 | **-1** | -0.9878 | **-1** | Correto |
| **7** | 0.3534 | 0.3646 | **1** | +0.9665 | **1** | Correto |
| **8** | 0.3268 | 0.2766 | **1** | +1.3232 | **1** | Correto |
| **9** | 0.6129 | 0.4518 | **-1** | -0.4682 | **-1** | Correto |
| **10** | 0.9948 | 0.4962 | **-1** | -0.9966 | **-1** | Correto |

### Desempenho Global:
$$\text{Taxa de Acerto (\%)} = \frac{\text{Total de Acertos}}{\text{Total de Amostras de Teste}} \times 100 = \frac{8}{10} \times 100 = \mathbf{80.00\%}$$

### Análise dos Erros de Classificação:
As amostras que sofreram erro foram as de **número 2** e **número 4**:
- **Amostra 2** ($x_1 = 0.0388, x_2 = 0.2703$, classe $d = +1$):  
  Localiza-se no extremo inferior-esquerdo. A distância para ambos os centros faz com que $\phi_1 \approx 0.108$ e $\phi_2 \approx 0.157$, gerando $y = -0.3231 < 0$.
- **Amostra 4** ($x_1 = 0.7075, x_2 = 0.1502$, classe $d = +1$):  
  Localiza-se na borda direita inferior ($x_1$ elevado). A ativação da gaussiana decai para $\phi_2 \approx 0.290$, resultando em $y = -0.2201 < 0$.

Em ambos os casos, os padrões positivos estão localizados nas periferias da distribuição dos dados de treino de radiação, fora do raio de captura efetivo dos dois únicos neurônios radiais configurados.

---

## Atividade 5 – Estratégias para Aumentar a Taxa de Acerto da RBF

Para elevar a taxa de acerto no conjunto de teste para 90% ou 100%, diversas estratégias bem fundamentadas na teoria de Redes de Base Radial podem ser adotadas:

### 1. Alocar Centros para Ambas as Classes (Presença e Ausência de Radiação)
- **Problema atual:** O enunciado instruiu o agrupamento via $k$-means utilizando **apenas** os padrões com presença de radiação ($d = +1$). A rede, portanto, só possui protótipos representativos da classe positiva e tenta modelar a classe negativa apenas pelo termo de repouso do bias.
- **Solução:** Aplicar o $k$-means separadamente em ambas as classes (ou no conjunto completo) alocando, por exemplo, 2 centros para $d=+1$ e 2 centros para $d=-1$.  
- **Resultado empírico comprovado:** Com centros para ambas as classes, a taxa de acerto no conjunto de teste salta de **80% para 100%**, e a taxa de acerto no treinamento atinge **97.5%**.

### 2. Calibração da Dispersão / Fator de Escala da Largura ($\sigma$)
- **Problema atual:** As variâncias calculadas estritamente a partir dos pontos de cada cluster resultam em raios de influência estreitos ($\sigma_1 \approx 0.173, \sigma_2 \approx 0.196$). Com isso, pontos periféricos como as amostras 2 e 4 sofrem decaimento exponencial abrupto.
- **Solução:** Introduzir um fator de escala de sobreposição entre clusters (heurística típica de RBF: $\sigma = \frac{d_{\max}}{\sqrt{2K}}$ ou multiplicar a variância por um fator entre $1.5$ e $2.0$).  
- **Resultado empírico comprovado:** Ao utilizar um multiplicador de escala de $1.5\times$ nas variâncias ($\sigma_1^2 = 0.0447, \sigma_2^2 = 0.0577$), a taxa de acerto no teste sobe imediatamente para **90.00%** (a amostra 4 passa a ser classificada corretamente com $y > 0$).

### 3. Ajuste Supervisionado Total dos Parâmetros (Ajuste Fino por Gradiente)
- Em vez de manter os centros $\mathbf{c}_k$ e variâncias $\sigma_k^2$ estáticos após a etapa não supervisionada (k-means), aplicar retropropagação do erro em todas as camadas (treinamento híbrido seguido de *fine-tuning* supervisionado). Os gradientes em relação a $\mathbf{c}_k$ e $\sigma_k$ deformam e deslocam as gaussianas exatamente nas direções que minimizam a perda de classificação das amostras de borda.

### 4. Aumento do Número de Neurônios Ocultos ($K > 2$) com Validação Cruzada
- Representar a região com presença de radiação utilizando 3 ou 4 centros menores permite recobrir concavidades e caudas periféricas da distribuição sem perda de especificidade. A determinação do número ideal de centros pode ser realizada via critério de informação de Akaike (AIC), BIC ou validação cruzada (*k-fold*).

### 5. Utilização de Matriz de Covariância Completa (Gaussianas Elípticas)
- A função gaussiana padrão utiliza dispersão esférica/isotrópica ($\|\mathbf{x}-\mathbf{c}\|^2 / 2\sigma^2$). A distribuição real das concentrações $x_1$ e $x_2$ apresenta correlações anisotrópicas. Substituir por uma distância de Mahalanobis com matriz de covariância $\mathbf{\Sigma}_k$:
  $$\phi_k(\mathbf{x}) = \exp\left( -\frac{1}{2}(\mathbf{x} - \mathbf{c}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{x} - \mathbf{c}_k) \right)$$
  permite criar campos receptivos elípticos orientados ao longo do espalhamento natural dos dados químicos.

---

## 6. Recursos Gerados

Todos os códigos, tabelas e gráficos da resolução foram implementados e executados com sucesso:
- **Código-fonte completo:** [`main.py`](file:///home/alunos/Downloads/linn/RBF_1/main.py)
- **Tabela de validação exportada:** [`tabela_item4.csv`](file:///home/alunos/Downloads/linn/RBF_1/tabela_item4.csv)
- **Gráfico de superfície de decisão e curva de EQM:** [`grafico_rbf.png`](file:///home/alunos/Downloads/linn/RBF_1/grafico_rbf.png)

