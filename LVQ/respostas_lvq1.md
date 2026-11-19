# Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG)
## Campus VIII – Varginha | Bacharelado em Sistemas de Informação
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 19/11/2026  
**Trabalho:** Rede Neural LVQ-1 (*Learning Vector Quantization 1*) Aplicada à Classificação de Perfis de Demanda de Potência Elétrica  

---

## 1. Visão Geral do Problema e Arquitetura da Rede LVQ-1

Para o planejamento operacional de sistemas de energia elétrica (escalonamento de manutenção, expansão e análise de despacho), é possível classificar o perfil de demanda de potência de um dia inteiro observando apenas as medições das primeiras horas da manhã.

O conjunto de dados fornecido em [`LVQ1.pdf`](file:///home/alunos/Downloads/linn/LVQ/LVQ1.pdf) (estruturado em [`data/treinamento.csv`](file:///home/alunos/Downloads/linn/LVQ/data/treinamento.csv) e [`data/teste.csv`](file:///home/alunos/Downloads/linn/LVQ/data/teste.csv)) consiste em:
- **Conjunto de treinamento:** $16$ dias (amostras $1$ a $16$), distribuídos igualmente entre **$4$ classes (perfis) de demanda** ($4$ dias por classe).
- **Conjunto de teste:** $8$ dias (Dias $1$ a $8$) cujos perfis de demanda devem ser classificados pela rede treinada.
- **Vetor de entrada:** $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5, x_6]^T \in \mathbb{R}^6$, representando a potência medida às **7h, 8h, 9h, 10h, 11h e 12h**, respectivamente.

### Arquitetura e Algoritmo LVQ-1

A rede **LVQ-1 (*Learning Vector Quantization 1*)** é uma rede competitiva com treinamento supervisionado composta por:
1. **Camada de entrada:** $n = 6$ entradas correspondentes aos horários de medição (7h a 12h).
2. **Camada competitiva (vetores protótipos / *codebook*):** $m = 4$ neurônios protótipos $\mathbf{w}_1, \mathbf{w}_2, \mathbf{w}_3, \mathbf{w}_4 \in \mathbb{R}^6$, associados às Classes $1, 2, 3$ e $4$ ($C_j = j$).
3. **Camada de saída linear:** Mapeia o neurônio vencedor $j^*$ à sua respectiva classe $C_{j^*} \in \{1, 2, 3, 4\}$.

Os parâmetros e etapas do algoritmo implementado em [`main.py`](file:///home/alunos/Downloads/linn/LVQ/main.py) são:
- **Taxa de aprendizagem:** $\gamma = 0.05$ (conforme especificado no enunciado).
- **Inicialização dos vetores protótipos:** Inicialização canônica de LVQ-1 utilizando a primeira amostra de cada classe do conjunto de treinamento (Amostras $1$, $5$, $9$ e $13$ para as Classes $1$, $2$, $3$ e $4$, respectivamente), posicionando cada vetor de referência dentro da região representativa de sua classe e evitando divergência por repulsão.
- **Critério de competição (Distância Euclidiana):** Para cada vetor de entrada $\mathbf{x}^{(k)}$ com classe conhecida $T^{(k)}$, determina-se o neurônio vencedor $j^*$:
  $$j^* = \arg\min_{j \in \{1, 2, 3, 4\}} \|\mathbf{x}^{(k)} - \mathbf{w}_j\|_2 = \arg\min_{j \in \{1, 2, 3, 4\}} \sqrt{\sum_{i=1}^{6} \left(x_i^{(k)} - w_{ji}\right)^2}$$
- **Regra de adaptação sináptica (LVQ-1):**
  - Se o neurônio vencedor pertence à classe correta ($C_{j^*} = T^{(k)}$) — **atração**:
    $$\mathbf{w}_{j^*}(t+1) = \mathbf{w}_{j^*}(t) + \gamma \left(\mathbf{x}^{(k)} - \mathbf{w}_{j^*}(t)\right)$$
  - Se o neurônio vencedor pertence a uma classe incorreta ($C_{j^*} \neq T^{(k)}$) — **repulsão**:
    $$\mathbf{w}_{j^*}(t+1) = \mathbf{w}_{j^*}(t) - \gamma \left(\mathbf{x}^{(k)} - \mathbf{w}_{j^*}(t)\right)$$
  - Os demais neurônios ($j \neq j^*$) mantêm seus pesos inalterados.
- **Critério de parada:** Estabilização dos pesos com variação máxima absoluta por época $\max_{j, i} |w_{ji}(t) - w_{ji}(t-1)| < 10^{-6}$.

---

## 2. Resultados do Treinamento e Vetores Protótipos Aprendidos

Com taxa de aprendizagem $\gamma = 0.05$, a rede LVQ-1 convergiu em **$50$ épocas**:
- **Variação máxima final dos pesos ($\max |\Delta w_{ji}|$):** $8.57 \times 10^{-7}$
- **Erro de Quantização Médio ($QE$) no treinamento:** $0.1152$
- **Acurácia no conjunto de treinamento:** **$16/16$ ($100.0\%$)** — cada protótipo ativou exatamente as $4$ amostras de sua própria classe.

### 2.1. Pesos Sinápticos Finais (Protótipos $\mathbf{w}_c$)

| Neurônio | Classe | Amostra Inicial | $w_{j1}$ (7h) | $w_{j2}$ (8h) | $w_{j3}$ (9h) | $w_{j4}$ (10h) | $w_{j5}$ (11h) | $w_{j6}$ (12h) | $\|\mathbf{w}_c - \bar{\mathbf{x}}_c\|_2$ | Amostras Ativadas |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **1** | Amostra 1 | 2.3397 | 1.4859 | 1.9419 | 1.2469 | 2.3296 | 1.8133 | 0.0042 | 4 / 4 |
| **2** | **2** | Amostra 5 | 1.0635 | 0.1319 | 1.2480 | 5.3618 | 3.1513 | 2.3526 | 0.0032 | 4 / 4 |
| **3** | **3** | Amostra 9 | 1.4049 | 2.2791 | 1.0377 | 2.4234 | 1.7355 | 5.0966 | 0.0046 | 4 / 4 |
| **4** | **4** | Amostra 13 | 2.9496 | 1.4917 | 4.6638 | 1.3822 | 4.2506 | 6.8533 | 0.0039 | 4 / 4 |

> **Análise dos Perfis de Demanda Identificados:**
> - **Classe 1 (Demanda moderada e estável):** Potência relativamente equilibrada ao longo de toda a manhã, oscilando entre $1.25$ (10h) e $2.34$ (7h e 11h), sem picos elevados.
> - **Classe 2 (Vale acentuado às 8h e pico forte às 10h):** Queda expressiva às 8h ($0.13$) seguida de um pico marcante às 10h ($5.36$), reduzindo gradualmente até as 12h ($2.35$).
> - **Classe 3 (Crescimento acentuado ao meio-dia):** Valores moderados entre 7h e 11h (mínimo de $1.04$ às 9h) com salto acentuado às 12h ($5.10$).
> - **Classe 4 (Alta demanda com dois picos fortes às 9h e 12h):** Inicia elevada às 7h ($2.95$), apresenta pico intermediário às 9h ($4.66$), vale às 10h ($1.38$) e forte ascensão às 11h ($4.25$) e 12h ($6.85$).

### 2.2. Separação entre os Protótipos Finais

As distâncias euclidianas par a par entre os $4$ protótipos confirmam que os perfis de demanda são altamente distintos no espaço $\mathbb{R}^6$:

| Par de Protótipos | Distância Euclidiana $\|\mathbf{w}_a - \mathbf{w}_b\|_2$ |
| :---: | :---: |
| $d(\mathbf{w}_1, \mathbf{w}_2)$ | $4.6735$ |
| $d(\mathbf{w}_1, \mathbf{w}_3)$ | $3.8520$ |
| $d(\mathbf{w}_1, \mathbf{w}_4)$ | $6.0737$ |
| $d(\mathbf{w}_2, \mathbf{w}_3)$ | $4.7895$ |
| $d(\mathbf{w}_2, \mathbf{w}_4)$ | $7.3740$ |
| $d(\mathbf{w}_3, \mathbf{w}_4)$ | $5.1624$ |

---

## 3. Classificação dos Dados de Teste

Após o treinamento, cada um dos $8$ dias da tabela **Dados de Teste** foi apresentado à rede LVQ-1, atribuindo-se a classe do protótipo com menor distância euclidiana:

### 3.1. Tabela de Dados de Teste Preenchida

| Dia | 7 horas | 8 horas | 9 horas | 10 horas | 11 horas | 12 horas | **Classe** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.9817 | 1.5656 | 4.8391 | 1.4311 | 4.1916 | 6.9718 | **4** |
| **2** | 1.5537 | 2.2615 | 1.3169 | 2.5873 | 1.7570 | 5.0958 | **3** |
| **3** | 1.2240 | 0.2445 | 1.3595 | 5.4192 | 3.2027 | 2.5675 | **2** |
| **4** | 2.5828 | 1.5146 | 2.1119 | 1.2859 | 2.3414 | 1.8695 | **1** |
| **5** | 2.4168 | 1.4857 | 1.8959 | 1.3013 | 2.4500 | 1.7868 | **1** |
| **6** | 1.0604 | 0.2276 | 1.2806 | 5.4732 | 3.2133 | 2.4839 | **2** |
| **7** | 1.5246 | 2.4254 | 1.1353 | 2.5325 | 1.7569 | 5.2640 | **3** |
| **8** | 3.0565 | 1.6259 | 4.7743 | 1.3654 | 4.2904 | 6.9808 | **4** |

### 3.2. Detalhamento das Distâncias Euclidianas aos Protótipos LVQ-1

| Dia | $d(\mathbf{x}, \mathbf{w}_1)$ | $d(\mathbf{x}, \mathbf{w}_2)$ | $d(\mathbf{x}, \mathbf{w}_3)$ | $d(\mathbf{x}, \mathbf{w}_4)$ | Distância Mínima | Margem para 2º Mais Próximo | **Classe Predita** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 6.2388 | 7.5167 | 5.2896 | **0.2391** | 0.2391 | 5.0505 | **Classe 4** |
| **2** | 3.8091 | 4.6848 | **0.3574** | 4.9499 | 0.3574 | 3.4517 | **Classe 3** |
| **3** | 4.6759 | **0.3209** | 4.6690 | 7.1564 | 0.3209 | 4.3481 | **Classe 2** |
| **4** | **0.3060** | 4.7401 | 3.8989 | 5.9279 | 0.3060 | 3.5928 | **Classe 1** |
| **5** | **0.1620** | 4.6242 | 3.8877 | 6.0715 | 0.1620 | 3.7258 | **Classe 1** |
| **6** | 4.7696 | **0.2091** | 4.7642 | 7.3155 | 0.2091 | 4.5551 | **Classe 2** |
| **7** | 4.0107 | 4.8899 | **0.2927** | 5.0418 | 0.2927 | 3.7180 | **Classe 3** |
| **8** | 6.2544 | 7.5715 | 5.3211 | **0.2445** | 0.2445 | 5.0766 | **Classe 4** |

**Resumo da Classificação por Perfil:**
- **Classe 1:** Dias **4 e 5**
- **Classe 2:** Dias **3 e 6**
- **Classe 3:** Dias **2 e 7**
- **Classe 4:** Dias **1 e 8**

Todas as amostras de teste apresentaram distância entre $0.1620$ e $0.3574$ em relação ao protótipo vencedor e margem mínima de $3.4517$ em relação à segunda classe mais próxima, indicando classificação inequívoca.

---

## 4. Verificação de Robustez

Para confirmar a estabilidade da solução com $\gamma_0 = 0.05$, foram testadas duas variantes adicionais do algoritmo LVQ-1:

| Configuração | Épocas | $QE$ (Treino) | Acurácia (Treino) | Classificação dos Dias $1 \dots 8$ |
| :--- | :---: | :---: | :---: | :---: |
| **LVQ-1 padrão ($\gamma = 0.05$ fixo, $4$ protótipos)** | 50 | 0.1152 | 100.0% | `[4, 3, 2, 1, 1, 2, 3, 4]` |
| LVQ-1 com decaimento linear ($\gamma_0 = 0.05 \to 0$, $4$ protótipos) | 200 | 0.1154 | 100.0% | `[4, 3, 2, 1, 1, 2, 3, 4]` |
| LVQ-1 estendida ($\gamma = 0.05$ fixo, $8$ protótipos / $2$ por classe) | 90 | 0.0712 | 100.0% | `[4, 3, 2, 1, 1, 2, 3, 4]` |

Todas as configurações atingiram $100\%$ de acurácia no treinamento e produziram exatamente a mesma classificação para os $8$ dias de teste.

---

## 5. Arquivos e Recursos Gerados

- **Código-fonte da implementação LVQ-1:** [`main.py`](file:///home/alunos/Downloads/linn/LVQ/main.py)
- **Tabela dos vetores protótipos finais:** [`tabela_prototipos_lvq1.csv`](file:///home/alunos/Downloads/linn/LVQ/tabela_prototipos_lvq1.csv)
- **Tabela de classificação dos dados de teste:** [`tabela_teste_lvq1.csv`](file:///home/alunos/Downloads/linn/LVQ/tabela_teste_lvq1.csv)
- **Gráfico dos perfis de potência por classe (treino, protótipos e teste):** [`perfis_potencia_lvq1.png`](file:///home/alunos/Downloads/linn/LVQ/perfis_potencia_lvq1.png)
- **Gráfico de convergência do treinamento e matriz de distâncias do teste:** [`convergencia_e_distancias_lvq1.png`](file:///home/alunos/Downloads/linn/LVQ/convergencia_e_distancias_lvq1.png)

