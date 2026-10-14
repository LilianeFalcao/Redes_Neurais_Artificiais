# PMC3 – Respostas teóricas

## Item 7 – Variantes do Backpropagation

### a. Resilient-Propagation (RProp)

Proposto por Riedmiller e Braun (1993), o RProp usa **apenas o sinal** da derivada parcial do erro em relação a cada peso, e não a sua magnitude.

**Funcionamento**
- Cada peso *w_ij* tem a sua própria taxa de ajuste Δ_ij (adaptativa e individual).
- Se a derivada mantém o mesmo sinal entre duas iterações, Δ_ij aumenta (η⁺ ≈ 1.2), o que acelera a descida.
- Se o sinal se inverte (o mínimo foi ultrapassado), Δ_ij diminui (η⁻ ≈ 0.5) e a atualização anterior pode ser desfeita.
- O peso é atualizado por −sinal(∂E/∂w)·Δ_ij, com Δ limitado ao intervalo [Δ_min, Δ_max].

**Principais vantagens**
- Elimina o problema dos **gradientes muito pequenos** nas regiões saturadas da sigmoide, onde g'(u) ≈ 0, que deixa o backprop padrão muito lento.
- Não exige escolher uma taxa de aprendizado η global. Os parâmetros padrão funcionam bem na maioria dos problemas.
- Converge muito mais rápido que o backprop padrão e o backprop com momentum, com custo computacional e de memória baixos (apenas um Δ por peso).
- É robusto e simples de implementar. É indicado para treinamento em lote (*batch*).

### b. Levenberg-Marquardt (LM)

É um método de **segunda ordem**, que aproxima o método de Newton e combina o gradiente descendente com o método de Gauss-Newton.

**Funcionamento**
- Usa a matriz Jacobiana **J** das derivadas dos erros em relação aos pesos e aproxima a Hessiana por **H ≈ JᵀJ**.
- A atualização é dada por: **Δw = −(JᵀJ + μI)⁻¹ Jᵀe**
  - Com μ grande, o método se comporta como o gradiente descendente com passo pequeno, o que é seguro longe do mínimo.
  - Com μ pequeno, o método se comporta como Gauss-Newton, com convergência quadrática perto do mínimo.
- μ é ajustado a cada iteração: diminui quando o erro cai e aumenta quando o erro sobe.

**Principais vantagens**
- É, em geral, o algoritmo **mais rápido** para redes de pequeno e médio porte, convergindo em dezenas de épocas em vez de milhares.
- Atinge **erros finais menores** e tem ótimo desempenho em problemas de aproximação de funções e regressão, como a previsão de séries temporais deste trabalho.

**Desvantagens**
- Exige armazenar e inverter uma matriz de dimensão (nº de pesos)², o que torna o custo de memória e processamento muito alto para redes grandes.

### Comparação

| Característica | Backprop padrão | Backprop c/ momentum | RProp | Levenberg-Marquardt |
|---|---|---|---|---|
| Ordem | 1ª | 1ª | 1ª (só o sinal) | 2ª (aproximada) |
| Taxa de aprendizado | fixa, global | fixa + inércia | adaptativa por peso | controlada por μ |
| Velocidade | lenta | moderada | rápida | muito rápida |
| Memória | baixa | baixa | baixa | alta (JᵀJ) |
| Indicado para | didático | uso geral | redes de qualquer porte | redes pequenas e médias |
