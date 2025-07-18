# Aula 3

## **1 - Revisando o Aprendizado Supervisionado:**

No aprendizado supervisionado, nós temos conhecimento dos rótulos de cada instância existente:

$X = \[x\_1, x\_2, ..., x\_n] \~ ϵ \~ \mathbb{R}^n$

$Y = \{{y\_1, y\_2, ..., y\_m\}}, m >= 2$

* $X$ é o espaço de características formado por $n$ atributos distintos;
* $Y$ é conjunto de rótulos formado por $m$ categorias distintas;
* Cada instância $x$ possui $n$ valores e um $y\_k$ associado.

Assim, no aprendizado supervisionado, queremos encontrar uma função $f()$ que consiga mapear $x$ em uma saída $y$, ou seja, $y = f(x)$.

$$y_i = f(X_i, \theta) + \epsilon_i$$

* $\theta$: representa o conjunto de parâmetros a serem aprendidos (são alterados conforme o aprendizado);
* $\epsilon\_i$: representa o erro alcançado à instância $i$ (valor que não pode ser capturado pelo modelo)

#### **- Tarefas de Classificação:**

Dado um conjunto de observações $D = \{{X, Y\}}$, temos uma função $f()$ que mapeia uma entrada $X\_i$ (conjunto de atributos) à sua respectiva saída $Y\_i$. Assim, $f(x)$ aprende a aproximação que permite estimar $y$ (classes pré-definidas) para os valores de $x$.

* A função $f()$ é um classificador que fornece probabilidades (score) dos dados de entrada nas possíveis saídas (classes);
* $y$ será um valor inteiro pertencente à $Y$;
* A métrica mais utilizada é a acurácia (taxa de acerto);
* Exemplos: {benigno, maligno}, {cachorro, gato}, {bom pagador, mal pagador}.

**Em tarefas de classificação queremos predizer variáveis qualitativas.**

```python
import numpy as np
import matplotlib.pyplot as plt

# Conjuntos de dados e modelos
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

# Carregando o conjunto de dados Iris
iris = load_iris()
X = iris.data       # Atributos (características das flores)
Y = iris.target     # Rótulos (classes das espécies)

print("Tamanho do conjunto de dados (X):", X.shape)
print("Tamanho do conjunto de rótulos (Y):", Y.shape)
```

```
Tamanho do conjunto de dados (X): (150, 4)
Tamanho do conjunto de rótulos (Y): (150,)
```

## **2 - Métricas de avaliação para Tarefas de Classificação:**

**Matriz de Confusão em uma classificação binária:**

* A diagonal principal representa o que predito corretamente.

<figure><img src=".gitbook/assets/image (14).png" alt=""><figcaption><p>Figura 1 -</p></figcaption></figure>

**Taxa de Falso Positivo (Custo):** $$TFP = \frac{FP}{FP+VN} = 1 - TVN$$

**Taxa de Falso Negativo:** $$TFN = \frac{FN}{VP+FN} = 1 - TVP$$

***

**Taxa de Verdadeiro Positivo (Benefício/Sensibilidade/Revocação):**

* Dos pacientes doentes, quantos foram corretamente identificados?
* É focada na taxa de erros por falso negativo.

$$TVP = \frac{VP}{VP+FN}$$

**Taxa de Verdadeiro Negativo (Especificidade):**

* Dos pacientes sadios, quantos foram corretamente identificados?

$$TVN = \frac{VN}{VN+FP}$$

**Valor Preditivo Positivo (Precision):**

* Dos pacientes classificados como doentes, quantos foram corretamente identificados?
* É focada na taxa de erros por falso positivo.

$$VPP = \frac{VP}{VP + FP}$$

**Valor Preditivo Negativo:**

* Dos pacientes classificados como sadios, quantos foram corretamente identificados?

$$VPN = \frac{VN}{VN + FN}$$

***

**Acurácia:**

* Representa o percentual de quantos foram corretamente identificados, independentemente da classe;
* É a métrica mais utilizada, mas pode enganar na sua análise;

$$Acc = \frac{VP+VN}{VP+VN+FP+FN}$$

* Em conjunto totalmente desbalanceado, o modelo pode acertar tudo da classe majoritária e errar tudo da classe minoritária, obtendo uma performance elevada;
* Adicionalmente, os erros possuem o mesmo peso.

**Acurácia Balanceada:**

$$Acc_{Bal} = \frac{TVN+TVP}{2}$$

**F1-Score:**

* Considera a precisão e a revocação;
* Se uma das duas métricas for próxima a zero, a performance será baixa.

$$F1 = \frac{2}{\frac{1}{VPP}+\frac{1}{TVP}}$$

## **3 - KNN:**

**KNN** é um dos classificadores mais simples de ser implementado, de fácil compreensão e com boa performance.

Este algoritmo pertence a um grupo de técnicas denominada de _**Instance-based learning**_, que considera que o aprendizado é baseado nas instâncias fornecidas (qualquer exemplo pertencente a um conjunto de dados).

**Conceitualmente:**

* Todas as instâncias correspondem a pontos em um espaço n-dimensional;
* A ideia é encontrar os $k$ vizinhos ($k >= 1$) mais próximos da instância-objeto;
* A instância-objeto recebe o rótulo mais relevante de seus vizinhos.

**Composição:**

* Função de distância: mensura a disparidade entre os atributos de duas instâncias;
* Regra de classificação: atua na definição do rótulo final.

#### **- Função de Distância:**

Quanto menor a distância, maior a similaridade entre os exemplos.

**Algumas propriedades sobre Medida de Dissimilaridade:**

* **Não negativa:** $d(p, q) >= 0$ para todo $p$ e $q$ e $d(p, q) = 0$ se, e somente se, $p = q$;
* **Reflexiva:** $d(p,q) = 0$ se, e somente se, $p = q$;
* **Simétrica:** $d(p,q) = d(q,p)$ para todo $p$ e $q$;
* **Inequalidade triangular:** $d(p,r) <= d(p,q)+d(q,r)$ para todo $p$, $q$, e $r$.

**Distância Euclidiana:** $\[0, \infty]$

$$D_E(p,q) = \sqrt(\sum_{i=1}^{n} (p_i-q_i)^2)$$

**Distância Minkowski:** $\[0, \infty]$

$$D_M(p,q) = (\sum_{i=1}^{n} |p_i-q_i|^r)^{1/r}$$

* Se $r = 2$, temos a Distância Euclidiana  \
  -Se $r = 1$, temos a Distância Manhattan

**Distância Chebyshev:** $\[0, \infty]$

$$D_C(p,q) = max (|p_i-q_i|)$$

**Com atributos nominais:**

* Se $d(p,q) = 0$, se $p$ e $q$ são iguais
* Se $d(p,q) = 1$, se $p$ e $q$ são diferentes

#### **- Regras de Classificação:**

Se $k = 1$, a regra de classificação escolhida é irrelevante, pois se atribui à instância-objeto a classe pertencente ao vizinho mais próximo.

**Maioria na votação (Moda):**

* Cada elemento tem uma influência igual e a classe assinalada é aquela que possui mais representantes entre os vizinhos selecionados;
* É necessário haver uma regra adicional para decidir os empates:
  * Pode-se atribuir o rótulo da instância mais próxima dentre as classes empatadas;
  * Pode-se utilizar valores ímpares para $k$ (reduz a chance de empates, mas ainda pode acontecer).

**Peso pela distância:**

* Cada vizinho tem um peso inversamente proporcional à sua distância para a instância objeto.

$$W_i = \frac{1}{d(x_p,x_i)^2}$$

#### **- A importância de escolher o valor k:**

* Se $k$ é muito pequeno, o modelo se ajusta muito ao conjunto, causando overfitting;
* Se $k$ é muito grande, o modelo não se ajusta, causando underfitting.
* Se $k = n$, sendo $n$ a quantidade de instâncias no conjunto de treinamento, a classe majoritária será sempre a saída.

**Como escolher o valor ideal para k?**

* Diferentes valores podem ser considerados ótimos, dependendo da base de dados;
* Recomendável utilizar números ímpares ou primos para evitar empates;
* Otimização por GridSearch para encontrar o melhor valor;
* Heurísticas mais complexas.

<figure><img src=".gitbook/assets/image (15).png" alt=""><figcaption><p>Figura 2 -</p></figcaption></figure>

```python
# Avaliação do K-NN com diferentes valores de k (n_neighbors)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris

# Dados Iris
iris = load_iris()
X, Y = iris.data, iris.target

# Validação cruzada com 5 divisões
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lista de valores de k a testar
valores_k = [1, 3, 5, 11]

for k in valores_k:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=kf)
    print(f"Acurácia com {k:2d} K-NN: {scores.mean():.4f} +/- {scores.std():.4f}")

```

```
Acurácia com  1 K-NN: 0.9600 +/- 0.0327
Acurácia com  3 K-NN: 0.9667 +/- 0.0211
Acurácia com  5 K-NN: 0.9733 +/- 0.0249
Acurácia com 11 K-NN: 0.9733 +/- 0.0133
```

## **4 - Métodos Probabilísticos:**

Em aprendizado de máquina nós estamos interessados em determinar a melhor hipótese (modelo) para um conjunto de dados de treinamento. Então, **uma maneira natural para encontrar a melhor hipótese é determinar a hipótese mais provável**. Assim, o Teorema de Bayes provê o cálculo das probabilidades das hipóteses para os dados observados:

* $P(h)$ denota a probabilidade de $h$ antes de observar os dados de treinamento (probabilidade a priori, baseando-se no conhecimento que temos sobre $h$ ser válido ou não);
* se não temos este conhecimento a priori atribuimos a mesma probabilidade para todas as hipóteses candidatas;
* $P(D)$ é a probabilidade de observar o conjunto de treinamento $D$;
* $P(D|h)$ é a probabilidade de observar os dados $D$ na hipótese $h$;
* $P(h|D)$ é a probabilidade posterior da hipótese $h$ após conhecer $D$.

Assim, o **Teorema de Bayes** é dado por:

$$P(h|D) = \frac {P(D|h) P(h)}{P(D)}$$

Assim, **informalmente**:

$$posterior = \frac{probabilidade * anterior}{evidência}$$

**Generalizando para $n$ temos:**

$$P(D) = \sum_{i=1}^{n} P(D|h_i).P(h_i)$$

Inicialmente, temos diversas hipóteses $h$ candidatas para o mesmo conjunto de dados $D$. Assim, queremos maximizar a hipótese mais provável (_Maximum a posteriori_)

$$h_{MAP} = argmax ~~P(h|D) = argmax ~~\frac{P(D|h) P(h)}{P(D)} = argmax~~ P(D|h)$$

devido a $P(D)$ ser constante para todas as hipóteses e por assumir que $P(h\_i)$ são equiprováveis.

**Exemplo:**

* $\omega\_1$ é a hipótese de um paciente estar doente;
* $\omega\_2$ é a hipótese de um paciente não estar doente;
* $x$ é um atributo qualquer da base de dados $D$;
* Supondo que o valor de $x$ para uma instância seja 10, esta instância possui maior probabilidade de pertencer à hipótese $\omega\_1$ do que à hipótese $\omega\_2$.
* $\theta\_a$ estabelece um limiar para a separação das classes.

<figure><img src=".gitbook/assets/image (16).png" alt=""><figcaption><p>Figura 3 -</p></figcaption></figure>

Ou seja, temos a **Regra de Decisão Bayesiana** para definir a melhor hipótese:

$$\omega_1 ~~if~~ P(\omega_1|x) > P(\omega_2|x); ~~caso~~contrário~~ \omega_2$$

Consequentemente, garantimos o **menor erro** e maximizamos a hipótese mais provável:

$$P(erro|x) = min [P(\omega_1|x),P(\omega_2|x)]$$

## **4.1 - Naive Bayes:**

O classificador Naive-Bayes considera que cada instância $x$ é formado por um conjunto de atributos $a$ e que a função alvo $f(x)$ assume uma das respectivas hipóteses $h\_i$ das classes disponíveis.

Assim, para uma instância $x$ é atribuído a hipótese mais provável como rótulo de saída, em que:

$$h_{MAP} = argmax P(h_i|a_1,...,a_n)$$

Por meio do Teorema de Bayes obtemos:

$$h_{MAP} = argmax P(a_1,...,a_n|h_i)~P(h_i)$$

De forma geral:

* $P(h\_i)$ é a frequência de cada hipótese ocorrida no conjunto de treinamento
* $P(a\_1,...,a\_n|h\_i)$ muito complicado de obter pois não temos base para saber qual a probabilidade do conjunto de valores do atributos ocorrer para cada $h\_i$.

Devido à essa dificuldade, assume-se que os atributos são condicionalmente independentes de $h\_i$ (abordagem ingênua!!!). Consequentemente:

$$h_{NB} = argmax P(h_i) \prod P(a_j|h_i)$$

Assim, para cada hipótese $h$ calculamos apenas a probabilidade de cada atributo $a$. Isso torna o modelo menos complexo. Consequentemente, se os dados realmente forem condicionalmente independentes $h\_{NB} = h\_{MAP}$.

Outra característica é que não há necessidade de estimar os parâmetros, apenas a contagem de frequência para estabelecer as probabilidades.

**Algoritmo (dados de treinamento como entrada):**

* Para cada classe $C\_j$:
  * Obter a probabilidade incondicional $P(C\_j)$
  * Para cada atributo $A\_i$:
    * Obter a probabilidade estimada $P(A\_i|C\_j)$

A saída do modelo proporciona:

* A probabilidade de cada classe no conjunto de dados de treino;
* A probabilidade condicional de cada atributo dada a classe;

**Exemplo:**

Dados de Treinamento:

| Dia | Outlook  | Temperature | Humidity | Wind   | PlayTennis? |
| --- | -------- | ----------- | -------- | ------ | ----------- |
| D1  | Sunny    | Hot         | High     | Weak   | No          |
| D2  | Sunny    | Hot         | High     | Strong | No          |
| D3  | Overcast | Hot         | High     | Weak   | Yes         |
| D4  | Rain     | Mild        | High     | Weak   | Yes         |
| D5  | Rain     | Cool        | Normal   | Weak   | Yes         |
| D6  | Rain     | Cool        | Normal   | Strong | No          |
| D7  | Overcast | Cool        | Normal   | Strong | Yes         |
| D8  | Sunny    | Mild        | High     | Weak   | No          |
| D9  | Sunny    | Cool        | Normal   | Weak   | Yes         |
| D10 | Rain     | Mild        | Normal   | Weak   | Yes         |
| D11 | Sunny    | Mild        | Normal   | Strong | Yes         |
| D12 | Overcast | Mild        | High     | Strong | Yes         |
| D13 | Overcast | Hot         | Normal   | Weak   | Yes         |
| D14 | Rain     | Mild        | High     | Strong | No          |

Definição do Classificador Naive-Bayes:

$$h_{NB}= argmax ~P(h_i) \prod P(a_j|h_i)$$

Classificar (PlayTennis) a seguinte instância (Sunny, Cool, High, Strong):

$= P(Yes)\~P(Sunny|Yes)\~P(Cool|Yes)\~P(High|Yes)\~P(Strong|Yes) = \frac{9}{14} . \frac{2}{9} . \frac{3}{9} . \frac{3}{9} . \frac{3}{9} = 0.0053$

$= P(No)\~\~P(Sunny|No)\~\~\~P(Cool|No)\~\~\~P(High|No)\~\~\~P(Strong|No) = \frac{5}{14} . \frac{3}{5} . \frac{1}{5} . \frac{2}{5} . \frac{3}{5} = 0.0206$

$h\_{NB}= argmax {0.0053, 0.0206}$, ou seja, a saída é No

Como estamos utilizando frequência absoluta o somátorio não é 1. Se usarmos frequência relativa a soma passará a ser 1.

$\frac{0.0206}{0.0206+0.0053} = 0.795$ para No

$\frac{0.0053}{0.0206+0.0053} = 0.205$ para Yes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Classificador Naive-Bayes Gaussiano
nb = GaussianNB()
scores = cross_val_score(nb, X, Y, cv=kf)
print(f"Acurácia com Naive-Bayes: {scores.mean():.4f} +/- {scores.std():.4f}")

```

```
Acurácia com Naive-Bayes: 0.9600 +/- 0.0249
```

## **4.2 - Regressão Logística:**

É um modelo baseado em probabilidades que utiliza funções lineares para predizer $K$ classes, enquanto garante que a soma dessas probabilidades é 1 e todas estão no intervalo $\[0,1]$. Consequentemente, estamos estimando a probabilidade de uma variável (saída) assumir um determinado valor em função dos valores das outras variáveis:

$$ln(\frac{p}{1-p}) = g(x)$$

Aplicando antilogaritmo:

$$\frac{p}{1-p} = e^{g(x)}$$

Isolando $p$:

$$p = \frac{1}{1 + e^{-g(x)}}$$

em que $g(x) = \beta\_0 + \beta\_1 x\_1 + ... + \beta\_p x\_t$ e $t$ é a quantidade de variáveis.

De forma geral, **queremos estimar $\beta$ de forma a maximizar a probabilidade** da amostra ter sido observada. Assim:

* Quando $g(x) → +\infty$, $p → 1$
* Quando $g(x) → -\infty$, $p → 0$

Para otimizar a função queremos maximizar $l(\beta$) por meio do **Método da Máxima Verossimilhança**:

$$l(\beta) = \frac{1}{n}\sum_{i=1}^n [y_i~ln(p_i) + (1 - y_i)~ln(1 - p_i)]$$

em que:

* Se $y\_i = 1$ maximizamos $ln(p\_i)$
* Se $y\_i = 0$ maximizamos $ln(1 - p\_i)$

Assim, o **modelo final será definido** por ($p'\_i$ estimador de $p$):

$$\sum_{i=1}^n y_i - \sum_{i=1}^n p'_i = 0 ~~para~~ \beta_0$$

$$\sum_{i=1}^n x_{ij}y_i - \sum_{i=1}^n x_{ij}p'_i = 0 ~~para~~\beta_j$$

Com os parâmetros definidos, a **inferência é obtida por meio de um limiar para determinar a classe resultante**:

* Se $p > 0.5$ então $Y = 1$
* Se $p < 0.5$ então $Y = 0$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

# Regressão Logística com validação cruzada
rlog = LogisticRegression(max_iter=1000)
scores = cross_val_score(rlog, X, Y, cv=kf)
print(f"Acurácia com Regressão Logística: {scores.mean():.4f} +/- {scores.std():.4f}")

# Avaliação pontual com separação treino/teste
xTrainIris, xTestIris, yTrainIris, yTestIris = train_test_split(X, Y, test_size=0.2, shuffle=True)
rlog = LogisticRegression(max_iter=300).fit(xTrainIris, yTrainIris)
predIris = rlog.predict(xTestIris)
probaIris = rlog.predict_proba(xTestIris)

print("\nClasse real:", yTestIris[0])
print("Classe predita:", predIris[0])
print("Probabilidades preditas:", probaIris[0])

```

```
Acurácia com Regressão Logística: 0.9733 +/- 0.0249

Classe real: 2
Classe predita: 2
Probabilidades preditas: [7.12100922e-06 3.91879312e-02 9.60804948e-01]
```

## **5 - Máquina de Vetor de Suporte (SVM):**

O funcionamento de uma SVM pode ser descrito da seguinte forma:

* Dadas duas classes e um conjunto de pontos que pertencem a essas classes, **determina-se o hiperplano** que separa os pontos de forma a colocar o maior número de pontos de uma mesma classe em um mesmo lado, enquanto maximiza a distância de cada classe a esse hiperplano.
* A distância de uma classe a um hiperplano é a menor distância entre ele e os pontos dessa classe, sendo chamada de **margem de separação**.
* O hiperplano gerado pela SVM é determinado por um subconjunto dos pontos das duas classes, chamados de **vetores de suporte**.

**Exemplo:**

* Considere um conjunto de dados cujos pontos pertencem a duas classes distintas (quadrados amarelos e círculos vermelhos) em apenas duas dimensões ($X\_1$ e $X\_2$).
* Inicialmente, podemos traçar **diversos hiperplanos** que possibilitem a separação das classes. Entretanto, SVM se propõe a encontrar o **hiperplano ótimo** que possibilite a maximização da distância entre as classes. Assim, os pontos contidos nas margens são chamados de vetores de suporte.
* **O objetivo é encontrar o hiperplano separador entre as duas classes.** Ao encontrar o hiperplano separador, baseando-se no conjunto de treinamento, o modelo pode ser generalizado e dados não vistos podem ser classificados.
* Infinitos hiperplanos podem ser selecionados. Contudo, o que possibilitará **melhor generalização é aquele que maximiza a distância entre as classes** (linha verde tracejada).

<figure><img src=".gitbook/assets/image (17).png" alt=""><figcaption><p>Figura 4 -</p></figcaption></figure>

## **5.1 - Máquinas de Vetor de Suporte lineares:**

Constroem um conjunto de hiperplanos cujos limites da dimensão VC possam ser computados, utilizando o Princípio de Minimização do Risco Estrutural para identificar o hiperplano ótimo que maximize a margem dos exemplos mais próximos, sendo equivalente a minimizar o limite da dimensão VC.

No caso de padrões separáveis, a Máquina de Vetor de Suporte tem valor zero para a taxa de erro de treinamento e minimiza a dimensão VC.

O treinamento consiste em achar um hiperplano que separe perfeitamente os pontos de cada classe e cuja margem de separação seja máxima:

$$h(x) = Wx + b = ~~ < w . x > + b = ~~ \sum_{i=1}^{n} w_i x_i + b$$

* $W$ sendo o vetor de pesos que define a direção perpendicular ao hiperplano;
* $b$ sendo o bias que move o hiperplano paralelamente a si mesmo.

Quando o hiperplano encontrado é ótimo temos $Wx + b = 0$

#### **- SVM com Margens Maximais:**

O hiperplano ótimo é encontrado minimizando a distância euclidiana entre um hiperplano ($H: Wx + b = 0$) e os pontos que estão sobre a margem (vetores de suporte). Assim, temos:

* **$H\_1$:** $Wx^- + b = -1$
* **$H\_2$:** $Wx^+ + b = +1$

A distância entre $H\_1$ e $H\_2$ é a maior possível, equidistantes de $H$, e não possui nenhum ponto entre eles.

Todos os demais pontos que não estão sobre as margens podem ser removidos do conjunto de dados, reduzindo o conjunto necessário para o aprendizado para a posterior classificação.

#### **- SVM com Margens Suaves:**

Quando não se permite flexibilidade no hiperplano separador, o tamanho das margens é reduzido implicando em menor generalização. Assim, podemos tolerar alguns exemplos dentro das margens, relaxando as restrições impostas durante a otimização.

SVM com Margens Suaves introduz o conceito de variáveis soltas $६$ associadas a cada vetor de treinamento, além de um parâmetro $C$ (definido por experimentação) para controlar o peso do número de erros:

* Caso $0 <= ६ <= 1$, o ponto se encontra do lado correto do hiperplano de separação;
* Caso $६ > 1$, o ponto se encontra do lado errado do hiperplano de separação;
* Quando $C$ possui valores altos permite-se poucos erros ($C → \infty$, então erro igual a 0), ou seja, baixo viés e alta variância;
* Quando $C$ possui valores pequenos permite-se mais erros, ou seja, alto viés e baixa variância.

<figure><img src=".gitbook/assets/image (18).png" alt=""><figcaption><p>Figura 5 -</p></figcaption></figure>

## **5.2 - Máquinas de Vetores de Suporte não lineares:**

O **Teorema de Cover** garante que um espaço de características com padrões não linearmente separáveis pode ser transformado em um novo espaço em que os padrões são linearmente separáveis, desde que duas condições sejam satisfeitas:

* A transformação seja não linear;
* A dimensão do espaço de características seja suficientemente grande.

**Modelo não linear:**

* Substitui $x$ por uma função $φ(x)$, em que $φ$ é um mapeamento não linear do espaço de características;
* A função $φ$ é conhecida como kernel de produto interno e podem ser funções polinomiais ou gaussianos.

$$W φ(x) + b = 0$$

Assim, um kernel satisfaz o Teorema de Mercer (seus autovalores devem ser maiores que zero) e é definido por:

$$k(x,x') = φ(x_i) φ(x_j)$$

**Principais Kernels:**

* Polinomial:
  * Polinômios de alto grau $p$ tendem a causar overfitting (definido pelo usuário);
  * É frequentemente utilizado em dados normalizados;
* Gaussiano:
  * É indicado quando a quantidade de objetos é maior que a quantidade de atributos;
  * Parâmetro $\sigma$ pode proporcionar diferentes performances (especificado pelo usuário);
* Radial Basis Function (RBF):
  * Quando $\gamma$ é baixo, a região de fronteira se torna ampla;
  * Quando $\gamma$ é alto, a região de fronteira se torna muito fragmentada, criando até mesmo "ilhas";
* Sigmoide:
  * Muito utilizado em Multi Layer Perceptron;
  * Apenas alguns valores de $\eta$ e $\nu$ satisfazem o Teorema de Mercer.

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# SVM com kernel linear
svm = SVC(kernel='linear')
scores = cross_val_score(svm, X, Y, cv=kf)
print("Acurácia com SVM Linear: %.4f +/- %.4f" % (scores.mean(), scores.std()))

# SVM com kernel RBF (radial basis function)
svm = SVC(kernel='rbf')
scores = cross_val_score(svm, X, Y, cv=kf)
print("Acurácia com SVM RBF: %.4f +/- %.4f" % (scores.mean(), scores.std()))

# SVM com kernel polinomial de grau 3
svm = SVC(kernel='poly', degree=3)
scores = cross_val_score(svm, X, Y, cv=kf)
print("Acurácia com SVM Poly: %.4f +/- %.4f" % (scores.mean(), scores.std()))

```

```
Acurácia com SVM Linear: 0.9733 +/- 0.0249
Acurácia com SVM RBF: 0.9667 +/- 0.0298
Acurácia com SVM Poly: 0.9733 +/- 0.0249
```

## **6 - Árvores de Decisão:**

**Árvores de decisão** são modelos estatísticos que se baseiam na estratégia de dividir e conquistar, ou seja, um problema complexo é decomposto em problemas mais simples de forma recursiva.

Esta técnica é fundamentada na identificação de **relacionamentos entre as variáveis** em que o classificador é induzido (construído) pela generalização dos exemplos fornecidos para posterior dedução.

**Estrutura e conceitos iniciais:**

* Nós-folhas indicam as classes (elipses finais);
* Nós de decisão (elipses intermediárias) definem um teste sobre o valor de um atributo específico, determinando uma ramificação a ser percorrida (linhas);
* O primeiro nó determina a raiz da árvore e nós intermediários determinam sub-árvores;
* O nó-raiz pode ser rearranjado para reduzir o percurso até uma folha, alterando a topologia da árvore;
* O percurso contido da raiz até uma classe determina a regra de classificação;
* Em cada nó da árvore existe um teste que representa o direcionamento do percurso da raiz para um nó-folha, determinando qual das arestas existentes deverá ser escolhida.

**Exemplo:**

* 7 classes: watermelon; apple; grape; grapefruit; lemon; banana; cherry
* 4 atributos: color; size; shape; taste

Considerando instâncias de teste:

* $x\_1$ = \[green, small, round, sour], a classe será "Grape"
* $x\_2$ = \[red, small, round, sour], a classe será "Grape"

<figure><img src=".gitbook/assets/image (19).png" alt=""><figcaption><p>Figura 6 -</p></figcaption></figure>

#### **- Grau de impureza:**

O critério de seleção dos atributos nos nós varia entre os algoritmos, mas são baseados na distribuição das classes dos exemplos antes e depois de uma divisão. Assim, **a ideia é reduzir ao máximo o grau de impureza dos nós-filhos**, maximizando o balanceamento da distribuição de classes:

**Exemplo 1:** Não homogênea -> alto grau de impureza

* Classe 0 com 5 elementos
* Classe 1 com 5 elementos

**Exemplo 2:** Homogênea -> baixo grau de impureza

* Classe 0 com 9 elementos
* Classe 1 com 1 elemento

Entretanto, **uma árvore perfeita geralmente não será uma boa solução por ter memorizado os dados utilizados na construção**, não sendo generalizável para os dados de teste. A fim de evitar esta perfeição, durante a construção, pode-se utilizar:

* Um conjunto de validação;
* Um limiar para o grau de impureza; ou
* Uma quantidade de nós pré-definida.

O Índice de Gini e o Ganho de Informação são comumente utilizados para medir o grau de impureza nos nós:

* Quanto menor o grau de impureza, mais desbalanceada é a distribuição de classes;
* Em um determinado nó, a impureza é nula se todos os exemplos nele pertencerem à mesma classe;
* O grau de impureza é máximo no nó se houver o mesmo número de exemplos para cada classe possível.

#### **- Poda (pruning):**

A poda objetiva melhorar a taxa de acerto do modelo para novos exemplos, ou seja, permite que o modelo possa generalizar para exemplos que não foram utilizados durante o treinamento.

A necessidade deste recurso advém da possibilidade de ocorrer o sobreajuste, em que as arestas definidas podem refletir ruídos. Consequentemente, a exclusão de algumas arestas torna a árvore mais simples e facilita a interpretação pelo usuário.

**Abordagens:**

* A **pré-poda** é aplicada durante o processo de construção da árvore, **interrompendo a divisão de um nó interno e transformando-o em um nó-folha**. Por exemplo, se o Ganho de Informação for menor que um valor pré-estabelecido o nó atual se transforma em nó-folha interrompendo a divisão.
* O **pós-poda** é realizado após a construção total da árvore, em que se calcula **uma taxa de erro para cada nó demonstrando o impacto da exclusão e da manutenção de suas respectivas ramificações**. Se essa diferença for menor que um valor pré-estabelecido ocorre a poda.

```python
from sklearn import tree
from sklearn.model_selection import KFold, cross_val_score

# Definindo o cross-validation
numeroFolds = 5
kf = KFold(n_splits=numeroFolds, shuffle=True)

# Decision Tree com critério Gini (padrão)
dct = tree.DecisionTreeClassifier()
scores = cross_val_score(dct, X, Y, cv=kf)
print('Acurácia com critério Gini: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

# Decision Tree com critério Entropy
dct = tree.DecisionTreeClassifier(criterion='entropy')
scores = cross_val_score(dct, X, Y, cv=kf)
print('Acurácia com critério Entropy: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

# Decision Tree com critério Gini e profundidade máxima 3 (poda)
dct = tree.DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(dct, X, Y, cv=kf)
print('Acurácia com Gini e profundidade 3: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

# Nota: Em geral, o critério de parada (max_depth, poda) tem impacto maior na performance do que o critério de impureza (Gini ou Entropy).

```

```
Acurácia com critério Gini: 0.9600 +/- 0.0133
Acurácia com critério Entropy: 0.9333 +/- 0.0422
Acurácia com Gini e profundidade 3: 0.9467 +/- 0.0340
```

```python
from sklearn import tree
import matplotlib.pyplot as plt

# Treinando a árvore de decisão com critério Gini usando todo o conjunto de dados
dctGini = tree.DecisionTreeClassifier(criterion='gini').fit(X, Y)

# Criando uma figura para plotar a árvore
plt.figure(figsize=(15, 10))

# Plotando a árvore de decisão com cores preenchidas para as classes
tree.plot_tree(dctGini, filled=True)

# Exibindo o gráfico
plt.show()
```

<figure><img src=".gitbook/assets/image (20).png" alt=""><figcaption><p>Figura 7 -</p></figcaption></figure>

## **7 - Ensembles:**

Os modelos estudados até o momento formalizam uma otimização de parâmetros única, em que **somente um conjunto de hipóteses é provido**, ou seja, para classificar uma classe $A$ há somente uma possibilidade.

Os métodos de _ensembles_ visam ampliar este horizonte promovendo um grupo de conjuntos de hipóteses possíveis, ou seja, **múltiplos modelos são instanciados simultaneamente**, cada um com sua modelagem própria. Ao final deste processo, suas **respostas são combinadas a fim de potencializar a capacidade preditiva**.

Consequentemente, quando combinamos classificadores objetivamos uma capacidade preditiva mais acurada com aumento na complexidade computacional. Assim, **o que devemos avaliar é se a combinação é justificável.**

A principal metodologia para combinar diferentes métodos preditivos é por votação, podendo ter três cenários:

* Unanimidade: muito utilizado em diagnósticos médicos, ou seja, um diagnóstico só é positivo se todos os modelos derem o mesmo resultado;
* Maioria simples: em alguns sistemas críticos, por exemplo, dados de diversos sensores para apontar uma possível falha em indústrias;
* Mais votado: cenários não impactantes, por exemplo, classificar imagens de animais, gêneros literários, etc.

#### **- Bagging (Bootstrap AGGregatION):**

Utiliza **múltiplas versões do conjunto de treinamento**, cada um sendo criado selecionando $n' < n$ exemplos a partir de $D$ **com reposição**. Cada um dos subconjuntos formados é utilizado para **treinar um modelo paralelo** e, ao final, **o resultado final é composto pela classe mais votada**.

Geralmente, o modelo é exatamente o mesmo (árvores de decisão, SVMs, ou outro qualquer) e **tende a melhorar a performance** por ter muitos modelos combinados. Contudo, ainda não há uma teoria que comprove que Bagging realmente estabiliza os classificadores já estudados.

#### **- Random Forest:**

**Random Forest** é o método de Bagging mais conhecido e utilizado, sendo baseado em Árvores de Decisão, mas que possui algumas particularidades:

* A execução do boostrap busca criar subconjuntos descorrelacionados entre si;
* Pode ser utilizado tanto para classificação (classe mais votada) quanto para regressão (média dos resultados).

#### **- AdaBoost (Adaptive Boosting):**

A ideia de **Boosting** é criar um classificador base e ir adicionando novos classificadores para fortalecer este primeiro, aumentando sua performance.

```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Bagging com SVC como estimador base (com reposição - bootstrap=True)
bag = BaggingClassifier(estimator=SVC(), n_estimators=10)
scores = cross_val_score(bag, X, Y, cv=kf)
print('Acurácia Bagging com SVC: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

# Pasting: Bagging sem reposição de exemplos (bootstrap=False)
bag = BaggingClassifier(estimator=SVC(), n_estimators=10, bootstrap=False)
scores = cross_val_score(bag, X, Y, cv=kf)
print('Acurácia Pasting com SVC: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

# Random Forest com 100 árvores e profundidade máxima 4
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4)
scores = cross_val_score(rf, X, Y, cv=kf)
print('Acurácia Random Forest: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

# AdaBoost com 50 estimadores fracos (padrão: Decision Trees)
bag = AdaBoostClassifier(n_estimators=50)
scores = cross_val_score(bag, X, Y, cv=kf)
print('Acurácia Ada Boost: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

```

```
Acurácia Bagging com SVC: 0.9733 +/- 0.0249
Acurácia Pasting com SVC: 0.9467 +/- 0.0748
Acurácia Random Forest: 0.9533 +/- 0.0452
Acurácia Ada Boost: 0.9400 +/- 0.0133
```

## **Resumo:**

Tarefas de Classificação:

* Conceitos
* Métricas

KNN:

* Função de distância
* Regras de classificação
* Heurísticas para escolher o valor de $k$

Métodos Probabilísticos:

* Naive-Bayes
* Regressão Logística

Árvores de Decisão:

* Grau de impureza
* Poda

Ensembles:

* Bagging
* Florestas Aleatórias
* Ada Boost

## **Considerações finais:**

* KNN:
  * Por ser um método baseado em distância é extremamente importante fazer **normalizações e padronizações dos atributos**.
  * A **função de distância** é o maior viés de estudo no k-NN, pois possibilita as maiores variações nos resultados obtidos. Outra questão importante em relação à função de distância é o **espaço dos objetos**. Se o espaço é plano, a menor distância é uma reta; se o espaço é uma esfera, a menor distância é uma curva.
  * Uma das vantagens da regra de classificação por **peso na distância é sua robustez em relação a dados ruidosos** do treinamento e sua eficiência quando o conjunto de treinamento é suficientemente grande.
  * k-NN é extremamente **sensível ao conceito da “maldição de dimensionalidade”**. Contudo, se cada atributo tiver um nível de importância especificado ou se for eliminado com pré-processamentos, este problema pode ser amenizado.
  * Uma das particularidades do k-NN é a **ausência da fase de treinamento**, comum a outras técnicas de classificação supervisionada. Isso ocorre devido à sua fundamentação, consistindo em uma análise de distância e não em aprendizado propriamente.
* Métodos probabilísticos:
  * O classificador Naive Bayes descreve um caso particular de uma Rede Bayesiana, o qual **considera que as variáveis do domínio são condicionalmente independentes, ou seja, uma característica não é relacionada com a outra**. Em decorrência desta restrição utiliza-se o termo "naive". Mesmo sendo um modelo simples, ele tem obtido sucesso comparado a outros classificadores mais sofisticados.
  * O **treinamento do Naive Bayes** é relativamente rápido devido à ausência de coeficientes, se resumindo a cálculos de probabilidades de cada classe.
  * A **Regressão Logística** tem como vantagens o fato de fornecer os resultados em termos de probabilidade e requerer pequeno número de suposições para um alto grau de confiabilidade.
* SVM:
  * Originalmente, SVMs foram desenvolvidas para classificação binária. No caso de **classificação de múltiplas classes** é necessária a utilização de algum método para estender a SVM binária ou combinar os resultados de várias SVMs binárias.
  * A **principal desvantagem** é o custo excessivo em grandes datasets, pois a matriz do kernel de treinamento cresce quadraticamente.
  * Outra desvantagem é em relação a **dados desbalanceados**, pois encontrar o hiperplano separador com diferentes proporções pode influenciar drasticamente o hiperplano ótimo.
* Árvores de Decisão:
  * As árvores de decisão são **estruturas de fácil entendimento** devido ao seu conceito ser amplamente utilizado na computação.
  * Não requer normalização dos dados (um atributo por vez) e pode ser utilizada com atributos numéricos e categóricos, sendo **robusto a outliers** (dados não métricos).
  * Considerando uma árvore já construída seu uso é imediato e de complexidade computacional baixa (logarítmica). Entretanto, **a construção da árvore é altamente custosa**.
  * Possui **baixo desempenho em problemas com muitas classes e poucos dados**.
  * Um dos problemas recorrentes com árvores de decisão é o **overfitting que ocorre quando a árvore cresce até sua profundidade máxima**. Entretanto, o processo de poda mitiga este problema atribuindo uma profundidade ou quantidade de folhas máximas.
  * Outro ponto controverso é sua instabilidade: **pequenas variações nos dados de treinamento podem gerar árvores completamente distintas**.
* Ensembles:
  * **Bagging** treina classificadores de forma paralela e **Boosting** de forma sequencial.
  * Na prática, podemos escolher quaisquer **classificadores base** para montar o ensemble.
  * Podemos até mesmo **montar ensembles próprios** com diferentes classificadores ou regressores e combinar suas saídas manualmente.

## **Bibliografia básica:**

* Barber. Bayesian Reasoning and Machine Learning, 2016.
* Bishop. Pattern Recognition and Machine Learning, 2006.
* Duda, Hart e Stork, Pattern Classification, Wiley, 2nd. Edition, 2001.
* Fukunaga. Introduction to Statistical Pattern Recognition,  \
  Academic, 1990.
* Hastie, Tibshirani e Friedman. The Elements of Statistical  \
  Learning: Data Mining, Inference, and Prediction, Springer, 2002.
* Haykin. Neural Networks and Learning Machines, 3nd. Edition, Prentice Hall, 2009.
* Izbicki, Santos. Aprendizado de Máquina - Uma abordagem estatística, 2020.
* Kuncheva. Combining Pattern Classifiers: Methods and Algorithms, Wiley-Interscience, 2004.
* Mitchell. Machine Learning, McGrawHill, 1997.

## **Leituras complementares:**

* Fukunaga, Narendra. **A branch and bound algorithm for computing k-nearest neighbors**, 1975.
* Zhang, Li, Zong, Zhu, Cheng, Debo. **Learning k for knn classification**, 2017.
* Zhang, Li, Zong, Zhu, Wang. **Efficient kNN classification with different numbers of nearest neighbors**, 2017.
* Zhang. **Cost-sensitive KNN classification**, 2020.
* Marcot, Bruce G and Penman, Trent D. **Advances in Bayesian network modelling: Integration of modelling technologies**, 2019.
* Shih, Andy and Choi, Arthur and Darwiche, Adnan. **A symbolic approach to explaining bayesian network classifiers**, 2018.
* Cervantes, Garcia-Lamont, Rodriguez-Mazahua, Lopez. **A comprehensive survey on support vector machine classification: applications, challenges and trends**, 2020.
* Mello, Ponti. Machine learning: a practical approach on the statistical learning theory, Elsevier, 2018.
* Pisner, Schnyer. **Support vector machine**, 2020.
* Vapnik. **An overview of statistical learning theory**, 1999.
* Kulkarni, Harman. **Statistical learning theory: a tutorial**, 2011.
* Su, Zhang. **A fast decision tree learning algorithm**, 2006.
* Brijain, Patel, Kushik, Rana. **A survey on decision tree algorithm for classification**, 2014.
* Jin, De-Lin, Fen-Xiang. **An improved ID3 decision tree algorithm**, 2009.
