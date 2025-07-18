# Aula 5

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

#### **- Regressão:**

Dado um conjunto de observações $D = \{{X, Y\}}$, temos uma função $f()$ que mapeia uma entrada $X\_i$ (conjunto de atributos) à sua respectiva saída $y\_i$. Assim, $f(x)$ aprende a aproximação que permite estimar $y$ (valores continuos) para os valores de $x$.

* A função $f()$ é um regressor;
* $y$ será um valor real;
* Exemplos: peso, temperatura, demanda.

**Em tarefas de regressão queremos predizer variáveis quantitativas.**

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# Carrega o dataset diabetes (variáveis independentes X e variável alvo y)
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)

# Divide os dados em treino e teste
# Utiliza todas as amostras exceto as últimas 20 para treino
X_train, y_train = X[:-20], y[:-20]

# Últimas 20 amostras para teste
X_test, y_test = X[-20:], y[-20:]

print("Training data:", X_train.shape)  # Exibe o formato dos dados de treino
print("Testing data:", X_test.shape)    # Exibe o formato dos dados de teste

```

```
Training data: (422, 10)
Testing data: (20, 10)
```

## **2 - Métricas de avaliação para Tarefas de Regressão:**

**Soma dos Quadrados dos Erros (SSE):**

* Quanto menor, melhor a performance

$$SSE = \sum_{i=1}^{n} (y^i - f(x^i))^2$$

**Erro Quadrático Médio (MSE):**

* Quanto menor, melhor a performance

$$MSE = \frac{1}{n}\sum_{i=1}^{n} (y^i - f(x^i))^2 = \frac{SSE}{n}$$

**Raiz Quadrada do Erro Quadrático Médio (RMSE):**

* Quanto menor, melhor a performance
* Mesma unidade de medida do valor predito em y

$$RMSE = \sqrt(MSE)$$

**Erro Absoluto Médio (MAE):**

* Quanto menor, melhor a performance
* Mesma unidade de medida do valor predito em y

$$MAE = \frac{1}{n}\sum_{i=1}^{n} |y^i - f(x^i)|$$

***

**Coeficiente de Determinação (R2):**

* Quanto maior, melhor a performance
* Mesma unidade de medida do valor predito em y

$$R^2 = 1 - \frac{MSE}{Var(y)}$$

**Erro Percentual Absoluto Médio (MAPE):**

* Quanto menor, melhor a performance
* Mede a acurácia do regressor em termos de porcentagem

$$MAPE = \frac{1}{n}\sum_{i=1}^{n} \frac{|y^i - f(x^i)|}{y^i}$$

## **3 - Regressão Linear Múltipla:**

A Regressão Linear Múltipla possibilita utilizar várias variáveis simultaneamente para encontrar a saída desejada, buscando definir os coeficientes $\beta$.

$$f(x) = \sum_{j=1}^{p} \beta_j X_j + \beta_0$$

Veja que a definição acima é uma generalização da Regressão Linear Simples quando $p=1$: $f(x) = \beta\_1x + \beta\_0$

A modelagem básica Regressão Linear Múltipla aplica os conceitos do **Método dos Mínimos Quadrados (Linear Least Square)**, em que ajusta o modelo linear com os coeficientes $\beta$ minimizando a soma dos quadrados residuais (RSS) entre os valores alvos ($y$) e os valores preditos ($f(x)$):

$$RSS(\beta) = \sum_{i=1}^{N} (y_i - f(x_i))^2 = \sum_{i=1}^{N} (y_i - \beta_0 - \sum_{j=1}^{p} \beta_j X_{ij})^2 = \sum_{i=1}^{N} \epsilon_i^2$$

De forma geral, para todas as instâncias, queremos que a diferença entre $y$ e $f(x) = \hat{y}$ seja a menor possível:

<figure><img src=".gitbook/assets/image (21).png" alt=""><figcaption><p>Figura 1 - </p></figcaption></figure>

```python
# Regressão Linear usando todos os atributos do dataset
regr = LinearRegression()

# Treina o modelo com os dados de treino e realiza predições no conjunto de teste
y_pred = regr.fit(X_train, y_train).predict(X_test)

# Avaliação do modelo
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))  # Erro quadrático médio
print("R Score: %.2f" % r2_score(y_test, y_pred))  # Coeficiente de determinação (quanto mais próximo de 1, melhor)

```
