# Aula 2

## **1 - Visão Geral para um Projeto de Ciência de Dados:**

**Abordar o problema e analisar o panorama:**

* Definir o objetivo e como a solução será utilizada;
* Consultar soluções atuais.

**Obter os dados:**

* Tipo de dados, processo de coleta e obrigações legais;
* Forma de abordar o problema e métricas a serem utilizadas;
* Verificar capacidade de processamento e armazenamento.

**Explorar e preparar dados:**

* Analisar atributos e visualizar sua distribuição;
* Pré-processamento para limpeza, correção e transformação dos dados;
* Remoção de outliers e seleção/redução de atributos.

**Seleção de modelos:**

* Escolher modelos de diferentes categorias e formar um _baseline_;
* Analisar e validar os resultados obtidos de forma mais aprofundada;
* Selecionar os modelos mais promissores (consultar literatura).

**Otimização dos modelos:**

* Ajustar hiperparâmetros e utilizar todos os dados disponíveis para treinamento;
* Explorar modelos mais complexos (_ensembles_ e redes neurais);
* Validar os resultados obtidos e comparar performances.

**Finalização:**

* Treinar o modelo com todo o conjunto de dados disponível;
* Documentar e apresentar a modelagem com resultados;
* Implementar rotinas para colocar o modelo em produção;
* Monitorar constantemente o funcionamento do modelo.

## **2 - Os Dados:**

Podem estar organizados em diferentes formatos, podendo ser lidos e analisados manualmente em um determinado contexto cultural. Assim, temos a capacidade de compreender estes dados, pois:

* Compreendemos mistura de símbolos e elementos;
* Detectamos e corrigimos possíveis erros;
* Preenchemos informações faltantes;
* Lidamos com ambiguidades;
* Reconhecemos sentimentos (humor, tristeza, alegria) por meio de elementos complexos.

**Exemplos:**

* O símbolo ? denota entonação para perguntas. (símbolos e elementos)
* Meu nome é F3rnand0. (possíveis erros)
* Me desculpe, não quis dizer iss. (informações faltantes)
* João foi atrás do táxi correndo. (ambiguidade)
* Maria chorou. (sentimento)

**Estes conceitos são aplicados somente em textos?**

## **2.1 - Dados não estruturados:**

**Dados não estruturados são de análise computacional difíceis e são caracterizados por:**

* Arquivos binários (imagens, vídeos, etc) ou composto por caracteres (texto, por exemplo);
* São flexíveis, ou seja, não possuem uma estrutura fixa;
* Podem ser coletados de diversas maneiras e com baixo controle;
* Podem possuir redundância (múltiplas fontes podem conter o mesmo dado);
* Possuem alta dimensionalidade.

**Exemplos de dados:**

* E-mails, artigos e notícias;
* Comentários em redes sociais;
* Review de produtos;
* Áudio;
* Imagens e vídeos;
* Sinais provenientes de sensores.

## **2.2 - Dados estruturados:**

Dados estruturados são os que:

* Possuem elementos cujo armazenamento e recuperação estão formatados;
* Seu acesso e controle são diretos e mais fáceis de manipular;
* São comumente encontrados em banco de dados relacionais ou planilhas.

| Nome            | CPF          | Nascimento | Email                   |
| --------------- | ------------ | ---------- | ----------------------- |
| Fernando Santos | 123456789-00 | 1985       | fernando@web.com        |
| Maria Silva     | 456789123-00 | 1960       | mariasilva@internet.com |
| Pedro Almeida   | 789123456-00 | 1974       | almeida@internet.com    |

Qual a dificuldade de estruturar dados não-estruturados?

| Arquivo                       | Largura | Altura | Tipo      |
| ----------------------------- | ------- | ------ | --------- |
| manga\_camiseta.jpg           | 164     | 180    | Vestuário |
| manga\_camiseta\_vermelha.jpg | 164     | 180    | Vestuário |
| manga\_fruta.jpg              | 164     | 180    | Fruta     |

**Termos em bases estruturadas:**

* Campo, atributo ou variável: representado por uma coluna, especifica uma característica;
* Tupla ou registro: representando por uma linha, especifica um exemplo;
* Tabela ou relação: conjunto de linhas e colunas, formando um conjunto de dados.

**Cada um dos atributos pode ser quantitativo ou qualitativo:**

* **Quantitativos:** são representatos por valores numéricos discretos. Exemplos: peso, altura, quantidade de patas, preço, etc.
* **Qualitativos:** representam níveis ou escala de valores. Exemplos: nível de escolaridade, fabricante de um determinado produto, etc.

Os atributos qualitativos e quantitativos contínuos requerem sua transformação para valores quantitativos discretos.

## **2.3 - Coleta de Dados:**

**Na dúvida, colete o máximo de dados possível.**

**Alguns pontos importantes a serem considerados:**

* Os dados a serem coletados estão dentro dos limites éticos e legais?
  * Lei Geral de Proteção de Dados Pessoais (LGPD): http://www.planalto.gov.br/ccivil\_03/\_Ato2015-2018/2018/Lei/L13709.htm
  * GDPR Europeia: https://gdpr.eu/
* Quais dados serão coletados?
* Por quanto tempo estaremos coletando?
* Qual a quantidade de dados necessária?
* Quanto tempo levará para implementar o protocolo de coleta?

**Possíveis fontes de coleta:**

* Entrevistas:
  * Pessoal e de caráter informal;
  * Os pontos a serem discutidos devem ser listados previamente;
  * O discurso a ser utilizado deve ser adequado ao entrevistado;
  * Não se pode influenciar o entrevistado.
* Questionário:
  * Aplicável a uma grande quantidade de pessoas;
  * Pode ser realizado de forma automatizada (Google Forms, por exemplo);
  * Questões abertas (dissertativas) e fechadas (alternativas).
* Scripts de rastreamento em websites (preferências e perfil do cliente);
* Redes Sociais (notícias, opiniões, sentimentos);
* Bases públicas governamentais (dados gerais sobre a população);
* Sites especializados em disponibilizar datasets (áudio, imagens, vídeos, textos);
* Dados privados da própria empresa (vendas, qualidade dos produtos, perfil do cliente, imagens, aúdios, vídeos, documentos, e-mails, etc).

**O que devemos coletar?**

* Que tipo de dados é mais apropriado?
* Quem é o interessado principal na coleta?
* Os dados podem ser reaproveitados por terceiros?
* Quais as variáveis envolvidas?
* Qual o resultado esperado?

**Viés durante a coleta de dados:**

* Poucos dados;
* Medidas;
* Amostragem.

**Problemas durante a coleta:**

* Informações faltantes;
* Redundância;
* Despadronização;
* Outliers;
* Desbalanceamento;
* Variáveis categóricas.

## **3 - Tratando informações faltantes:**

Erros no protocolo de coleta de dados e na sua execução:

* Programação do sistema computacional;
* Digitação do usuário/papel para o sistema;
* Transferência e/ou migração de bases de dados em diferentes sistemas;
* Conversão de arquivos para diferentes formatos.

Omissão ou não preenchimento de campos:

* Esquecimento;
* Falta de esclarecimento sobre o dado a ser coletado;
* Falta de obrigatoriedade do dado a ser coletado.

```python
!pip install --upgrade openpyxl
```

**Possíveis ações para dados faltantes:**

* Remover linhas/colunas que contenham dados faltantes;
* Preencher os valores faltantes por alguma heurística.

```python
# Importações necessárias
import numpy as np
from tensorflow.keras.datasets import boston_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset Boston Housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print("Formato dos dados de treino:", x_train.shape)
print("Formato dos dados de teste:", x_test.shape)

# Treinando o modelo de regressão linear com todos os atributos
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Realizando predições no conjunto de teste
y_pred = regressor.predict(x_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nErro quadrático médio (MSE): {mse:.2f}")
print(f"Coeficiente de determinação (R²): {r2:.2f}")
```

```
Formato dos dados de treino: (404, 13)
Formato dos dados de teste: (102, 13)

Erro quadrático médio (MSE): 23.20
Coeficiente de determinação (R²): 0.72
```

**Remoção de linhas e colunas com dados faltantes:**

```python
import pandas as pd
import numpy as np

# Simulando um DataFrame semelhante ao Boston Housing com valores faltantes
np.random.seed(42)
x_train = pd.DataFrame(
    np.random.rand(100, 13),  # 100 amostras, 13 atributos (como no dataset Boston)
    columns=[f"Feature_{i}" for i in range(13)]
)

# Introduzindo valores faltantes aleatoriamente em algumas colunas
x_train.loc[5:15, "Feature_0"] = np.nan
x_train.loc[20:30, "Feature_2"] = np.nan
x_train.loc[40:50, "Feature_6"] = np.nan

# Etapas de análise e tratamento

print("Lendo dados com valores faltantes:")
print("Formato original:", x_train.shape)

# Remoção de linhas com valores faltantes
print("\nRemovendo linhas com valores faltantes:")
x_train_linhas_removidas = x_train.dropna()
print("Formato após remoção de linhas:", x_train_linhas_removidas.shape)

# Remoção de colunas com índices 0, 2 e 6 (independentemente de conterem ou não NaN)
print("\nRemovendo colunas 0, 2 e 6:")
x_train_colunas_removidas = x_train.drop(columns=x_train.columns[[0, 2, 6]])
print("Formato após remoção de colunas:", x_train_colunas_removidas.shape)

```

```
Lendo dados com valores faltantes:
Formato original: (100, 13)

Removendo linhas com valores faltantes:
Formato após remoção de linhas: (67, 13)

Removendo colunas 0, 2 e 6:
Formato após remoção de colunas: (100, 10)
```

**Preenchendo valores faltantes:**

* Com zeros, simplesmente para manter a instância na base e permitir o uso de seus atributos;
* Pela média, calculando o valor médio dos valores contidos no atributo.

```python
import pandas as pd
import numpy as np

# Simulando um DataFrame com estrutura semelhante ao Boston Housing
np.random.seed(42)
x_train = pd.DataFrame(
    np.random.rand(15, 6),  # apenas 15 linhas e 6 colunas para visualização clara
    columns=[f"Feature_{i}" for i in range(6)]
)

# Introduzindo valores faltantes em colunas 2 e 4
x_train.loc[[2, 5, 8], 2] = np.nan  # valores faltantes em Feature_2
x_train.loc[[1, 6, 10], 4] = np.nan  # valores faltantes em Feature_4

print("Carregando dados com valores faltantes (simulado):")
print(x_train[x_train[2].isnull()])

# Preenchendo com zeros
print("\n\nPreenchendo valores faltantes na coluna 2 com zeros:")
x_train[2] = x_train[2].fillna(0)
print(x_train.head(15))

# Preenchendo com a média da coluna 4 (ignorando os NaNs)
print("\n\nPreenchendo valores faltantes na coluna 4 com a média:")
media_col_4 = x_train[4].mean(skipna=True)
x_train[4] = x_train[4].fillna(media_col_4)
print(x_train.head(15))

```

```
Carregando dados com valores faltantes (simulado):
    Feature_0  Feature_1  Feature_2  Feature_3  Feature_4  Feature_5   2   4
0    0.374540   0.950714   0.731994   0.598658   0.156019   0.155995 NaN NaN
1    0.058084   0.866176   0.601115   0.708073   0.020584   0.969910 NaN NaN
2    0.832443   0.212339   0.181825   0.183405   0.304242   0.524756 NaN NaN
3    0.431945   0.291229   0.611853   0.139494   0.292145   0.366362 NaN NaN
4    0.456070   0.785176   0.199674   0.514234   0.592415   0.046450 NaN NaN
5    0.607545   0.170524   0.065052   0.948886   0.965632   0.808397 NaN NaN
6    0.304614   0.097672   0.684233   0.440152   0.122038   0.495177 NaN NaN
7    0.034389   0.909320   0.258780   0.662522   0.311711   0.520068 NaN NaN
8    0.546710   0.184854   0.969585   0.775133   0.939499   0.894827 NaN NaN
9    0.597900   0.921874   0.088493   0.195983   0.045227   0.325330 NaN NaN
10   0.388677   0.271349   0.828738   0.356753   0.280935   0.542696 NaN NaN
11   0.140924   0.802197   0.074551   0.986887   0.772245   0.198716 NaN NaN
12   0.005522   0.815461   0.706857   0.729007   0.771270   0.074045 NaN NaN
13   0.358466   0.115869   0.863103   0.623298   0.330898   0.063558 NaN NaN
14   0.310982   0.325183   0.729606   0.637557   0.887213   0.472215 NaN NaN


Preenchendo valores faltantes na coluna 2 com zeros:
    Feature_0  Feature_1  Feature_2  Feature_3  Feature_4  Feature_5    2   4
0    0.374540   0.950714   0.731994   0.598658   0.156019   0.155995  0.0 NaN
1    0.058084   0.866176   0.601115   0.708073   0.020584   0.969910  0.0 NaN
2    0.832443   0.212339   0.181825   0.183405   0.304242   0.524756  0.0 NaN
3    0.431945   0.291229   0.611853   0.139494   0.292145   0.366362  0.0 NaN
4    0.456070   0.785176   0.199674   0.514234   0.592415   0.046450  0.0 NaN
5    0.607545   0.170524   0.065052   0.948886   0.965632   0.808397  0.0 NaN
6    0.304614   0.097672   0.684233   0.440152   0.122038   0.495177  0.0 NaN
7    0.034389   0.909320   0.258780   0.662522   0.311711   0.520068  0.0 NaN
8    0.546710   0.184854   0.969585   0.775133   0.939499   0.894827  0.0 NaN
9    0.597900   0.921874   0.088493   0.195983   0.045227   0.325330  0.0 NaN
10   0.388677   0.271349   0.828738   0.356753   0.280935   0.542696  0.0 NaN
11   0.140924   0.802197   0.074551   0.986887   0.772245   0.198716  0.0 NaN
12   0.005522   0.815461   0.706857   0.729007   0.771270   0.074045  0.0 NaN
13   0.358466   0.115869   0.863103   0.623298   0.330898   0.063558  0.0 NaN
14   0.310982   0.325183   0.729606   0.637557   0.887213   0.472215  0.0 NaN


Preenchendo valores faltantes na coluna 4 com a média:
    Feature_0  Feature_1  Feature_2  Feature_3  Feature_4  Feature_5    2   4
0    0.374540   0.950714   0.731994   0.598658   0.156019   0.155995  0.0 NaN
1    0.058084   0.866176   0.601115   0.708073   0.020584   0.969910  0.0 NaN
2    0.832443   0.212339   0.181825   0.183405   0.304242   0.524756  0.0 NaN
3    0.431945   0.291229   0.611853   0.139494   0.292145   0.366362  0.0 NaN
4    0.456070   0.785176   0.199674   0.514234   0.592415   0.046450  0.0 NaN
5    0.607545   0.170524   0.065052   0.948886   0.965632   0.808397  0.0 NaN
6    0.304614   0.097672   0.684233   0.440152   0.122038   0.495177  0.0 NaN
7    0.034389   0.909320   0.258780   0.662522   0.311711   0.520068  0.0 NaN
8    0.546710   0.184854   0.969585   0.775133   0.939499   0.894827  0.0 NaN
9    0.597900   0.921874   0.088493   0.195983   0.045227   0.325330  0.0 NaN
10   0.388677   0.271349   0.828738   0.356753   0.280935   0.542696  0.0 NaN
11   0.140924   0.802197   0.074551   0.986887   0.772245   0.198716  0.0 NaN
12   0.005522   0.815461   0.706857   0.729007   0.771270   0.074045  0.0 NaN
13   0.358466   0.115869   0.863103   0.623298   0.330898   0.063558  0.0 NaN
14   0.310982   0.325183   0.729606   0.637557   0.887213   0.472215  0.0 NaN
```

## **4 - Removendo dados redundantes:**

A redundância de dados pode influenciar diretamente no aprendizado de modelos de Data Science.

Em dados não-estruturados:

* Timbre em documentos e cartas de uma mesma empresa/instituição;
* Stopping-words: artigos, preposições, conjunções;
* Imagens: logotipos, background.

Em dados estruturados:

* Linhas repetidas;
* Colunas repetidas;
* Colunas com valores constantes;
* Colunas com alta correlação.

| Nome            | Altura | Peso | Esporte   | e-Sport   | Nascimento |
| --------------- | ------ | ---- | --------- | --------- | ---------- |
| Fernando Santos | 174    | 82   | Badminton | Badminton | 1985       |
| Fernando Santos | 174    | 82   | Badminton | Badminton | 1985       |
| Ana Silva       | 170    | 60   | Vôlei     | Vôlei     | 1985       |

**Quais as possíveis consequências de utilizar dados redundantes em um modelo preditivo?**

### **Remoção de linhas repetidas:**

```python
# Verificando o número de linhas antes da remoção
print("Lendo dados com redundância de linhas:")
print("Formato original (com duplicatas):", x_train.shape)

# Remoção das duplicatas (mantendo apenas a primeira ocorrência)
x_train = x_train.drop_duplicates(keep='first')

# Verificando o número de linhas após remoção
print("\nApós remoção da redundância:")
print("Formato após remoção de duplicatas:", x_train.shape)
```

```
Lendo dados com redundância de linhas:
Formato original (com duplicatas): (20, 8)

Após remoção da redundância:
Formato após remoção de duplicatas: (15, 8)
```

### **Correlação de atributos:**

* Oferecem pouca contribuição para distinguir o conjunto de dados;
* Só é computada com valores numéricos.

```python
import numpy as np
import matplotlib.pyplot as plt

# Gerando dados com relação linear (alta covariância)
n = 100
X = np.zeros((2, n))
X[0, :] = np.random.uniform(-1, 1, size=n)
X[1, :] = 4 * X[0, :] + np.random.uniform(-0.15, 0.15, size=n)

# Gráfico 1: Variância em y
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_aspect('equal')
ax1.axis('off')
ax1.scatter(X[0, :], X[1, :], color='red', edgecolor='black', linewidth=1)
ax1.plot([0, 0], [-5, 5], color='black')  # eixo y
ax1.plot([-5, 5], [0, 0], color='black')  # eixo x
ax1.plot([0, 0], [-4, 4], color='blue', linewidth=5)  # variância y
ax1.plot([0, 1], [4, 4], ':k')
ax1.plot([0, -1], [-4, -4], ':k')
ax1.annotate('variância da coordenada y', xy=(-0.05, 3), xytext=(-4, 3),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Gráfico 2: Variância em x
ax2.set_aspect('equal')
ax2.axis('off')
ax2.scatter(X[0, :], X[1, :], color='red', edgecolor='black', linewidth=1)
ax2.plot([0, 0], [-5, 5], color='black')
ax2.plot([-5, 5], [0, 0], color='black')
ax2.plot([-1, 1], [0, 0], color='blue', linewidth=5)  # variância x
ax2.plot([1, 1], [4, 0], ':k')
ax2.plot([-1, -1], [-4, 0], ':k')
ax2.annotate('variância da coordenada x', xy=(-0.5, 0), xytext=(-4, 2),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()


# Segunda parte: covariância positiva e nula
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Dados com alta covariância (x sobe, y também)
Xc = np.zeros((2, n))
Xc[0, :] = np.random.uniform(-1, 1, size=n)
Xc[1, :] = Xc[0, :] + np.random.uniform(-0.15, 0.15, size=n)
Xc[0, 0] = 0.95
Xc[1, 0] = 0.8
Xc[0, 1] = -0.1
Xc[1, 1] = 0.1

ax1.axis('off')
ax1.set_aspect('equal')
ax1.scatter(Xc[0, :], Xc[1, :], color='red', edgecolor='black', linewidth=1)
ax1.plot([0, 0], [-1, 1], color='black')
ax1.plot([-1, 1], [0, 0], color='black')
ax1.set_title(r'$cov(x, y) \approx 1$')

# Anotações
ax1.text(Xc[0, 0] + 0.1, Xc[1, 0], r'$(x\uparrow, y\uparrow)$')
ax1.plot([Xc[0, 0], Xc[0, 0], 0], [0, Xc[1, 0], Xc[1, 0]], '--k')

ax1.text(Xc[0, 1] - 0.4, Xc[1, 1], r'$(x\downarrow, y\downarrow)$')
ax1.plot([Xc[0, 1], Xc[0, 1], 0], [0, Xc[1, 1], Xc[1, 1]], '--k')

# Dados com covariância próxima de zero (x e y independentes)
Xnc = np.zeros((2, n))
Xnc[0, :] = np.random.uniform(-1, 1, size=n)
Xnc[1, :] = np.random.uniform(-1, 1, size=n)
Xnc[0, 0] = 0.95
Xnc[1, 0] = 0.1

ax2.axis('off')
ax2.set_aspect('equal')
ax2.scatter(Xnc[0, :], Xnc[1, :], color='red', edgecolor='black', linewidth=1)
ax2.plot([0, 0], [-1, 1], color='black')
ax2.plot([-1, 1], [0, 0], color='black')
ax2.set_title(r'$cov(x, y) \approx 0$')

# Anotação
ax2.text(Xnc[0, 0] + 0.1, Xnc[1, 0], r'$(x\uparrow, y\downarrow)$')
ax2.plot([Xnc[0, 0], Xnc[0, 0], 0], [0, Xnc[1, 0], Xnc[1, 0]], '--k')

plt.tight_layout()
plt.show()
```

<figure><img src=".gitbook/assets/image (22).png" alt=""><figcaption><p>Figura 1 -</p></figcaption></figure>

<figure><img src=".gitbook/assets/image (23).png" alt=""><figcaption><p>Figura 2 - </p></figcaption></figure>

**Matriz de correlação para remoção de atributos:**

* Valores negativos indicam correlação inversa
* Tem diagonal igual a 1 (auto-correlação)
* É simétrica

```python
import pandas as pd
import numpy as np

# Simulando um DataFrame semelhante ao Boston Housing
np.random.seed(42)
data = pd.DataFrame(
    np.random.rand(100, 13),  # 100 amostras, 13 atributos
    columns=[f"Feature_{i}" for i in range(13)]
)

# Introduzindo uma correlação artificial entre Feature_0 e Feature_1
data["Feature_1"] = 2 * data["Feature_0"] + np.random.normal(0, 0.1, size=100)

# Calculando e exibindo a matriz de correlação
print("Nível de correlação entre as variáveis dos dados simulados:")
correlation_matrix = data.corr()
print(correlation_matrix)

```

```
Nível de correlação entre as variáveis dos dados simulados:
            Feature_0  Feature_1  Feature_2  Feature_3  Feature_4  Feature_5  \
Feature_0    1.000000   0.986205   0.074976  -0.099773  -0.045426   0.046724   
Feature_1    0.986205   1.000000   0.080201  -0.083691  -0.060579   0.039408   
Feature_2    0.074976   0.080201   1.000000   0.156992   0.143122   0.047534   
Feature_3   -0.099773  -0.083691   0.156992   1.000000  -0.083380   0.170759   
Feature_4   -0.045426  -0.060579   0.143122  -0.083380   1.000000   0.132162   
Feature_5    0.046724   0.039408   0.047534   0.170759   0.132162   1.000000   
Feature_6   -0.044283  -0.054725  -0.222581   0.005057  -0.110860   0.078206   
Feature_7   -0.093526  -0.104375  -0.243983   0.023449   0.066312   0.008497   
Feature_8   -0.001011   0.017353   0.081767  -0.072069  -0.016520  -0.099115   
Feature_9    0.114260   0.124649  -0.099356  -0.054333  -0.008457   0.151662   
Feature_10   0.130473   0.128135   0.119545  -0.074409   0.166489   0.003329   
Feature_11  -0.002073  -0.012977  -0.084338  -0.066257   0.106652   0.050698   
Feature_12  -0.125141  -0.115236   0.072257   0.030996  -0.001189  -0.102866   

            Feature_6  Feature_7  Feature_8  Feature_9  Feature_10  \
Feature_0   -0.044283  -0.093526  -0.001011   0.114260    0.130473   
Feature_1   -0.054725  -0.104375   0.017353   0.124649    0.128135   
Feature_2   -0.222581  -0.243983   0.081767  -0.099356    0.119545   
Feature_3    0.005057   0.023449  -0.072069  -0.054333   -0.074409   
Feature_4   -0.110860   0.066312  -0.016520  -0.008457    0.166489   
Feature_5    0.078206   0.008497  -0.099115   0.151662    0.003329   
Feature_6    1.000000   0.084960  -0.180060  -0.032910   -0.103416   
Feature_7    0.084960   1.000000  -0.016908  -0.102249   -0.136430   
Feature_8   -0.180060  -0.016908   1.000000   0.020813   -0.032728   
Feature_9   -0.032910  -0.102249   0.020813   1.000000   -0.013110   
Feature_10  -0.103416  -0.136430  -0.032728  -0.013110    1.000000   
Feature_11   0.032264   0.078094  -0.252941  -0.060039   -0.058979   
Feature_12   0.015333  -0.024576  -0.007647  -0.000016   -0.089531   

            Feature_11  Feature_12  
Feature_0    -0.002073   -0.125141  
Feature_1    -0.012977   -0.115236  
Feature_2    -0.084338    0.072257  
Feature_3    -0.066257    0.030996  
Feature_4     0.106652   -0.001189  
Feature_5     0.050698   -0.102866  
Feature_6     0.032264    0.015333  
Feature_7     0.078094   -0.024576  
Feature_8    -0.252941   -0.007647  
Feature_9    -0.060039   -0.000016  
Feature_10   -0.058979   -0.089531  
Feature_11    1.000000   -0.075725  
Feature_12   -0.075725    1.000000
```

```python
import pandas as pd
import numpy as np

# Simulação de um DataFrame com estrutura semelhante ao Boston Housing
np.random.seed(42)
data = pd.DataFrame(
    np.random.rand(100, 13),
    columns=[f"Feature_{i}" for i in range(13)]
)

# Criando correlação artificial entre algumas colunas
data["Feature_1"] = data["Feature_0"] * 0.95 + np.random.normal(0, 0.02, size=100)
data["Feature_5"] = data["Feature_2"] * -0.9 + np.random.normal(0, 0.05, size=100)

# Calculando a matriz de correlação
corr = data.corr()

# Definindo o limite máximo de correlação aceitável
corr_maxima = 0.8

# Criando a máscara da matriz superior (sem diagonais nem repetições)
matriz_superior = np.triu(np.ones(corr.shape), k=1).astype(bool)
correlacao_superior = corr.where(matriz_superior)

# Exibindo a matriz superior
print("Matriz superior com correlação:")
print(correlacao_superior)

# Identificando colunas com correlação acima do limite
atributos_correlacionados = [
    coluna for coluna in correlacao_superior.columns
    if any(correlacao_superior[coluna].abs() >= corr_maxima)
]

print("\n\nAtributos com alta correlação:")
print(atributos_correlacionados)

# Removendo os atributos correlacionados
data_sem_correlacao = data.drop(columns=atributos_correlacionados)

print("\n\nAtributos não correlacionados (primeiras linhas):")
print(data_sem_correlacao.head())

```

```
Matriz superior com correlação:
            Feature_0  Feature_1  Feature_2  Feature_3  Feature_4  Feature_5  \
Feature_0         NaN   0.997463   0.074976  -0.099773  -0.045426  -0.044079   
Feature_1         NaN        NaN   0.077477  -0.093196  -0.052097  -0.048928   
Feature_2         NaN        NaN        NaN   0.156992   0.143122  -0.979550   
Feature_3         NaN        NaN        NaN        NaN  -0.083380  -0.164226   
Feature_4         NaN        NaN        NaN        NaN        NaN  -0.127009   
Feature_5         NaN        NaN        NaN        NaN        NaN        NaN   
Feature_6         NaN        NaN        NaN        NaN        NaN        NaN   
Feature_7         NaN        NaN        NaN        NaN        NaN        NaN   
Feature_8         NaN        NaN        NaN        NaN        NaN        NaN   
Feature_9         NaN        NaN        NaN        NaN        NaN        NaN   
Feature_10        NaN        NaN        NaN        NaN        NaN        NaN   
Feature_11        NaN        NaN        NaN        NaN        NaN        NaN   
Feature_12        NaN        NaN        NaN        NaN        NaN        NaN   

            Feature_6  Feature_7  Feature_8  Feature_9  Feature_10  \
Feature_0   -0.044283  -0.093526  -0.001011   0.114260    0.130473   
Feature_1   -0.048924  -0.098509   0.006883   0.119116    0.129911   
Feature_2   -0.222581  -0.243983   0.081767  -0.099356    0.119545   
Feature_3    0.005057   0.023449  -0.072069  -0.054333   -0.074409   
Feature_4   -0.110860   0.066312  -0.016520  -0.008457    0.166489   
Feature_5    0.206662   0.257485  -0.102833   0.081458   -0.094889   
Feature_6         NaN   0.084960  -0.180060  -0.032910   -0.103416   
Feature_7         NaN        NaN  -0.016908  -0.102249   -0.136430   
Feature_8         NaN        NaN        NaN   0.020813   -0.032728   
Feature_9         NaN        NaN        NaN        NaN   -0.013110   
Feature_10        NaN        NaN        NaN        NaN         NaN   
Feature_11        NaN        NaN        NaN        NaN         NaN   
Feature_12        NaN        NaN        NaN        NaN         NaN   

            Feature_11  Feature_12  
Feature_0    -0.002073   -0.125141  
Feature_1    -0.006769   -0.121306  
Feature_2    -0.084338    0.072257  
Feature_3    -0.066257    0.030996  
Feature_4     0.106652   -0.001189  
Feature_5     0.077574   -0.084296  
Feature_6     0.032264    0.015333  
Feature_7     0.078094   -0.024576  
Feature_8    -0.252941   -0.007647  
Feature_9    -0.060039   -0.000016  
Feature_10   -0.058979   -0.089531  
Feature_11         NaN   -0.075725  
Feature_12         NaN         NaN  


Atributos com alta correlação:
['Feature_1', 'Feature_5']


Atributos não correlacionados (primeiras linhas):
   Feature_0  Feature_2  Feature_3  Feature_4  Feature_6  Feature_7  \
0   0.374540   0.731994   0.598658   0.156019   0.058084   0.866176   
1   0.212339   0.183405   0.304242   0.524756   0.291229   0.611853   
2   0.199674   0.592415   0.046450   0.607545   0.065052   0.948886   
3   0.440152   0.495177   0.034389   0.909320   0.662522   0.311711   
4   0.939499   0.597900   0.921874   0.088493   0.045227   0.325330   

   Feature_8  Feature_9  Feature_10  Feature_11  Feature_12  
0   0.601115   0.708073    0.020584    0.969910    0.832443  
1   0.139494   0.292145    0.366362    0.456070    0.785176  
2   0.965632   0.808397    0.304614    0.097672    0.684233  
3   0.520068   0.546710    0.184854    0.969585    0.775133  
4   0.388677   0.271349    0.828738    0.356753    0.280935
```

## **5 - Normalizando e padronizando os dados:**

Vamos considerar o cenário de comparar cidades no Brasil (dados fictícios):

* Temperatura: -5°C até 43°C
* Umidade: 15% até 80%

Poderíamos ter algo assim:

| Cidade                | Temperatura | Umidade |
| --------------------- | ----------- | ------- |
| São Paulo             | 21          | 70      |
| São José do Rio Preto | 34          | 40      |
| Santos                | 30          | 80      |

As variáveis temperatura e umidade possuem intervalos diferentes:

* Temperatura: 21 a 34
* Umidade: 40 a 80

**Será que elas possuem a mesma magnitude?**

**Variáveis de maior magnitude tendem a dominar as de menor magnitude estatisticamente.** Assim, a variável umidade irá prevalecer sobre a temperatura durante o aprendizado em modelos preditivos.

```python
import pandas as pd
from sklearn.datasets import load_iris

# Carregando o dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Exibindo mínimo, máximo e variação para cada atributo
for col in X.columns:
    min_val = X[col].min()
    max_val = X[col].max()
    variation = max_val - min_val
    print(f"{col}: Mínimo = {min_val:.2f}, Máximo = {max_val:.2f}, Variação = {variation:.2f}")

```

```
sepal length (cm): Mínimo = 4.30, Máximo = 7.90, Variação = 3.60
sepal width (cm): Mínimo = 2.00, Máximo = 4.40, Variação = 2.40
petal length (cm): Mínimo = 1.00, Máximo = 6.90, Variação = 5.90
petal width (cm): Mínimo = 0.10, Máximo = 2.50, Variação = 2.40
```

**Normalização Min-Max.** Normaliza o atributo para estar compreendido no intervalo $\[a,b]$:

$$x' = a+\frac{[x-\min(x)](b-a)}{\max(x)-\min(x)}$$

Assumindo o intervalo $\[0,1]$:

$$x' = \frac{x-\min(x)}{\max(x)-\min(x)}$$

**Normalização pela Média.** Normaliza o atributo centralizando-o em relação à média, em que $\bar{x}$ é a média entre todos os valores de $x$:

$$x' = \frac{x-\bar{x}}{\max(x)-\min(x)}$$

**Padronização Z-Score.** Normaliza o atributo centralizando-o em relação à média ($\bar{x}$), em que o desvio padrão ($\sigma$) é unitário:

$$x' = \frac{x-\bar{x}}{\sigma},$$

```python
import pandas as pd
from sklearn.datasets import load_iris

# Carregando o dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Selecionando o atributo 0 (sepal length por padrão)
atributo = X.columns[0]
col = X[atributo]

# Estatísticas básicas
media = col.mean()
desvio = col.std()
min_val = col.min()
max_val = col.max()

print(f"Valores originais para o atributo '{atributo}':")
print(f"Mínimo: {min_val:.2f}; Máximo: {max_val:.2f}")
print(f"Variação: {max_val - min_val:.2f}")
print(f"Média: {media:.2f}")
print(f"Desvio padrão: {desvio:.2f}")

# Normalização Min-Max [0, 1]
X_minmax = (col - min_val) / (max_val - min_val)
print("\nNormalização Min-Max [0,1]:")
print(f"Mínimo: {X_minmax.min():.2f}; Máximo: {X_minmax.max():.2f}")
print(f"Variação: {X_minmax.max() - X_minmax.min():.2f}")
print(f"Média: {X_minmax.mean():.2f}")
print(f"Desvio padrão: {X_minmax.std():.2f}")

# Normalização pela média (centraliza em zero, escala pela amplitude)
X_media = (col - media) / (max_val - min_val)
print("\nNormalização pela Média [-1,1]:")
print(f"Mínimo: {X_media.min():.2f}; Máximo: {X_media.max():.2f}")
print(f"Variação: {X_media.max() - X_media.min():.2f}")
print(f"Média: {X_media.mean():.2f}")
print(f"Desvio padrão: {X_media.std():.2f}")

# Padronização z-score
X_zscore = (col - media) / desvio
print("\nPadronização z-score:")
print(f"Mínimo: {X_zscore.min():.2f}; Máximo: {X_zscore.max():.2f}")
print(f"Variação: {X_zscore.max() - X_zscore.min():.2f}")
print(f"Média: {X_zscore.mean():.2f}")
print(f"Desvio padrão: {X_zscore.std():.2f}")
```

```
Valores originais para o atributo 'sepal length (cm)':
Mínimo: 4.30; Máximo: 7.90
Variação: 3.60
Média: 5.84
Desvio padrão: 0.83

Normalização Min-Max [0,1]:
Mínimo: 0.00; Máximo: 1.00
Variação: 1.00
Média: 0.43
Desvio padrão: 0.23

Normalização pela Média [-1,1]:
Mínimo: -0.43; Máximo: 0.57
Variação: 1.00
Média: -0.00
Desvio padrão: 0.23

Padronização z-score:
Mínimo: -1.86; Máximo: 2.48
Variação: 4.35
Média: -0.00
Desvio padrão: 1.00
```

**Observações:**

* Quando o desvio padrão do atributo é pequeno, uma boa estratégia é fazer a normalização dos dados.
* A padronização z-score garante média igual a 0 e desvio padrão igual a 1.
* Deve-se aplicar o mesmo processo em todos os atributos para que fiquem na mesma magnitude.
* Um problema recorrente em normalização é a perda de significado para interpretação da variável. Por exemplo, altura e peso negativos após a normalização parecem incorretos, mas a nova projeção é justificada para ser utilizada em modelos preditivos.

## **6 - Contornando dados desbalanceados:**

Alguns modelos de aprendizado tendem a ponderar para categorias que possuem mais exemplos, ou seja, para a classe majoritária. Assim, o desbalanceamento pode prejudicar severamente o aprendizado dos modelos preditivos.

```python
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Gerando um conjunto de dados artificial com 3 classes desequilibradas
X, y = make_classification(
    n_samples=5000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.01, 0.05, 0.94],  # Distribuição desigual
    class_sep=0.8,
    random_state=0
)

# Contando quantas amostras existem por classe
classes, counts = np.unique(y, return_counts=True)

print("Distribuição de amostras por classe:")
for c, count in zip(classes, counts):
    print(f"Classe {c}: {count} amostras ({100 * count / len(y):.2f}%)")

# (Opcional) Visualização
plt.figure(figsize=(8, 6))
plt.title("Distribuição das Classes")
for label in classes:
    plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Classe {label}", s=10)
plt.legend()
plt.xlabel("Atributo 1")
plt.ylabel("Atributo 2")
plt.grid(True)
plt.show()
```

```
Distribuição de amostras por classe:
Classe 0: 64 amostras (1.28%)
Classe 1: 262 amostras (5.24%)
Classe 2: 4674 amostras (93.48%)
```

<figure><img src=".gitbook/assets/image (24).png" alt=""><figcaption><p>Figura 3 - </p></figcaption></figure>

**Como tratar o desbalanceamento?**

* Subamostragem, removendo exemplos da classe majoritária;
* Aumentação ou sobreamostragem, criando novos exemplos da classe minoritária;
* Fazendo combinações de ambas as abordagens.

```python
from imblearn.under_sampling import ClusterCentroids
import numpy as np

# Aplicando ClusterCentroids para balancear as classes
print("Aplicando ClusterCentroids para reduzir a classe majoritária...")
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X, y)

# Verificando a nova distribuição de classes
classes, counts = np.unique(y_resampled, return_counts=True)
print("\nDistribuição após reamostragem:")
for c, count in zip(classes, counts):
    print(f"Classe {c}: {count} amostras")
```

```
Aplicando ClusterCentroids para reduzir a classe majoritária...

Distribuição após reamostragem:
Classe 0: 64 amostras
Classe 1: 64 amostras
Classe 2: 64 amostras
```

```python
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# Aplicando RandomOverSampler para aumentar as classes minoritárias
print("Aumentando os dados das classes minoritárias com RandomOverSampler...")
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Contando a nova distribuição das classes
classes, counts = np.unique(y_resampled, return_counts=True)
print("\nDistribuição após oversampling:")
for c, count in zip(classes, counts):
    print(f"Classe {c}: {count} amostras")
```

```
Aumentando os dados das classes minoritárias com RandomOverSampler...

Distribuição após oversampling:
Classe 0: 4674 amostras
Classe 1: 4674 amostras
Classe 2: 4674 amostras
```

```python
from imblearn.combine import SMOTEENN
import numpy as np

print("Aplicando SMOTE + ENN para balancear e limpar o dataset...")
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Contando as amostras por classe após reamostragem
classes, counts = np.unique(y_resampled, return_counts=True)
print("\nDistribuição após SMOTEENN:")
for c, count in zip(classes, counts):
    print(f"Classe {c}: {count} amostras")

```

```
Aplicando SMOTE + ENN para balancear e limpar o dataset...

Distribuição após SMOTEENN:
Classe 0: 4060 amostras
Classe 1: 4381 amostras
Classe 2: 3502 amostras
```

```python
from imblearn.combine import SMOTETomek
import numpy as np


# Aplicando SMOTE + Tomek Links
print("Aplicando SMOTE + Tomek Links para balancear e limpar o dataset...")
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Exibindo a nova distribuição de classes
classes, counts = np.unique(y_resampled, return_counts=True)
print("\nDistribuição após SMOTE Tomek:")
for c, count in zip(classes, counts):
    print(f"Classe {c}: {count} amostras")
```

```
Aplicando SMOTE + Tomek Links para balancear e limpar o dataset...

Distribuição após SMOTE Tomek:
Classe 0: 4499 amostras
Classe 1: 4566 amostras
Classe 2: 4413 amostras
```

## **7 - Codificando dados categóricos:**

Variáveis categóricas não são diretamente aplicadas em modelos estatísticos e computacionais. Assim, discretizá-las é um meio de aplicar estes dados nos modelos.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Carregando os dados diretamente da URL
url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
data = pd.read_csv(url)

# Visão geral do dataset
print("Visão geral dos dados:")
print(data.head())

# Valores únicos do atributo 'thal'
print("\nValores únicos no atributo 'thal':")
print(data['thal'].unique())

# Verificando se há valores nulos em 'thal'
print("\nValores ausentes em 'thal':", data['thal'].isnull().sum())

# Histograma do atributo 'thal'
print("\nDistribuição de valores no atributo 'thal':")
plt.figure(figsize=(6,4))
data['thal'].hist(edgecolor='black', grid=False)
plt.title("Distribuição do atributo 'thal'")
plt.xlabel("Categoria (thal)")
plt.ylabel("Frequência")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

```
Visão geral dos dados:
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \
0   63    1   1       145   233    1        2      150      0      2.3      3   
1   67    1   4       160   286    0        2      108      1      1.5      2   
2   67    1   4       120   229    0        2      129      1      2.6      2   
3   37    1   3       130   250    0        0      187      0      3.5      3   
4   41    0   2       130   204    0        2      172      0      1.4      1   

   ca        thal  target  
0   0       fixed       0  
1   3      normal       1  
2   2  reversible       0  
3   0      normal       0  
4   0      normal       0  

Valores únicos no atributo 'thal':
['fixed' 'normal' 'reversible' '1' '2']

Valores ausentes em 'thal': 0

Distribuição de valores no atributo 'thal':
```

<figure><img src=".gitbook/assets/image (25).png" alt=""><figcaption><p>Figura 4 -</p></figcaption></figure>

```python
import matplotlib.pyplot as plt

# Verificando valores únicos antes da filtragem
print("Valores originais no atributo 'thal':", data['thal'].unique())

# Removendo valores discrepantes (strings '1' e '2')
# Garantindo que a coluna seja string para evitar bugs com tipos mistos
data['thal'] = data['thal'].astype(str)
data = data[(data['thal'] != '1') & (data['thal'] != '2')]

# Exibindo os valores restantes
print("\nValores restantes no atributo 'thal' após remoção de '1' e '2':")
print(data['thal'].unique())

# Visualizando distribuição
print("\nDistribuição do atributo 'thal' após limpeza:")
plt.figure(figsize=(6, 4))
data['thal'].hist(edgecolor='black')
plt.title("Distribuição de 'thal' após remover categorias '1' e '2'")
plt.xlabel("thal")
plt.ylabel("Frequência")
plt.grid(False)
plt.tight_layout()
plt.show()
```

```
Valores originais no atributo 'thal': ['fixed' 'normal' 'reversible']

Valores restantes no atributo 'thal' após remoção de '1' e '2':
['fixed' 'normal' 'reversible']

Distribuição do atributo 'thal' após limpeza:
```

<figure><img src=".gitbook/assets/image (26).png" alt=""><figcaption><p>Figura 5 -</p></figcaption></figure>

Uma forma de discretizar variáveis categórias é por meio de uma escala direta de valores. Por exemplo:

* fixed = 1
* normal = 2
* reversible = 3

**Qual o problema de utilizarmos esta abordagem?**

A solução mais viável é transformar cada valor existente em uma variável binária, indicando ou não a presença deste valor:

```python
import pandas as pd

# Supondo que 'data' já foi carregado e filtrado
# Transformando o atributo 'thal' em variáveis dummies (one-hot encoding)
dummy_vars = pd.get_dummies(data['thal'], prefix='thal')

# Concatenando as novas colunas dummy com a original (para fins de visualização)
newData = pd.concat([data['thal'], dummy_vars], axis=1)

# Exibindo o resultado
print("Atributo 'thal' convertido em variáveis dummies:")
print(newData.head())

```

```
Atributo 'thal' convertido em variáveis dummies:
         thal  thal_fixed  thal_normal  thal_reversible
0       fixed        True        False            False
1      normal       False         True            False
2  reversible       False        False             True
3      normal       False         True            False
4      normal       False         True            False
```

Assim, ao invés de considerarmos o atributo "thal", consideramos somente suas respectivas codificações.

Este método também chamado de **one-hot encoding**!

## **8 - Identificando Outliers:**

**Outliers, pontos "fora-da-curva" ou pontos aberrantes** são exemplos ou instâncias que:

* Dentre do espaço de possíveis valores, recaem num intervalo _fora_ daquele relativo a maior parte dos exemplos de uma base de dados;
* Desvia tanto das outras observações que levanta suspeita de que foi gerada por um mecanismo diferenciado;
* São meramente uma manifestação extrema da variedade aleatória inerente aos dados;
* São resultados de um desvio grosseiro do procedimento experimental.

```python
import pandas as pd               
import matplotlib.pyplot as plt    
from sklearn.datasets import fetch_california_housing  

# Carregando o dataset California Housing
# O parâmetro 'as_frame=True' retorna os dados como um DataFrame do pandas, facilitando a manipulação
california = fetch_california_housing(as_frame=True)

# Extraindo os dados em um único DataFrame
# Esse DataFrame contém as variáveis independentes (características das casas) e a variável alvo (valor médio das casas)
data = california.frame
print(data.head())
```

```
MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   

   Longitude  MedHouseVal  
0    -122.23        4.526  
1    -122.22        3.585  
2    -122.24        3.521  
3    -122.25        3.413  
4    -122.25        3.422
```

```python
import seaborn as sns

# Vamos analisar a variável alvo: preço médio das casas (em milhares de dólares)
precos = data['MedHouseVal']

plt.figure(figsize=(8, 5))
sns.histplot(precos, bins=50, kde=True, color='skyblue')
plt.title("Distribuição dos preços das casas (MedHouseVal)")
plt.xlabel("Preço em milhares de dólares")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()
```

<figure><img src=".gitbook/assets/image (10).png" alt=""><figcaption><p>Figura 6 -</p></figcaption></figure>

## **8.1 - Detecção de outliers pela Distribuição dos Dados:**

Estes métodos assumem uma determinada distribuição de probabilidade ou modelo estatístico para o conjunto de dados, sendo possível **aplicar diversos testes estatísticos para diferenciar e identificar as amostras que mais desviam da distribuição assumida**.

Os dados normais serão aqueles que se situam em regiões de alta probabilidade do modelo e **outliers são as observações que desviam fortemente da distribuição assumida no modelo adotado**.

**Desvio Padrão (medida de dispersão):**

Desvio padrão amostral (_standard deviation_), em que $\mu$ é a média dos valores da variável:

$$\sigma = \frac{\sqrt{ \sum_i (x_i - \mu)^2}}{n-1}$$

```python
# Variável alvo: preço médio das casas (em milhares de dólares)
precos = data['MedHouseVal']

# Cálculo da média e desvio padrão da variável
media = precos.mean()
desvp = precos.std()

print("Média = %.2f, Desvio padrão = %.2f" % (media, desvp))

# Identificando outliers: valores fora do intervalo [média - 2*dp, média + 2*dp]
outliers = precos[(precos < media - 2*desvp) | (precos > media + 2*desvp)]
inliers = precos[(precos >= media - 2*desvp) & (precos <= media + 2*desvp)]

# Exibindo informações no console
print("Número total de dados:", len(precos))
print("Número de outliers (média ± 2*desvio):", len(outliers))
print("Número de inliers:", len(inliers))
print("Limite inferior para outliers: %.2f" % (media - 2*desvp))
print("Limite superior para outliers: %.2f" % (media + 2*desvp))
```

```
Média = 2.07, Desvio padrão = 1.15
Número total de dados: 20640
Número de outliers (média ± 2*desvio): 1383
Número de inliers: 19257
Limite inferior para outliers: -0.24
Limite superior para outliers: 4.38
```

**Amplitude interquartil (IQR,&#x20;**_**interquartile range**_**):**

* $Q\_{1}$ o valor relativo aos primeiros 25% dados,
* $Q\_{2}$ o valor relativo aos primeiros 50% dados (mediana),
* $Q\_{3}$ o valor relativo aos primeiros 75% dos dados,

$$IQR = Q_{3} - Q_{1}$$

```python
# Variável alvo: preço médio das casas (em milhares de dólares)
precos = data['MedHouseVal']

# Cálculo do IQR para identificar outliers
Q1 = precos.quantile(0.25)  # 1º quartil
Q3 = precos.quantile(0.75)  # 3º quartil
IQR = Q3 - Q1               # Intervalo interquartil

# Limites inferior e superior para detecção de outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Separando inliers e outliers com base nos limites
inliers = precos[(precos >= limite_inferior) & (precos <= limite_superior)]
outliers = precos[(precos < limite_inferior) | (precos > limite_superior)]

# Imprimindo informações no console
print(f"Número total de dados: {len(precos)}")
print(f"Número de inliers: {len(inliers)}")
print(f"Número de outliers: {len(outliers)}")
print(f"Limite inferior para outliers: {limite_inferior:.3f}")
print(f"Limite superior para outliers: {limite_superior:.3f}")
```

```
Número total de dados: 20640
Número de inliers: 19569
Número de outliers: 1071
Média dos preços: 2.069
Desvio padrão dos preços: 1.154
Limite inferior para outliers: -0.981
Limite superior para outliers: 4.824
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plotando histograma + KDE da distribuição dos preços (inliers)
plt.figure(figsize=(10, 6))
sns.histplot(inliers, bins=50, kde=True, color='skyblue', label='Inliers')

# Plotando os outliers como pontos vermelhos na parte inferior do gráfico
plt.scatter(outliers, np.full_like(outliers, -5), color='red', label='Outliers', marker='x', zorder=5)

# Marcando a média (linha verde)
plt.axvline(media, color='green', linestyle='--', linewidth=2, label='Média')

# Marcando a média + 1 desvio padrão (linha azul pontilhada)
plt.axvline(media + desvio, color='blue', linestyle=':', linewidth=2, label='Média + 1 DP')

# Marcando a média - 1 desvio padrão (linha azul pontilhada)
plt.axvline(media - desvio, color='blue', linestyle=':', linewidth=2, label='Média - 1 DP')

# Marcando limites inferior e superior de outliers (linhas laranja tracejadas)
plt.axvline(limite_inferior, color='orange', linestyle='--', linewidth=2, label='Limite inferior outlier')
plt.axvline(limite_superior, color='orange', linestyle='--', linewidth=2, label='Limite superior outlier')

# Configurações adicionais do gráfico
plt.title("Distribuição dos preços das casas com Inliers, Outliers, Média e Desvio Padrão")
plt.xlabel("Preço em milhares de dólares")
plt.ylabel("Frequência")
plt.legend()
plt.grid(True)
plt.show()
```

<figure><img src=".gitbook/assets/image (11).png" alt=""><figcaption><p>Figura 7 - </p></figcaption></figure>

## **9 - Redução de Dimensionalidade:**

**Ideia geral dos Métodos de Redução de Dimensionalidade:**

Métodos de redução de dimensionalidade objetivam encontrar **transformações das covariáveis originais**, que capturam parte considerável das informações presentes, de modo a reduzir redundâncias e a quantidade destas.

Consequentemente, essas técnicas criam uma quantidade reduzida de variáveis:

* $X = \{{X\_1, X\_2, …, X\_d\}}$, sendo o espaço de características original com dimensionalidade $d$
* $Z = \{{Z\_1, Z\_2, …, Z\_m\}}$, sendo o espaço de características resultante com dimensionalidade $m, (m < d)$.

## **9.1 - Principal Component Analysis (PCA):**

PCA é um técnica de **redução de dimensionalidade não supervisionada** que mapeia o espaço original de atributos em outro espaço, de dimensão inferior, por meio de combinações lineares das covariáveis originais:

* Elimina a redundância e preserva informações importantes;
* Permite selecionar uma quantidade de dimensões (atributos) desejável;
* Mantem grande parte da variância dos dados;
* Proporciona um novo conjunto de atributos em que as variáveis não são correlacionadas e são ortogonais uma às outras.

**Observação:**

* Na teoria, o PCA transforma $d$ variáveis correlacionadas em $d$ variáveis não correlacionadas;
* Ao selecionar $m$ variáveis $(m < d)$, as variáveis com menos variância são descartadas, ocorrendo a redução da dimensionalidade.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# Carregando o dataset Iris
iris = load_iris()
X = iris.data
Y = iris.target

# Informações sobre o conjunto de dados
print("Conjunto de dados IRIS:", X.shape)
print("Conjunto de rótulos IRIS:", Y.shape)
print("Classes existentes:", np.unique(Y))

# Matriz de correlação entre as variáveis
df = pd.DataFrame(X, columns=iris.feature_names)
corr = df.corr()

# Visualização com mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação - Iris")
plt.show()
```

```
Conjunto de dados IRIS: (150, 4)
Conjunto de rótulos IRIS: (150,)
Classes existentes: [0 1 2]
```

<figure><img src=".gitbook/assets/image (12).png" alt=""><figcaption><p>Figura 8 - </p></figcaption></figure>

```python
from sklearn.decomposition import PCA, IncrementalPCA

# Redução de dimensionalidade com PCA (Principal Component Analysis)
# PCA tradicional: carrega todos os dados na memória
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Shape dos dados após PCA:", X_pca.shape)

# IPCA (Incremental PCA): versão que processa os dados em partes (útil para grandes volumes)
ipca = IncrementalPCA(n_components=2, batch_size=10)
X_ipca = ipca.fit_transform(X)
print("Shape dos dados após Incremental PCA:", X_ipca.shape)

```

```
Shape dos dados após PCA: (150, 2)
Shape dos dados após Incremental PCA: (150, 2)
```

```python
# Comparando a visualização dos dados: original, após PCA e após IPCA
plt.figure(figsize=(18, 6))
colors = ['red', 'green', 'blue']

# Espaço original (atributos 0 e 1)
plt.subplot(1, 3, 1)
plt.title("Original")
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[Y[i]])

# Espaço transformado com PCA
plt.subplot(1, 3, 2)
plt.title("PCA")
for i in range(X_pca.shape[0]):
    plt.scatter(X_pca[i][0], X_pca[i][1], c=colors[Y[i]])

# Espaço transformado com IPCA
plt.subplot(1, 3, 3)
plt.title("IPCA")
for i in range(X_ipca.shape[0]):
    plt.scatter(X_ipca[i][0], X_ipca[i][1], c=colors[Y[i]])

plt.show()
```

<figure><img src=".gitbook/assets/image (13).png" alt=""><figcaption><p>Figura 9 -</p></figcaption></figure>

## **Resumo:**

Tipos de Dados:

* Dados não estruturados;
* Dados estruturados;
* Coleta de Dados;

Informações faltantes:

* Erros no protocolo de coleta de dados e na sua execução;
* Omissão ou não preenchimento de campos;
* Possíveis ações:
  * Remover linhas/colunas que contenham dados faltantes;
  * Preencher os valores faltantes por alguma heurística;

Dados redundantes:

* Presença de cópias, colunas com valores constantes e/ou alta correlação;
* Possíveis ações:
  * Remoção de linhas e/ou colunas;
  * Descoberta de atributos correlacionados;

Normalização e Padronização:

* MinMax;
* Média;
* Z-Score;

Dados desbalanceados:

* Uma categoria se sobressai a outras em quantidades;
* Possíveis ações:
  * Redução da classe majoritária;
  * Aumento das classes minoritárias.
  * Combinação dos métodos;

Codificação de Dados.

Identificação de Outliers:

* Desvio padrão;
* Amplitude interquartil.

Redução de Dimensionalidade:

* PCA.

## **Considerações finais:**

* Uma das vantagens dos Métodos baseados em Distribuições (outliers) é que **se as proposições relativas à distribuição de dados forem verdadeiras**, as técnicas constituem uma estratégia eficiente.
* Os **aspectos negativos** dos Métodos baseados em distribuição se devem à falta de robustez, pois apenas um outlier pode impactar significativamente nas medidas estatísticas da distribuição em uso.
* Estas técnicas **são sensíveis à quantidade de instâncias no conjunto de dados**. Quanto maior o número de registros, maior representatividade estatística a amostra terá.
* Os modelos estatísticos normalmente **são adequados para conjuntos de dados quantitativos de valor real**. Os dados ordinais podem ser transformados em valores numéricos se adequando ao processamento estatístico. Isso limita a aplicabilidade e aumenta o tempo de processamento, caso um pré-processamento dos dados seja necessário.
* É importante **identificar a distribuição de forma correta**; caso contrário a análise de dados pode ser imprecisa.
* Métodos de redução de dimensionalidade objetivam encontrar transformações das covariáveis originais de modo a reduzir redundâncias e a quantidade destas.
* **Principal Component Analysis (PCA)** reduz a quantidade de representações descartando as combinações lineares que possuem pequenas variâncias, retendo as que possuem alta variabilidade.

## **Bibliografia básica:**

* Duda, Hart, Stork. Pattern Classification (Capítulo 2).

## **Leituras complementares:**

* Documentação sklearn para tratamento de dados (https://scikit-learn.org/stable/data\_transforms.html)
