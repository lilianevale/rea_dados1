# Pandas

## Introdução ao Pandas

A biblioteca `pandas` é uma das mais poderosas ferramentas da ciência de dados em Python. Ela permite analisar, transformar, visualizar e tratar grandes conjuntos de dados com facilidade.

***

### Como importar o Pandas

```python
import pandas as pd
```

### Criando um DataFrame de exemplo

Antes de começar a explorar as funcionalidades do `pandas`, vamos criar um DataFrame simples para praticar:

```python
import pandas as pd

dados = {
    "nome": ["Ana", "Bruno", "Carlos", "Diana", "Eduardo"],
    "idade": [25, 32, 40, 28, 35],
    "sexo": ["F", "M", "M", "F", "M"],
    "salario": [3500, 4200, 5000, 3900, 4400],
    "estado": ["SP", "RJ", "MG", "SP", "RS"]
}

df = pd.DataFrame(dados)
```

Agora temos um DataFrame com as seguintes colunas:

* `nome`: nome da pessoa
* `idade`: idade em anos
* `sexo`: F para feminino, M para masculino
* `salario`: salário mensal
* `estado`: estado de origem

***

### Lendo dados de arquivos

Também é póssivel carregar dados de diferentes tipos de arquivos, como:

* Arquivos CSV
* Planilhas Excel (`.xls`, `.xlsx`)
* Arquivos de texto separados por tabulação (`.tsv`)
* Arquivos JSON
* Arquivos Parquet
* SQL (bancos de dados)
* HTML (tabelas da web)

#### Exemplos de leitura:

```python
# CSV
df = pd.read_csv("dados.csv")

# Excel
df = pd.read_excel("dados.xlsx")

# Arquivo separado por tabulação
df = pd.read_csv("dados.tsv", sep="\t")

# JSON
df = pd.read_json("dados.json")

# Parquet
df = pd.read_parquet("dados.parquet")

# SQL (usando conexão)
import sqlite3
conn = sqlite3.connect("banco.db")
df = pd.read_sql("SELECT * FROM tabela", conn)

# Tabela de uma página HTML
df_list = pd.read_html("https://exemplo.com/tabela.html")  # retorna uma lista de DataFrames
```

O objeto retornado é chamado de **DataFrame**, que é a estrutura de dados principal do `pandas`, parecida com uma tabela de banco de dados ou planilha.

***

### Explorando o DataFrame

```python
df.head()      # Mostra as 5 primeiras linhas
df.tail()      # Mostra as 5 últimas linhas
df.shape       # Retorna (linhas, colunas)
df.columns     # Lista os nomes das colunas
df.info()      # Informações gerais do DataFrame
df.describe()  # Estatísticas descritivas
```

***

### Selecionando colunas

```python
df["nome_coluna"]        # Uma coluna (como Series)
df[["col1", "col2"]]     # Múltiplas colunas (como novo DataFrame)
```

***

### Selecionando linhas

#### Por índice

```python
df.iloc[0]         # Primeira linha (por índice)
df.iloc[0:5]       # Linhas da 0 até 4
```

#### Por condição

```python
df[df["idade"] > 30]               # Linhas com idade > 30
df[df["nome"] == "João"]          # Filtra por nome
df[(df["idade"] > 20) & (df["sexo"] == "M")]  # Filtros combinados
```

***

### Removendo colunas ou linhas

```python
df.drop("coluna", axis=1, inplace=True)     # Remove coluna
df.drop([0, 1], axis=0)                     # Remove linhas por índice
```

***

### Lidando com valores ausentes (NaN)

```python
df.isna().sum()               # Conta valores ausentes por coluna
df.dropna(inplace=True)       # Remove linhas com qualquer valor ausente
df.fillna(0)                  # Substitui ausentes por zero
df.fillna(method="ffill")     # Preenche com valor anterior (forward fill)
```

***

### Criando e modificando colunas

```python
df["nova_coluna"] = df["salario"] * 0.1       # Criação
df["idade"] = df["idade"] + 1                 # Modificação
```

***

### Operações estatísticas

```python
df["salario"].mean()     # Média
df["salario"].sum()      # Soma
df["salario"].min()      # Mínimo
df["salario"].max()      # Máximo
df["salario"].std()      # Desvio padrão
```

***

### Agrupamento de dados

```python
df.groupby("sexo")["salario"].mean()    # Média de salário por sexo
df.groupby("departamento").sum()        # Soma de colunas numéricas por grupo
```

***

### Ordenando dados

```python
df.sort_values("idade")                    # Ordem crescente
df.sort_values("idade", ascending=False)   # Ordem decrescente
```

***

### Filtrando por valores únicos

```python
df["estado"].unique()         # Valores únicos
df["estado"].value_counts()   # Contagem de cada valor
```

### Salvando o DataFrame

O `pandas` permite salvar DataFrames em vários formatos de arquivo. Abaixo estão os mais comuns:

#### Exemplos de salvamento:

```python
# CSV
df.to_csv("saida.csv", index=False)

# Excel
df.to_excel("saida.xlsx", index=False)

# TSV (texto separado por tabulação)
df.to_csv("saida.tsv", sep="\t", index=False)

# JSON
df.to_json("saida.json", orient="records", lines=True)

# Parquet (formato binário eficiente)
df.to_parquet("saida.parquet", index=False)

# SQL (requer conexão com banco)
import sqlite3
conn = sqlite3.connect("banco.db")
df.to_sql("nome_da_tabela", conn, if_exists="replace", index=False)

# HTML (gera uma tabela)
df.to_html("saida.html", index=False)
```

#### Parâmetros úteis:

* `index=False`: não salva o índice como coluna
* `sep="\t"`: define separador personalizado (ex: tabulação)
* `if_exists="replace"`: substitui a tabela no banco SQL (ou "append" para adicionar)

Essas opções permitem integrar seu DataFrame com outros sistemas, planilhas, APIs, bancos de dados e mais.
