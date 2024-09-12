import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import random
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

def carregar_dados(url):
    try:
        if 'drive.google.com' in url:
            file_id = url.split('/')[-2]
            download_url = f"https://drive.google.com/uc?id={file_id}"
            response = requests.get(download_url).content
            df = pd.read_csv(StringIO(response.decode('utf-8')))
        else:
            df = pd.read_csv(url)
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

def limpeza_dados(df):
    print("\nLimpando dados...")

    # Remover duplicatas
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    print(f"Duplicatas removidas: {initial_rows - df.shape[0]}")

    # Remover colunas com todos os valores ausentes
    df = df.dropna(how='all', axis=1)

    # Preencher valores ausentes em colunas numéricas com a média
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Preencher valores ausentes em colunas categóricas com o valor mais frequente
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("Dados limpos com sucesso!")
    print("Informações do DataFrame após limpeza:")
    print(df.info())
    return df

def analise_exploratoria(df):
    print("\nPrimeiras linhas do DataFrame:")
    print(df.head())
    
    print("\nInformações sobre o DataFrame:")
    print(df.info())
    
    print("\nEstatísticas descritivas das variáveis numéricas:")
    print(df.describe())
    
    # Plotar gráficos
    plot_histograma(df)
    plot_grafico_de_barras(df)
    plot_grafico_de_pizza(df)
    plot_grafico_de_linha(df)
    plot_boxplot(df)
    plot_grafico_3d(df)
    plot_grafico_interativo(df)

def plot_histograma(df):
    print("\nHistograma:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        col = random.choice(numeric_columns)
        print(f"Plotando histograma para a coluna: {col}")
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Histograma de {col}')
        plt.show()
    else:
        print("Não há variáveis numéricas para plotar o histograma.")

def plot_grafico_de_barras(df):
    print("\nGráfico de Barras:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        col = random.choice(categorical_columns)
        print(f"Plotando gráfico de barras para a coluna: {col}")
        if df[col].nunique() > 0:
            sns.countplot(data=df, x=col)
            plt.title(f'Gráfico de Barras de {col}')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Não há dados únicos para a variável '{col}' para plotar o gráfico de barras.")
    else:
        print("Não há variáveis categóricas para plotar o gráfico de barras.")

def plot_grafico_de_pizza(df):
    print("\nGráfico de Pizza:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        col = random.choice(categorical_columns)
        print(f"Plotando gráfico de pizza para a coluna: {col}")
        counts = df[col].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
        plt.title(f'Gráfico de Pizza de {col}')
        plt.show()
    else:
        print("Não há variáveis categóricas para plotar o gráfico de pizza.")

def plot_grafico_de_linha(df):
    print("\nGráfico de Linha:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        cols = random.sample(list(numeric_columns), 2)
        print(f"Plotando gráfico de linha para as colunas: {cols[0]} e {cols[1]}")
        df[cols].plot()
        plt.title(f'Gráfico de Linha entre {cols[0]} e {cols[1]}')
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.show()
    else:
        print("Não há duas variáveis numéricas para plotar o gráfico de linha.")

def plot_boxplot(df):
    print("\nBoxplot:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        print(f"Plotando boxplot para as colunas: {', '.join(numeric_columns)}")
        sns.boxplot(data=df[numeric_columns])
        plt.title('Boxplot')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("Não há variáveis numéricas para plotar o boxplot.")

def plot_grafico_3d(df):
    print("\nGráfico 3D:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 2:
        cols = random.sample(list(numeric_columns), 3)
        print(f"Plotando gráfico 3D para as colunas: {', '.join(cols)}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]])
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
        plt.title(f'Gráfico 3D entre {cols[0]}, {cols[1]} e {cols[2]}')
        plt.show()
    else:
        print("Não há três variáveis numéricas para plotar o gráfico 3D.")

def plot_grafico_interativo(df):
    print("\nGráfico Interativo:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        cols = random.sample(list(numeric_columns), 2)
        print(f"Plotando gráfico interativo para as colunas: {cols[0]} e {cols[1]}")
        fig = px.scatter(df, x=cols[0], y=cols[1], title=f'Gráfico Interativo de {cols[0]} vs {cols[1]}')
        fig.show()
    else:
        print("Não há duas variáveis numéricas para plotar o gráfico interativo.")

def analise_especialista():
    # Aqui você pode adicionar sua análise especializada
    print("\nAnálise especializada:")
    print("Essa é a análise especializada do assunto.")
    print("Você pode adicionar suas próprias conclusões e insights aqui.")

def main(urls):
    for url in urls:
        df = carregar_dados(url)
        if df is not None:
            print(f"\nAnálise exploratória para o arquivo: {url}")
            df = limpeza_dados(df)
            analise_exploratoria(df)
            print("\n---")

    # Após a análise exploratória, realizamos uma análise especializada
    analise_especialista()

# URLs dos arquivos de dados
urls = [
    "https://drive.google.com/file/d/1Z-asiPD9n5-oJhSsWNo9sea_ylueVK2B/view?usp=drive_link", 
    "https://drive.google.com/file/d/1Hijfi4kEkXJPw55Rzh4BpBpnV3Mwjkt_/view?usp=drive_link", 
    "https://drive.google.com/file/d/1DaMedDAW0ZMYnMfjk0xw8ebkQUxxEJzL/view?usp=drive_link"
    # Adicione mais URLs aqui...
]

# Executar a análise de dados
main(urls)
