#Analise Exploratoria dos Dados (EDA)
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

#Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Avaliadores de Modelos
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("https://raw.githubusercontent.com/orlandojrps/stream/main/heart-disease.csv")

#Inspecionado as 5 primeiras observações do conjunto de dados
df.head()

#Inspecionado as 5 últimas observações do conjunto de dados
df.tail()

#Quantidade de Linhas e Colunas do conjunto de dados
df.shape

#É uma tupla
type(df.shape)

linhas, colunas = df.shape

print(f"O número de linhas do Conjunto de Dados é: {linhas}")

print(f"O número de colunas do Conjunto de Dados é: {colunas}")

#Inspecionando as informações macro do conjunto de dados
df.info()

#Inspecionando as informações estatísticas do conjunto de dados
df.describe()

df.hist(figsize=(12,12))
plt.show()

"""#### Existem dados faltantes?"""

df.isnull().sum()

"""#### "target" - Tem a doença ou não (1 - Sim, 0 = Não)"""

df["target"].dtype

df["target"].unique()

df["target"].nunique()

df['target'].value_counts()

"""#### "age" - representa a idade do paciente em anos."""

df["age"].dtype

df["age"].count()

#Menor idade
df["age"].min()

#Maior idade
df["age"].max()

#Desvio Padrão
df["age"].std()

#Mediana
df["age"].median()

#Média
df["age"].mean()

df["age"].describe()

df["age"].unique()

df["age"].nunique()

x_list = np.linspace(df["age"].min(), df["age"].max(), 100)

y_list = norm.pdf(x_list, loc=df['age'].mean(), scale=df['age'].std())

plt.plot(x_list, y_list)
df['age'].hist(density=True)

df["age"].skew()

df["age"].kurtosis()

pd.crosstab(df.target, df.age)

"""bold text#### "sex" - sexo do paciente (1 - Masculino; 0 - Feminino)"""

df["sex"].dtype

df["sex"].unique()

df["sex"].nunique()

df['sex'].value_counts()

df['sex'].value_counts().plot(kind="bar", color=["teal", "salmon"])

sns.barplot(x = "sex", y="age", hue="target", data=df)

pd.crosstab(df.target, df.sex)

"""#### "cp - chest pain" - Dor no Peito


*   0: Angina típica: dor no peito relacionada à diminuição do suprimento de sangue ao coração
*   1: Angina atípica: dor no peito não relacionada ao coração
*   2: Dor não anginosa: tipicamente espasmos esofágicos (não relacionados ao coração)
*   3: Assintomático: dor no peito sem sinais de doença


"""

df["cp"].dtype

df["cp"].count()

df["cp"].unique()

df["cp"].nunique()

#Não faz sentido!
df["cp"].describe()

#Relacionando com a Idade
pd.crosstab(df.cp, df.age)

#Relacionando com o Sexo
pd.crosstab(df.cp, df.sex)

#Relacionando com o Target
pd.crosstab(df.target, df.cp)

# Create a new crosstab and base plot
pd.crosstab(df.cp, df.target).plot(kind="bar", 
                                   figsize=(10,6), 
                                   color=["lightblue", "salmon"])

# Add attributes to the plot to make it more readable
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation = 0);

"""É interessante que a agina atípica (valor 1) afirma que não está relacionada ao coração, mas parece ter uma proporção maior de participantes com doenças cardíacas do que não.

#### "trestbps" - Pressão Arterial em Repouso



*   Qualquer coisa acima de 130-140 é normalmente motivo de preocupação
"""

df["trestbps"].dtype

df["trestbps"].count()

df["trestbps"].unique()

df["trestbps"].nunique()

df["trestbps"].min()

df["trestbps"].max()

df_trestbps_acima_130 = df[df["trestbps"] >= 130]

df_trestbps_acima_130.head()

#Relacionando com o Target
pd.crosstab(df_trestbps_acima_130.target, df_trestbps_acima_130.trestbps)

df_trestbps_abaixo_130 = df[df["trestbps"] < 130]

pd.crosstab(df_trestbps_abaixo_130.target, df_trestbps_abaixo_130.trestbps)

"""#### "chol" - Colesterol Sérico



*   Acima de 200 é motivo de preocupação

"""

df["chol"].dtype

df["chol"].unique()

df["chol"].nunique()

df["chol"].min()

df["chol"].max()

df["chol"].describe()

df_chol_acima_200 = df[df["chol"] >= 200]

pd.crosstab(df_chol_acima_200.target, df_chol_acima_200.chol)

"""

```
# This is formatted as code
```

#### "fbs" - Açucar no sangue em jejum acima de 120 (1 - Verdadeiro; 0 - False)




*   Acima de 126' mg/dL signals diabetes
"""

df["fbs"].dtype

df["fbs"].unique()

df["fbs"].nunique()

df["fbs"].value_counts()

pd.crosstab(df.target, df.fbs)

"""#### "restecg" - resultados eletrocardiográficos em repouso


*   0: Nada a notar
*   1: Anormalidade da onda ST-T
pode variar de sintomas leves a problemas graves
sinaliza batimento cardíaco não normal
*   2: 2: Hipertrofia ventricular esquerda possível ou definitiva

"""

df["restecg"].dtype

df["restecg"].unique()

df["restecg"].nunique()

df["restecg"].value_counts()

pd.crosstab(df.target, df.restecg)

"""#### "thalach" - frequência cardíaca máxima alcançada

1.   List item
2.   List item


"""

df["thalach"].dtype

df["thalach"].unique()

df["thalach"].nunique()

df["thalach"].describe()

"""#### "exang" - angina induzina por exercício (1 - sim; 0 - Não)"""

df["exang"].dtype

df["exang"].unique()

df["exang"].nunique()

df["exang"].value_counts()

pd.crosstab(df.target, df.exang)

"""#### "oldpeak" - Queda de ST induzida por exercício em relação ao repouso

*   olha para o estresse do coração durante o exercício

*   coração doentio irá estressar mais


"""

df["oldpeak"].dtype

df["oldpeak"].unique()

df["oldpeak"].nunique()

"""#### "slope" - Inclinação do segmento ST de pico do exercício



*   0: Upsloping: melhor frequência cardíaca com exercício (incomum)
*   1: Flatsloping: mudança mínima (coração saudável típico)
*   2: Downslopins: sinais de coração doentio



"""

df["slope"].dtype

df["slope"].unique()

df["slope"].nunique()

pd.crosstab(df.target, df.slope)

"""#### "ca" - número de vasos principais (0-3) coloridos por fluorosopia



*   Vaso colorido significa que o médico pode ver o sangue passando
*   Quanto mais circulação sanguínea, melhor (sem coágulos)


"""

df["ca"].dtype

df["ca"].unique()

df["ca"].nunique()

pd.crosstab(df.target, df.ca)

"""#### "thal" - resultado de estresse com tálio



*   1,3: normal
*   6: defeito corrigido: costumava ser defeito, mas agora está bem
*   7: defeito reversível: nenhum movimento sanguíneo adequado durante o exercício


"""

df["thal"].dtype

df["thal"].unique()

df["thal"].nunique()

pd.crosstab(df.target, df.thal)

"""Idade vs frequência cardíaca máxima (Max Heart rate) para doenças cardíacas"""

# Create another figure
plt.figure(figsize=(10,6))

# Start with positve examples
plt.scatter(df.age[df.target==1], 
            df.thalach[df.target==1], 
            c="salmon") # define it as a scatter figure

# Now for negative examples, we want them on the same plot, so we call plt again
plt.scatter(df.age[df.target==0], 
            df.thalach[df.target==0], 
            c="lightblue") # axis always come as (x, y)

# Add some helpful info
plt.title("Doença cardíaca em função da idade e freqüência cardíaca máxima")
plt.xlabel("Idade")
plt.legend(["Com Doença Cardíaca", "Sem Doença Cardíaca"])
plt.ylabel("Freqüência Cardíaca Máxima");

"""Quanto mais jovem alguém é, mais alta é a frequência cardíaca máxima (pontos mais altos a esquerda do gráfico)

Mas isso pode ocorrer pois há mais pontos do lado direito do gráfico (participantes mais velhos)
"""

#Oscila um pouco para a direita, o que se reflete no gráfico de dispersão acima
df["age"].plot.hist();

"""#### Correlação entre as variáveis indepentenders pode nos dar uma ideia de quais tem ou não impacto no target

#### Como cada varíavel se relaciona uma com a outra
"""

corr_matrix = df.corr()
corr_matrix

corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.5, 
            fmt= ".2f", 
            cmap="YlGnBu");

"""Muito melhor. Um valor positivo mais alto significa uma correlação positiva potencial (aumento) e um valor negativo mais alto significa uma correlação negativa potencial (diminuição).

Vamos usar modelos de machine learning para conduzir nossas próximas perguntas. 
"""

#Todas exceto a variavel target
X = df.drop("target", axis=1)

# Target variable
y = df.target.values

X.head()

y

"""Divisão de treinamento e teste

Agora vem um dos conceitos mais importantes no aprendizado de máquina, a divisão treinamento / teste.

É aqui que você dividirá seus dados em um conjunto de treinamento e um conjunto de teste.

Você usa seu conjunto de treinamento para treinar seu modelo e seu conjunto de teste para testá-lo.

O conjunto de teste deve permanecer separado do conjunto de treinamento.

Por que não usar todos os dados para treinar um modelo?

Digamos que você queira levar seu modelo ao hospital e começar a usá-lo em pacientes. Como você saberia o quão bem o seu modelo vai em um novo paciente não incluído no conjunto de dados completo original que você tinha?

É aqui que entra o conjunto de testes. Ele é usado para simular o máximo possível de levar seu modelo para um ambiente real.

E é por isso que é importante nunca deixar seu modelo aprender com o conjunto de teste, ele só deve ser avaliado nele.

Para dividir nossos dados em um conjunto de treinamento e teste, podemos usar train_test_split () do Scikit-Learn e alimentá-lo com nossas variáveis ​​independentes e dependentes (X e y).
"""

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 
                                                    y, # dependent variable
                                                    test_size = 0.2) # percentage of data to use for test set

"""O parâmetro test_size é usado para dizer à função train_test_split () quanto de nossos dados queremos no conjunto de teste.

Uma regra prática é usar 80% dos seus dados para treinar e os outros 20% para testar.

Em alguns casos você pode ter treinamento/validação e teste)
"""

X_train.head()

df.shape

X_train.shape

X_test.shape

y.shape

y_train.shape

y_test.shape

"""
Opções de modelo

Agora que preparamos nossos dados, podemos começar a ajustar os modelos. Estaremos usando o seguinte e comparando seus resultados.

* Regressão Logística - LogisticRegression ()
* K-vizinhos mais próximos - KNeighboursClassifier ()
* RandomForest - RandomForestClassifier ()

Se olharmos a folha de dicas do algoritmo Scikit-Learn, podemos ver que estamos trabalhando em um problema de classificação e esses são os algoritmos que ele sugere (e mais alguns).

Por enquanto, conhecer cada um desses algoritmos por dentro e por fora não é essencial.

O aprendizado de máquina e a ciência de dados são uma prática iterativa. Esses algoritmos são ferramentas em sua caixa de ferramentas.

No início, no caminho para se tornar um praticante, é mais importante entender o seu problema (como classificação versus regressão) e então saber quais ferramentas você pode usar para resolvê-lo.

Todos os algoritmos da biblioteca Scikit-Learn usam as mesmas funções, para treinar um modelo, model.fit (X_train, y_train) e para pontuar um modelo model.score (X_test, y_test). score () retorna a proporção de predições corretas (1,0 = 100% correto).

Como os algoritmos que escolhemos implementam os mesmos métodos para ajustá-los aos dados e também para avaliá-los, vamos colocá-los em um dicionário e criar um que os ajuste e pontue."""

models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores

"""
Comparação de Modelos

Como salvamos as pontuações de nossos modelos em um dicionário, podemos representá-los primeiro convertendo-os em um DataFrame."""

model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();

"""Olhando para o dicionário, o modelo LogisticRegression () tem melhor desempenho. Mas posso ir lá no hospital?

* Ajuste de hiperparâmetros - Cada modelo que você usa tem uma série de dials que você pode girar para ditar seu desempenho. Alterar esses valores pode aumentar ou diminuir o desempenho do modelo.

* Feature Selection and Importance - se houver uma grande quantidade de recursos que estamos usando para fazer previsões, alguns têm mais importância do que outros? Por exemplo, para prever doenças cardíacas, o que é mais importante, sexo ou idade?

* Matriz de confusão - Compara os valores preditos com os valores reais de forma tabular, se 100% correto, todos os valores na matriz serão do canto superior esquerdo para o inferior direito (linha de diagnóstico).

* Validação cruzada - divide seu conjunto de dados em várias partes, treina e testa seu modelo em cada parte e avalia o desempenho como uma média.

* Precision - proporção de verdadeiros positivos sobre o número total de amostras. Uma precisão mais alta leva a menos falsos positivos.

* Recall -  Proporção de verdadeiros positivos sobre o número total de verdadeiros positivos e falsos negativos. Uma recordação mais alta leva a menos falsos negativos.

* F1 Score - Combina precisão e recuperação em uma métrica. 1 é o melhor, 0 é o pior.

#### Ajuste de Hiperparâmetro e Validação Cruzada

Para preparar seu prato preferido, você sabe que deve colocar o forno em 180 graus e ligar a grelha. Mas quando seu colega de quarto cozinha seu prato favorito, ele define o uso de 200 graus e o modo de ventilação forçada. Mesmo forno, configurações diferentes, resultados diferentes.

O mesmo pode ser feito para algoritmos de aprendizado de máquina. Você pode usar os mesmos algoritmos, mas alterar as configurações (hiperparâmetros) e obter resultados diferentes.

Mas, assim como ligar o forno muito alto pode queimar sua comida, o mesmo pode acontecer com algoritmos de aprendizado de máquina. Você altera as configurações e funciona tão bem que se adapta (muito bem) aos dados.

Estamos procurando o modelo goldilocks. Um que se sai bem em nosso conjunto de dados, mas também se dá bem em exemplos não vistos.

Para testar diferentes hiperparâmetros, você pode usar um conjunto de validação, mas como não temos muitos dados, usaremos a validação cruzada.

O tipo mais comum de validação cruzada é k-fold. Envolve dividir seus dados em k-fold e, em seguida, testar um modelo em cada um. Por exemplo, digamos que tivemos 5 dobras (k = 5). É assim que pode parecer.
"""

predict = lr_model.predict([[58,	0,	0,	100,	248,	0,	0,	122,	0,	1.0,	1,	0,	2]])

predict

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

joblib.dump(lr_model, 'model.pkl')

!ls

from io import BytesIO
import requests
mLink = 'https://github.com/orlandojrps/stream/blob/main/model.pkl?raw=true'
mfile = BytesIO(requests.get(mLink).content)

model = joblib.load(mfile)

print(model)

# Load the pre-trained linear regression model
lr_model = joblib.load('model.pkl')

# Define the function to make a prediction
def predict_chd_risk(features):
    prediction = lr_model.predict(features)
    return prediction[0]

# Define the Streamlit app
def app():
    st.title('CHD Risk Prediction App')
    st.write('Enter the following information to predict your CHD risk:')
    age = st.slider('Age', 25, 80, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 500, 240)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Probable/Definite Left Ventricular Hypertrophy'])
    thalach = st.slider('Maximum Heart Rate Achieved', 50, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 2.0, 0.1)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3'])
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    if sex == 'Male':
        sex_val = 1
    else:
        sex_val = 0
    if cp == 'Typical Angina':
        cp_val = 0
    elif cp == 'Atypical Angina':
        cp_val = 1
    elif cp == 'Non-anginal Pain':
        cp_val = 2
    else:
        cp_val = 3
    if fbs == 'False':
        fbs_val = 0
    else:
        fbs_val = 1
    if restecg == 'Normal':
        restecg_val = 0
    elif restecg == 'ST-T Abnormality':
        restecg_val = 1
    else:
        restecg_val = 2
    if exang == 'No':
        exang_val = 0
    else:
        exang_val = 1
    features = np.array([age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = predict_chd_risk(features)
    st.write('Your predicted CHD risk is:', prediction)
