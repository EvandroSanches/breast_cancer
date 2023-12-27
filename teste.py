from keras.models import load_model
from pandasgui import show
import pandas as pd


previsores = pd.read_csv('entradas_breast.csv')
target = pd.read_csv('saidas_breast.csv')

def predict(dados):
    modelo = load_model('Modelo.0.1')

    resultado = modelo.predict(dados)

    resultado = (resultado > 0.5)

    return resultado

df_resultado = pd.DataFrame(predict(previsores), columns=['Resultado'])

previsores = previsores.join(df_resultado)

previsores = previsores.join(target)

show(previsores)


