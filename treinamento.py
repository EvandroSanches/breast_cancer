#Gera treinamento e testes para avaliar Overfitting, média de acertos e parametros

import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('entradas_breast.csv')
target = pd.read_csv('saidas_breast.csv')


def CriaRNA():
    modelo = Sequential()

    modelo.add(Dense(units=18, activation='leaky_relu', input_dim=30))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=18, activation='leaky_relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=1, activation='sigmoid'))

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.032,
            decay_steps=7460,
            decay_rate=0.012
        )
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return modelo

def analise_treino():
    modelo = KerasClassifier(build_fn=CriaRNA(),
                             epochs=400,
                             batch_size=10)


    resultado = cross_val_score(estimator=modelo,
                                X=previsores, y=target, cv=10, scoring='accuracy')

    media = resultado.mean()
    desvio = resultado.std()

    plt.bar(range(0,10),resultado)
    plt.xlabel('Épocas de Treino Cruzado')
    plt.ylabel('Porcentagem de Acertos')
    plt.title('Histórico de Treinamento\nMédia:'+str(media)+'\nDesvio Padrão:'+str(desvio))
    plt.show()

def gera_modelo():
    modelo = CriaRNA()
    modelo.fit(previsores, target, epochs=400, batch_size=10)
    modelo.save('Modelo.0.1')

gera_modelo()
