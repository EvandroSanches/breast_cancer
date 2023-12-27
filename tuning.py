#Tuning para gerar os melhores parametros para o modelo

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
target = pd.read_csv('saidas_breast.csv')

def CriaRNA(otimizador, perda,activation, neurons):
    modelo = Sequential()

    modelo.add(Dense(units=neurons, activation=activation, input_dim=30))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=neurons, activation=activation))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(units=1, activation='sigmoid'))

    modelo.compile(optimizer=otimizador, loss=perda, metrics=['binary_accuracy'])

    return modelo


parametros = {'batch_size' : [10, 15 ,20],
              'epochs' : [150, 200, 300],
              'optimizer' : ['adam', 'sgd'],
              'loss' : ['binary_crossentropy', 'hinge'],
              'activation' : ['leaky_relu', 'tanh', 'relu'],
              'neurons' : [16,8]
        }

modelo = KerasClassifier(build_fn=CriaRNA, activation=parametros['activation'],  neurons=parametros['neurons'], otimizador=parametros['optimizer'], perda=parametros['loss'] )

grid_search = GridSearchCV(estimator=modelo,
                           param_grid=parametros,
                           scoring= 'accuracy',
                           cv=5)

grid_search = grid_search.fit(previsores, target)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
