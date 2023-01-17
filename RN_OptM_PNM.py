# RED NEURONAL 
# La finalidad de la red es predecir el total de
# ingresos que podrá tener un nodo durante el día.
#
# ENTRADAS
# Nodo
# Producto: Internet - Television hogar - Telefonia
# Causa falla
# Naturaleza falla
# Nombre dia: Lunes - Martes - Miercoles...
# Tipo dia: Ordinario - Dominical - Festivo
# US SNR
# DS SNR
# US Power
# DS Power

# SALIDA
# Truckroll

# %% 
# PARTE 1: importar librerias
from unicodedata import decimal
import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD

print('Fin parte 1: librerias importadas')

# %% 
# Parte 2: Cargar, normalizar y vectorizar los datos
# Datos de entrenamiento (Marzo - Abril - Mayo)
train_data = pd.read_excel(r'RN_TrainData.xlsx', decimal = ',', thousands = ',')
train_labels = pd.read_excel(r'RN_TrainLabels.xlsx')
# Datos para validacion (Datos parciales de entrenamiento: Abril)
val_data = pd.read_excel(r'RN_ValidationData.xlsx', decimal = ',', thousands = ',')
val_labels = pd.read_excel(r'RN_ValidationLabels.xlsx')
# Datos para evaluacion (Datos nuevos: Junio)
test_data = pd.read_excel(r'RN_TestData.xlsx', decimal = ',', thousands = ',')
test_labels = pd.read_excel(r'RN_TestLabels.xlsx')

print("Datos cargados")

# Convertir a tipo flotante las entradas
train_data = train_data.astype('float32')
val_data = val_data.astype('float32')
test_data = test_data.astype('float32')

print("Entradas tipo flotantes")

# Salidas unicas
Y_train_labels = np.unique(train_labels)
Y_val_labels = np.unique(val_labels)
Y_test_labels = np.unique(test_labels)
print("Valores unicos de salidas: ", Y_train_labels)

# Vectorizamos los datos
var_NumClasses = [np.max(Y_train_labels), np.max(Y_val_labels), np.max(Y_test_labels)]
var_NumClasses = np.max(var_NumClasses) + 1
print("Num classes: ", var_NumClasses)

train_labels = to_categorical(train_labels, num_classes = var_NumClasses)
val_labels = to_categorical(val_labels, num_classes = var_NumClasses)
test_labels = to_categorical(test_labels, num_classes = var_NumClasses)

print("Salidas vectorizadas")

print('Forma de datos de entrenamiento:', train_data.shape, '\nForma de labels de entrenamiento:', train_labels.shape)
print('Forma de datos de validacion:', val_data.shape, '\nForma de labels de validacion:', val_labels.shape)
print('Forma de datos de evaluacion:', test_data.shape, '\nForma de labels de evaluacion:', test_labels.shape)
print('Fin parte 2: Cargar, normalizar y vectorizar los datos')

# %% ARQUITECTURA DE LA RED NEURONAL
# PARTE 3: Arquitectura de la red neuronal
# Modelo secuencial
model = Sequential()
Entradas = 10
Salidas = var_NumClasses
print("Salidas: ", Salidas)

# Capa de entrada
model.add(Dense(11, input_dim = Entradas, activation = 'relu'))
# Capa oculta
model.add(Dense(256, activation = 'relu'))
# Para evitar el overfitting el dropout desconecta y conecta neuronas
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
# Capa de salida
model.add(Dense(Salidas, activation = 'softmax'))

# VISTA DE LA ARQUITECTURA: tipo de red, capas con sus neuronas y parametros 
model.summary()

print('Fin parte 3: arquitectura de la red')

# %% COMPILAR RED
# PARTE 4: # Compilar la red
# optimizer =  SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True), metrics = ['accuracy'])

print('Fin parte 4: compilado')

# %% ENTRENAR LA RED Y EVALUARLA
# PARTE 5: Entrenamiento
history = model.fit(train_data, train_labels,
                    epochs = 200,
                    batch_size = 100,
                    validation_data = (val_data, val_labels),
                    verbose = 1)
print(history)
print('Fin parte 5: Entrenamiento')

# %% 
# Paso 6: Evaluacion del modelo
# Evaluamos el modelo
model.evaluate(test_data, test_labels)
print("Fin paso 6: Evaluacion del modelo")

#  %%
# Paso 7: Prediccion de TR
Predict_data = pd.read_excel(r'RN_PredictData.xlsx')
with pd.ExcelWriter("ResultadosPNM_TR_OptMultiple.xlsx", engine = 'openpyxl', mode = 'a', if_sheet_exists = 'overlay') as writer:
    Predict_data.to_excel(writer, sheet_name= "Hoja1", index = False)
Predict_data = Predict_data.astype('float32')
Prediccion = model.predict(Predict_data)

print(Prediccion)
print(Prediccion.shape)
print("fin parte 7")

# %%
# Paso 8: Guardar los resultados
#Resultado = {'1':Prediccion[:, 0], '2':Prediccion[:, 1], '3':Prediccion[:, 2], '4':Prediccion[:, 3], '5':Prediccion[:, 4],
#             '6':Prediccion[:, 5], '7':Prediccion[:, 6], '8':Prediccion[:, 7], '9':Prediccion[:, 8], '10':Prediccion[:, 9],
#             '11':Prediccion[:, 10], '12':Prediccion[:, 11], '13':Prediccion[:, 12], '14':Prediccion[:, 13], '15':Prediccion[:, 14],
#             '16':Prediccion[:, 15], '17':Prediccion[:, 16], '18':Prediccion[:, 17], '19':Prediccion[:, 18], '20':Prediccion[:, 19],
#             '21':Prediccion[:, 20], '22':Prediccion[:, 21], '23':Prediccion[:, 22], '24':Prediccion[:, 23], '25':Prediccion[:, 24],
#             '26':Prediccion[:, 25]}
#Resultado = pd.DataFrame(Resultado)

#with pd.ExcelWriter("Resultados_TR_OptMultiple.xlsx", engine = 'openpyxl', mode = 'a', if_sheet_exists = 'overlay') as writer:
#    Resultado.to_excel(writer, sheet_name = "Hoja1", startcol=7, index = False)