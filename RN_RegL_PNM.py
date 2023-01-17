# RED NEURONAL: tecnologia HFC
# La finalidad de la red es predecir el total de
# ingresos que podrá tener un nodo durante el día.
#
# ENTRADAS
# Nodo
# Producto: Internet - Television hogar - Telefonia
# Causa falla
# Naturaleza falla
# Nombre dia: Lunes - Martes - Miercoles - Jueves - Viernes - Sabado - Domingo
# Tipo dia: Ordinario - Dominical - Festivo
# US SNR
# DS SNR
# US Power
# DS Power

# SALIDA
# Truckroll

# %% 
# Paso 1: Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, models, optimizers

print("Fin paso 1: importar librerias")

# %% 
# Paso 2: Definicion de funciones
def build_model_regression(lr_var, input_data):
    # Arquitectura del modelo
    model = models.Sequential()
    # Capa de entrada
    model.add(layers.Dense(11, activation='relu', input_shape = (input_data,)))
    # Capas ocultas
    model.add(layers.Dense(256, activation = 'relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation = 'relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation = 'relu'))
    #model.add(layers.Dropout(0.5))
    # Capa de salida
    model.add(layers.Dense(1))
    
    # Compilar
    model.compile(optimizer = optimizers.RMSprop(lr = lr_var), loss = 'mse', metrics=['mae'])
    return model

print("Fin paso 2: Funciones definidas")

# %% 
# Paso 3: Cargar, normalizar y vectorizar los datos
print("Inicio codigo")
# Datos de entrenamiento (Marzo - Abril - Mayo)
train_data = pd.read_excel(r'RN_TrainData.xlsx', usecols = ['NODO', 'PRODUCTO', 'CAUSA FALLA', 'NATURALEZA FALLA', 
                                                            'DIA SEMANA', 'US SNR AVG (dB)', 'DS SNR AVG (dB)', 
                                                            'DS RX Power AVG (dB)', 'US TX Power AVG (dB)'])
train_labels = pd.read_excel(r'RN_TrainLabels.xlsx')
# Datos para validacion (Datos parciales de entrenamiento: Abril)
val_data = pd.read_excel(r'RN_ValidationData.xlsx', usecols = ['NODO', 'PRODUCTO', 'CAUSA FALLA', 'NATURALEZA FALLA', 
                                                            'DIA SEMANA', 'US SNR AVG (dB)', 'DS SNR AVG (dB)', 
                                                            'DS RX Power AVG (dB)', 'US TX Power AVG (dB)'])
val_labels = pd.read_excel(r'RN_ValidationLabels.xlsx')
# Datos para evaluacion (Datos nuevos: Junio)
test_data = pd.read_excel(r'RN_TestData.xlsx', usecols = ['NODO', 'PRODUCTO', 'CAUSA FALLA', 'NATURALEZA FALLA', 
                                                            'DIA SEMANA', 'US SNR AVG (dB)', 'DS SNR AVG (dB)', 
                                                            'DS RX Power AVG (dB)', 'US TX Power AVG (dB)'])
test_labels = pd.read_excel(r'RN_TestLabels.xlsx')

#print('Forma de datos de entrenamiento:', train_data.shape, '\nForma de labels de entrenamiento:', train_labels.shape)
#print('Forma de datos de validacion:', val_data.shape, '\nForma de labels de validacion:', val_labels.shape)
#print('Forma de datos de evaluacion:', test_data.shape, '\nForma de labels de evaluacion:', test_labels.shape)

print("Datos cargados") 

# Normalizar los datos
mean_train = train_data.mean(axis = 0)
train_data = train_data - mean_train
std_train = train_data.std(axis = 0)
train_data = train_data/std_train
#print(train_data)

val_data = val_data - mean_train
val_data = val_data/std_train

test_data = test_data - mean_train
test_data = test_data/std_train
#print(test_data)

print("Datos normalizados")

print('Forma de datos de entrenamiento:', train_data.shape, '\nForma de labels de entrenamiento:', train_labels.shape)
print('Forma de datos de validacion:', val_data.shape, '\nForma de labels de validacion:', val_labels.shape)
print('Forma de datos de evaluacion:', test_data.shape, '\nForma de labels de evaluacion:', test_labels.shape)

print("Fin paso 3: Cargar, normalizar y vectorizar los datos")

# %% 
# Paso 4: Entrenamiento
k = 10
num_val_samples = len(train_data)//k
num_epoch = 50
all_history = []

print("Inicio entrenamiento")
for i in range(k):
    print("Fold: ", i)
    # Datos de entrenamiento
    Xtrain_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis = 0)
    Ytrain_labels = np.concatenate([train_labels[:i*num_val_samples], train_labels[(i + 1)*num_val_samples:]], axis = 0)
    
    # Datos de validacion
    Xval_data = np.concatenate([val_data[:i*num_val_samples], val_data[(i + 1)*num_val_samples:]], axis = 0)
    Yval_labels = np.concatenate([val_labels[:i*num_val_samples], val_labels[(i + 1)*num_val_samples:]], axis = 0)
    
    # Entrenamientoa
    model = build_model_regression(0.001, 9)
    history = model.fit(Xtrain_data, Ytrain_labels,
                        epochs = num_epoch,
                        batch_size = 150,
                        validation_data = (val_data, val_labels),
                        verbose = 0)
    all_history.append(history.history['val_mae'])
print("Fin paso 4: Entrenamiento")

# %% 
# Paso 5: Graficar
all_mae_avg = pd.DataFrame(all_history).mean(axis = 0)

plt.plot(range(1, len(all_mae_avg) + 1), all_mae_avg)
plt.show()

print("Fin paso 5: Graficar")

# %% 
# Paso 6: Evaluacion del modelo
# Evaluamos el modelo
model.evaluate(test_data, test_labels)
print("Fin paso 6: Evaluacion del modelo")
      
# %% 
# Paso 7: Predicción
Predict_data = pd.read_excel(r'RN_PredictData.xlsx', usecols = ['NODO', 'PRODUCTO', 'CAUSA FALLA', 'NATURALEZA FALLA', 
                                                            'DIA SEMANA', 'US SNR AVG (dB)', 'DS SNR AVG (dB)', 
                                                            'DS RX Power AVG (dB)', 'US TX Power AVG (dB)'])
with pd.ExcelWriter("ResultadosPNM_TR_RegLineal.xlsx", engine = 'openpyxl', mode = 'a', if_sheet_exists = 'overlay') as writer:
    Predict_data.to_excel(writer, sheet_name= "Hoja1", index = False)
Predict_data = Predict_data - mean_train
Predict_data = Predict_data/std_train

Prediccion = model.predict(Predict_data)
#print(Prediccion.shape)
#print(Prediccion[1,0])

Resultado = []
for i in range(len(Prediccion)):
    Resultado.append(Prediccion[i, 0])
    
Resultado = {'Prediccion': Resultado}
Resultado = pd.DataFrame(Resultado)

#print(Prediccion)
print("Fin paso 7: Prediccion")

# %%
# Paso 8: Guardar los resultados
with pd.ExcelWriter("ResultadosPNM_TR_RegLineal.xlsx", engine = 'openpyxl', mode = 'a', if_sheet_exists = 'overlay') as writer:
    Predict_data.to_excel(writer, sheet_name= "Hoja1", startcol = 11, index = False)
    Resultado.to_excel(writer, sheet_name = "Hoja1", startcol=22, index = False)    