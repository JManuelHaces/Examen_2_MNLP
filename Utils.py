# Librerias
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Flatten, Conv1D, MaxPooling1D, Input, Bidirectional, TimeDistributed

# -------------------------------------- Funciones -------------------------------------- #

# Creando un ciclo para separa las series de tiempo
def separar_series(data_series: pd.DataFrame, m4_info: pd.DataFrame, series_a_usar:list, path:str):
    """
    Función para separar las series de tiempo en archivos individuales.
    Parámetros:
        data_series: Dataframe con las series de tiempo.
        m4_info: Dataframe con la información de las series de tiempo.
        series_a_usar: Lista con las series de tiempo a usar.
        path: Ruta donde se guardarán los archivos.
        """
    # Creando un ciclo para separa las series de tiempo
    for serie in series_a_usar:
        # Creando el dataframe
        df = pd.DataFrame()
        # Agregando la columna de la serie
        df[serie] = data_series[serie]
        # Agregando la columna de la fecha
        df['Fecha'] = pd.date_range(start=m4_info[m4_info['M4id'] == serie]['StartingDate'].values[0], periods=len(df), freq='H')
        # Ordenando las columnas
        df = df[['Fecha', serie]]    
        # Imprimiendo iteración
        print(f'- Serie {serie}')
        # Guardando el archivo
        df.to_csv(fr'{path}/{serie}.csv', index=False)

# Función para cargar una serie de tiempo y convertirla en un dataframe
def carga_serie(path_serie:str, freq_serie:str='H'):
    """
    Función para cargar una serie de tiempo y convertirla en un dataframe.
    Parámetros:
        path_serie: Ruta del archivo de la serie de tiempo.
        freq_serie: Frecuencia de la serie de tiempo.
        Salida:
        df: Dataframe con la serie de tiempo.
    """
    # Cargando la serie de tiempo
    df = pd.read_csv(path_serie)
    # Creando el índice de formato DatetimeIndex
    df.index = pd.to_datetime(df['Fecha'], format='%Y-%m-%d %H:%M:%S')
    df.index.freq = freq_serie
    # Quitando la columna de la fecha
    df = df.drop('Fecha', axis=1)
    # Quitandole el nombre al índice
    df.index.name = None
    # Quitando nulos
    df = df.dropna()
    # Mostrando longitud de la serie
    print(f'\n- Longitud de la serie {df.iloc[:, 0].name}: {len(df)} datos.')
    # Mostrando fechas
    print(f'\t- Fecha inicial: {df.index[0]}')
    print(f'\t- Fecha final: {df.index[-1]}')
    # Mostrando la temporalidad
    print(f'\t- Temporalidad: {df.index.freq}')
    return df

# Ciclo para graficar las series de tiempo en diferentes grids
def plot_series(series: list[pd.DataFrame], series_name: list[str]):
    for i in range(len(series)):
        plt.subplot(3, 2, i+1)
        # Aumentando el tamaño de la gráfica
        plt.gcf().set_size_inches(15, 10)
        # Graficando cada una de las series de tiempo
        plt.plot(series[i])
        # Agregando el título de cada una
        plt.title(f'Serie: {series_name[i]}')

# Función para Divbidir una secuencia univariada en muestras
def split_univariate_sequence(sequence: pd.DataFrame, column: str, n_steps: int, tensor: bool = False, n_features: int = 1):
    # Cambiando el DF a lista
    sequence = sequence[column].tolist()
    X, y = list(), list()
    for i in range(len(sequence)):
        # #ncontrar el final de este patrón
        end_ix = i + n_steps
        # Comprobar si estamos más allá de la secuencia
        if end_ix > len(sequence)-1:
            break
        # Reunir partes de entrada y salida del patrón
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    # Convirtiendo a numpy array
    X = np.array(X)
    y = np.array(y)
    # revisando si se quiere un tensor
    if tensor:
        # Agregando una dimensión a X
        X = X.reshape((X.shape[0], X.shape[1], n_features))
    # Imprimiendo el forma de X y y
    print(f'\t- Forma de X: {X.shape}')
    print(f'\t- Forma de y: {y.shape}')
    return X, y

# Función para separar el array en train y test dependiendo del porcentaje
def split_train_test(X, y, train_size: float):
    # Calculando el tamaño del train
    train_size = int(len(X) * train_size)
    # Separando el array en train y test para X
    X_train = X[:train_size]
    X_test = X[train_size:]
    # Separando el array en train y test para y
    y_train = y[:train_size]
    y_test = y[train_size:]
    # Imprimiendo el tamaño de los arrays para X
    print(f'- X:')
    print(f'\t- X_train: {X_train.shape}')
    print(f'\t- X_test: {X_test.shape}')
    # Imprimiendo el tamaño de los arrays para y
    print(f'- y:')
    print(f'\t- y_train: {y_train.shape}')
    print(f'\t- y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test

# Función para sacar el error de la predicción (r2, MAE, MAPE)
def error_metrics(nombre_modelo, y_true, y_pred):
    # Obteniendo las métricas
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # Creando el dataframe con los resultados
    df = pd.DataFrame({'Modelo': [nombre_modelo],
                       'R2': [np.round(r2, decimals=2)],
                       'MAE': [np.round(mae, decimals=2)],
                       'MAPE': [np.round(mape, decimals=2)]})
    return df

# Función para plotear la predicción y el test
def plot_pred_test(nombre_modelo, title, pred, test):
    # Obteniendo los errores
    errores = error_metrics(nombre_modelo, test, pred)
    # Imprimiendo los errores
    print(errores)
    # Creando la figura
    plt.figure(figsize=(10, 6))
    # Graficando las predicciones y el test
    plt.plot(test, label='Test')
    plt.plot(pred, label='Predicción')
    # Agregando título y leyenda
    plt.title(title)
    plt.legend()
    # Mostrando la gráfica
    plt.show()
    return errores

# Función para concatenar los df de errores
def concat_errores(errores: list[pd.DataFrame]):
    # Df vacío
    df = pd.DataFrame()
    # Ciclo para concatenar los df
    for i in errores:
        # Concatenando
        df = pd.concat([df, i], axis=0)
    # Ordenando por el error
    df = df.sort_values(by=['R2'], ascending=False).reset_index(drop=True)
    return df

# Creando una función para crear el modelo MLP
def gen_MLP_model(X, y, val_split, n_steps, 
                  activation, num_layers, num_neurons, 
                  optimizer, lr, loss, metrics, 
                  patience, epochs, verbose, 
                  plot_history=True):
    # Creando el modelo secuencial
    model = Sequential()
    # Agregando la capa de entrada
    model.add(Input(shape=(n_steps,)))
    # Agregando las capas ocultas
    for i in range(num_layers):
        model.add(Dense(num_neurons, activation=activation))
    # Agregando la capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Entrenando el modelo
    history = model.fit(X, y, validation_split=val_split,
                        epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                        verbose=verbose)
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    # Graficando el historial
    if plot_history:
        # Analizar función de pérdida
        plt.plot(history.history['loss'], '-*', label='loss', markersize=3.5)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
    return model, history

# Función para crear una red CNN
def gen_CNN_model(X, y, val_split, n_steps, 
                  num_layers_cnn, num_filters, kernel_size, padding,
                  activation, num_layers_dense, num_neurons, 
                  optimizer, lr, loss, metrics, 
                  patience, epochs, verbose, 
                  plot_history):
    # Definición del modelo
    model = Sequential()
    # Agregando la capa de entrada
    model.add(Input(shape=(n_steps, 1)))
    # Creando un ciclo para agregar las capas CNN
    for i in range(num_layers_cnn):
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation=activation, padding=padding))
        model.add(MaxPooling1D(pool_size=2, padding=padding))
    # Aplanando la salida
    model.add(Flatten())
    # Creando un ciclo para agregar las capas Dense
    for i in range(num_layers_dense):
        model.add(Dense(num_neurons, activation=activation))
    # Agregando la capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Entrenando el modelo
    history = model.fit(X, y, validation_split=val_split,
                        epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                        verbose=verbose)
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    # Graficando el historial
    if plot_history:
        # Analizar función de pérdida
        plt.plot(history.history['loss'], '-*', label='loss', markersize=3.5)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
    return model, history

# Función para generar el modelo LSTM
def gen_LSTM_model(X, y, val_split, n_steps, n_features,
                  num_layers_lstm, activation_lstm, num_units_lstm, bidireccional,
                  activation, num_layers_dense, num_neurons, 
                  optimizer, lr, loss, metrics, 
                  patience, epochs, verbose, 
                  plot_history):
    # Definición del modelo
    model = Sequential()
    # Agregando capa de entrada
    model.add(Input(shape=(n_steps, n_features)))
    # Agregando las capas LSTM
    for i in range(num_layers_lstm):
        if bidireccional:
            model.add(Bidirectional(LSTM(units=num_units_lstm, activation=activation_lstm, return_sequences=True)))
        else:
            model.add(LSTM(units=num_units_lstm, activation=activation_lstm, return_sequences=True))
    # Flatten
    model.add(Flatten())
    # Agregando las capas Dense
    for i in range(num_layers_dense):
        model.add(Dense(num_neurons, activation=activation))
    
    # Agregando capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Entrenando el modelo
    history = model.fit(X, y, validation_split=val_split,
                        epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                        verbose=verbose)
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    # Graficando el historial
    if plot_history:
        # Analizar función de pérdida
        plt.plot(history.history['loss'], '-*', label='loss', markersize=3.5)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
    return model, history


# Función para crear una arquitectura de red neuronal CNN - LSTM
def gen_CNN_LSTM_model(X, y, val_split, n_steps, n_features,
                       num_layers_cnn, num_filters, kernel_size, padding,
                       num_layers_lstm, activation_lstm, num_units_lstm,
                       activation, num_layers_dense, num_neurons, 
                       optimizer, lr, loss, metrics, 
                       patience, epochs, verbose, 
                       plot_history):
    # Creando el modelo
    model = Sequential()
    # Capa de entrada
    model.add(Input(shape=(None, X.shape[2], X.shape[3])))
    # Ciclo para agregar las capas CNN
    for i in range(num_layers_cnn):
        model.add(TimeDistributed(Conv1D(filters=num_filters, kernel_size=kernel_size, padding=padding, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2, padding=padding)))
    # Aplanando la salida de las capas CNN
    model.add(TimeDistributed(Flatten()))
    # Agregando las capas LSTM
    for i in range(num_layers_lstm):
        model.add(LSTM(num_units_lstm, activation=activation_lstm, return_sequences=True))
    # Capa de salida
    model.add(LSTM(num_units_lstm, activation=activation_lstm))
    model.add(Flatten())
    # Capas densas
    for i in range(num_layers_dense):
        model.add(Dense(num_neurons, activation=activation))
    # Capa de salida
    model.add(Dense(1))
    # Compilando el modelo
    model.compile(optimizer=getattr(optimizers, optimizer)(learning_rate=lr),
                  loss=loss, 
                  metrics=metrics)
    # Entrenando el modelo
    history = model.fit(X, y, validation_split=val_split,
                        epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
                        verbose=verbose)
    ultimo_estado = np.round(history.history[f"val_{metrics[0]}"][-1], decimals=4)
    print(f"El último estado de la métrica {metrics[0]} es: {ultimo_estado}")
    # Graficando el historial
    if plot_history:
        # Analizar función de pérdida
        plt.plot(history.history['loss'], '-*', label='loss', markersize=3.5)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
    return model, history