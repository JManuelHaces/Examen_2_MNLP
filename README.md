# Examen de Series de Tiempo con Redes Neuronales

Instrucciones: En este examen, se te proporcionarán dos conjuntos de datos diferentes. El primer conjunto de datos contiene datos univariados y se encuentran dentro de la carpeta `univariate` y el segundo conjunto de datos contiene datos multivariados y se encuentran dentro de la carpeta `multivariate`. 
___
El conjunto de datos `univariate` es el siguiente:
- El archivo `Hourly-train.csv` es parte del conjunto de datos M4, que es una colección de 100,000 series de tiempo utilizadas para la cuarta edición de la Competencia de Pronóstico Makridakis. Este archivo en particular contiene series de tiempo horarias que se utilizan para el entrenamiento en la competencia M4. Cada fila del archivo representa una serie de tiempo diferente y cada columna representa un valor de tiempo diferente. Los valores en cada fila son los valores históricos de la serie de tiempo correspondiente. El archivo `m4_info.csv` también es parte del conjunto de datos M4 y proporciona información adicional sobre cada serie de tiempo en el conjunto de datos. Contiene información como la frecuencia de cada serie de tiempo (anual, trimestral, mensual, semanal, diaria u horaria), el horizonte en el cuál se desea realizar la predicción y la fecha de inicio de la primera muestra. Estos datos serán relevantes para poder generar el índice de fechas y también que cada modelo debe de generar una predicción conforme al `Horizont` especificado. Cada estudiante deberá generar predicciones para las siguientes series de datos:

|           **Series de datos**          |            **Estudiante**            |
|:--------------------------------------:|:------------------------------------:|
| 'H204', 'H116', 'H156', 'H360', 'H387' |    PAULA DANIELA CARDENAS GALLARDO   |
| 'H193', 'H135', 'H46', 'H298', 'H48'   | CLAUDIA CELESTE CASTILLEJOS JAUREGUI |
| 'H199', 'H76', 'H365', 'H271', 'H72'   |        RAFAEL GALLARDO VAZQUEZ       |
| 'H229', 'H251', 'H405', 'H136', 'H300' |        JOSE MANUEL HACES LOPEZ       |
| 'H78', 'H118', 'H393', 'H247', 'H194'  |    OMAR ANTONIO HERNANDEZ SANCHEZ    |
| 'H56', 'H394', 'H263', 'H393', 'H297'  |     RAFAEL JUAREZ BADILLO CHAVEZ     |
| 'H16', 'H393', 'H240', 'H88', 'H96'    |    LEONARDO XAVIER PEREZ BALCORTA    |
| 'H295', 'H204', 'H104', 'H124', 'H201' |     PAULO ADRIAN VILLA DOMINGUEZ     |
| 'H124', 'H233', 'H34', 'H64', 'H45'    |     ANA ROSAURA ZAMARRON ALVAREZ     |
___
El conjunto de datos `multivariate` es el siguiente:
- El conjunto de datos “Daily Demand Forecasting Orders” contiene 60 filas y 13 columnas. Cada fila representa un día diferente y cada columna representa un atributo diferente. Los atributos en este conjunto de datos son los siguientes:

        Week of the month: La semana del mes (1 a 5).
        Day of the week (Monday to Friday): El día de la semana (2 a 6).
        Non-urgent order: El número de pedidos no urgentes.
        Urgent order: El número de pedidos urgentes.
        Order type A: El número de pedidos de tipo A.
        Order type B: El número de pedidos de tipo B.
        Order type C: El número de pedidos de tipo C.
        Fiscal sector orders: El número de pedidos del sector fiscal.
        Orders from the traffic controller sector: El número de pedidos del sector controlador de tráfico.
        Banking orders (1): El número de pedidos bancarios (1).
        Banking orders (2): El número de pedidos bancarios (2).
        Banking orders (3): El número de pedidos bancarios (3).
        Target (Total orders): El total de pedidos (objetivo).

    El objetivo es predecir el total de pedidos diarios utilizando los otros atributos como variables predictoras.

------

Tu tarea es utilizar las siguientes redes neuronales: `MLP`, `CNN`, `LSTM`, `CNN-LSTM` para realizar predicciones en ambos conjuntos de datos. Para cada red neuronal se debe realizar los siguientes análisis:

1. Carga el conjunto de datos (univariado o multivariado) donde se visualicen los datos de interés y además, se realicen las transformaciones y/o análisis que consideren pertinentes en los datos. Justifique su análisis y procedimientos.
2. Divida los datos en conjuntos de entrenamiento, validación y prueba. Tenga en cuenta que para los datos univariados los datos deben de generar un horizonte de predicción en base al dataset asignado.
3. Cree la red neuronal `elegida` y encuentre el conjunto de adecuado de capas ocultas, concatenación de redes `CNN` o `LSTM` en función del caso que esté analizando y seleccione la estructura que que arroje mejores resultados. Pruebe al menos 3 estructuras distintas. Discuta sus resultados y explique cómo podría mejorar el rendimiento de tu modelo.
4. Evalúa el modelo en el conjunto de prueba y mide su precisión mediante la comparación del pronóstico con los valores reales. Recuerde que si alguna tranformación fue realizada debe volver los datos a su valor real para poder compararlos adecuadamente.
5. Use Optuna para ajustar los hiperparámetros del modelo para obtener una precisión óptima. Evalúe el modelo en un conjunto de prueba y mida su precisión mediante la comparación del pronóstico con los valores reales.

Recuerda mostrar todo tu trabajo y explicar tus decisiones a medida que avanzas, comentar las funciones y código adecuadamente para una legibilidad del código. ¡Éxitos!