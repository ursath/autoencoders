# Autoencoder

## Opciones de configuración

En el archivo config.json se encuentran todas las opciones disponibles para ejecutar el programa:

### 1. Autoencoder

- Arquitectura: define todas las capas (incluyendo la de entrada, salida y latente). Ejemplo [35, 20, 10, 2, 10, 20, 35]

- X_range: [x, y] siendo x el índice indice inicial desde donde se quieren considerar los caracteres de font e y el índice final. Es decir, es el subconjunto que se quiere tomar de la totalidad de los caracteres.

- hidden_layers_activation_functions: "tanh", "relu" o "logistic"

- output_layer_activation_function: "tanh", "relu" o "logistic"

- optimizer: "adam", "gradient_descent" o "momentum"

- error_functions: "squared_error", "mean_squared_error" o "mean_error"

### 2. Problem

- name: "normal", "generate", "denoising", "variational"

- font_data: "font_1", "font_2", "font_3"


## Resultados

En la carpeta stats se guardan los datos a través de las épocas y los gráficos, mientras que las letras resultantes, las generadas y el gráfico 2D del espacio latente se encuentra en la carpeta results.

## Observaciones

Para generar una nueva letra, se debe correr el programa con la opción "generate". Una vez que termine de realizar el entrenamiento, se le solicitará por consola el vector a partir del cual desea generar el nuevo caracter. Para ello puede fijarse en el gráfico del espacio latente en la carpeta results que ya estará generado.
