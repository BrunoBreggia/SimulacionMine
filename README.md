# Realizaciones con Mine

Código para poner a prueba las implementaciones del 
algoritmo MINE en el repositorio BrunoBreggia/CodigoMine.

Se realizaron varias realizaciones:

1. **Sim01**: realización con datos gaussianos simulados. Se realizaron 24
realizaciones con la siguiente combinación de parámteros:
   * Cantidad de muestras: 512, 1024, 2048, 4096, 8192, 16384
   * Coeficiente de correlación: 0, 0.5, 0.7, 0.98
   * Épocas de entrenamiento: 256, 512, 1024, 2048
   * Cantidad de capas: 3, 4, 5
   * Cantidad de neuronas por capa: 25, 50, 100, 200
   * Funciones de activación: relu, leaky relu, elu, gelu
   * Particionado del batch: 25%, 50%, 100% del tamaño del dataset
   * Versión [Mine 1](mine/mine.py), con su variante 
   tradicional y su variante MINE EMA (ver paper de Belghazi _et al_)

Los resultados con MINE EMA mostraron resultados menos satisfactorios que
   MINE tradicional, por lo que se sospecha una mala implementación de la 
   misma. Se decide trabajar de aquí en más únicamente con MINE tradicional.

---

2. **Sim02**: realización con datos gaussianos simulados. Se realizaron 48
realizaciones con la siguiente combinación de parámteros:
   * Cantidad de muestras: 5000
   * Coeficiente de correlación: 0.1, 0.5, 0.9
   * Épocas de entrenamiento: mil, 5 mil, 10 mil, 50 mil
   * Cantidad de capas: 1, 2, 3
   * Cantidad de neuronas por capa: 30, 60, 90
   * Funciones de activación: relu
   * Particionado del batch: 1%, 10%, 100% del tamaño del dataset
   * Versión Mine: [Mine 2](mine/mine2.py) (MINE tradicional)
   
---

3. **Sim03**: realización con datos gaussianos simulados. Se realizaron 24
realizaciones con la siguiente combinación de parámteros:
   * Cantidad de muestras: mil, 3 mil, 5 mil, 10 mil
   * Coeficiente de correlación: 0.0, 0.5, 0.98
   * Épocas de entrenamiento: máximo 15 mil (corta antes si converge)
   * Cantidad de capas: 1, 2, 3
   * Cantidad de neuronas por capa: 50, 100, 200
   * Funciones de activación: relu, leaky relu, elu
   * Particionado del batch: 10% del tamaño del dataset
   * Versión Mine: [Mine 2,](mine/mine2.py) (MINE tradicional)
   
---

4. **Sim04**: Evaluación de la informacion mutua entre senales biomecanicas,
a ser altura del pie y apertura angular de la rodilla en el sujeto 06 de la 
base de datos de Camargo _et al_. Se realizaron 24 realizaciones con la siguiente 
combinación de parámteros:
   * Épocas de entrenamiento: máximo 15 mil (corta antes si converge)
   * Cantidad de capas: 1, 2, 3
   * Cantidad de neuronas por capa: 50, 100, 200
   * Funciones de activación: relu, leaky relu, elu
   * Particionado del batch: 10% del tamaño del dataset
   * Versión Mine: [Mine 2](mine/mine2.py) (MINE tradicional)

Los resultados de esta simulación sirvieron para una publicación en el 
   congreso de RPIC 2023 en Oberá, Misiones, y para un póster en las 
   Jornadas de IA del Litoral 2023 (de la UNL).

---

5. **Sim05**: Evaluación de la información mutua entre las senales 
biomecánicas de la altura del pie y apertura angular de las articulaciones
del miembro inferior. Probado para todos los sujetos de la base de datos de
Camargo _et al_. Se realizaron 48 realizaciones con la siguiente 
combinación de parámteros:

   * Épocas de entrenamiento: máximo 15 mil (corta antes si converge)
   * Cantidad de capas: 3
   * Cantidad de neuronas por capa: 50
   * Funciones de activación: relu
   * Particionado del batch: 10% del tamaño del dataset
   * Versión Mine: [Mine 2](mine/mine2.py) (MINE tradicional)

Se evalúa la información con señales de marcha a velocidad normal. 
La principal finalidad es comparar con los resultados de Restrepo _et al_.

---

6. **Sim06**: Evaluación de la información mutua entre las señales 
**de marcha rápida** de la altura del pie y apertura angular de las 
articulaciones del miembro inferior. Probado para todos los sujetos de la
base de datos de Camargo _et al_. Se realizaron 48 realizaciones con la
siguiente combinación de parámteros:
   * Épocas de entrenamiento: máximo 15 mil (corta antes si converge)
   * Cantidad de capas: 3
   * Cantidad de neuronas por capa: 50
   * Funciones de activación: relu
   * Particionado del batch: 10% del tamaño del dataset
   * Versión Mine: [Mine 2](mine/mine2.py) (MINE tradicional)

Estos son los resultados que se interpretan en el proyecto.

