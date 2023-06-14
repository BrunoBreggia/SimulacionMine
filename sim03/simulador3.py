#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Simulacion para probar los efectos de los distintos parámetros de red en
la estimación de información mutua de señales aleatorias con distribución gaussiana.
Se usará la implementación tradicional de MINE (Mutual information neural estimator).

--------------------------------------------------------------------------------------
[SimConfig]
Sim_filename='Exp03'
Sim_variables={'RHO_IDX':[0,1,2],'ACT_IDX':[0,1,2]}
Sim_realizations={'DUMMY':1}
Sim_name='E03'
Sim_hostname='cluster-fiuner'
[endSimConfig]
[SlurmConfig]
#SBATCH --mail-user=bruno.breggia@uner.edu.ar
#SBATCH --partition=internos
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
[endSlurmConfig]
'''
import numpy as np
import torch
import os
import ray
import time
import pandas as pd
from datetime import datetime
from mine.mine2 import Mine2

# ############# variables de simulacion  ##############
# Iterables de la simulacion
rhos = [0.0, 0.5, 0.98]  # Fijado por simulacion
act_funcs = ["relu", "Lrelu", "elu"]  # Fijado por simulacion
valores_capas = [1, 2, 3]
valores_neuronas = [50, 100, 200]
valores_muestras = [1e3, 3e3, 5e3, 10e3]

# Indices fijados por simconfig
RHO_IDX = 1
ACT_IDX = 1
DUMMY = 1
REA = 24

# Constantes de archivo
RHO = rhos[RHO_IDX]
ACT_FUNC = act_funcs[ACT_IDX]
LR = 1e-3  # 0.001
TRAIN_PERCENT = 80  # porcentaje respecto del total del dataset
MINIBATCH_PERCENT = 0.10  # porcentaje respecto del total del dataset
MAX_EPOCAS = 15_000
LR_PATIENCE = 250
LR_FACTOR = 0.5
VALIDATION_AVG = 100
STOP_PATIENCE = 1000
# TODO: podria crear aca el diccionario con los datos globales de la simulacion
#  en vez de hacerlo en la función de abajo de todo.

# Donde va a correr la simulacion
cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
# cuda = "cpu"

# Directorio con resultados
sim = "sim03"
path = os.getcwd()
OUTDIR = path + "/outData"
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    print("created folder : ", OUTDIR, flush=True)
else:
    print(OUTDIR, "folder already exists.", flush=True)
output_file = f"{OUTDIR}/{sim}_R{RHO_IDX}_A{ACT_IDX}.csv"


# ##############  Generacion de señales aleatorias  ###############
def signal_generator(mean=(0, 0), correlation_rho=0.5, samples=1000):
    """
    Genero pares de señales aleatorias con distribución normal y coeficiente de correlación rho.
    Devuelvo el par de señales generadas y el valor de información mutua correspondiente.
    """
    # defino la media
    mu = np.array(mean)
    # defino matriz de covarianza
    cov_matrix = np.array([[1, correlation_rho], [correlation_rho, 1]])
    # Genero la señal
    joint_samples_train = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=(samples, 1))
    X_samples = joint_samples_train[:, :, 0]
    Z_samples = joint_samples_train[:, :, 1]
    # Convert to tensors
    x = torch.from_numpy(X_samples).float().to(device=cuda)
    z = torch.from_numpy(Z_samples).float().to(device=cuda)
    # Entropia mediante formula (para distribuciones normales)
    true_mi = -0.5 * np.log(np.linalg.det(cov_matrix))
    return (x, z), true_mi

# ############## inicializo ray ##############
# ray.init(num_cpus=REA)

# TODO: instanciar mine aqui entro.
#  Debe recibir como paramteros neuronas, capas, etc
# @ray.remote  # TODO: Descomentar el ray y correr en la compu de la oficina
def correr_epocas(red: Mine2, samples: int):
    dataLocal = {}

    # TODO: llamar una sola vez a generar datos para todas las simulaciones
    #  usando el ray.put, por lo que esta linea deberia ir por fuera del corer_epocas
    #  y recibir las señales como parametro
    (x, z), true_mi = signal_generator(correlation_rho=RHO, samples=samples)
    minibatch_size = int(samples * MINIBATCH_PERCENT)

    red.fit(x, z, train_percent=TRAIN_PERCENT, minibatch_size=minibatch_size, learning_rate=LR,
            num_epochs=MAX_EPOCAS, random_partition=True, patience=LR_PATIENCE, scaling_factor=LR_FACTOR)

    estimador1, estimador2, estimador3 = red.estimacion_mi()

    # TODO: no repetir tanto los datos comunes a todas las realizaciones
    #  Meter la info en comun en un archivo externo, comun para todas las realizaciones
    # # Almacenamiento de datos de la realizacion # #
    # Parametros de la señal
    dataLocal["rho"] = RHO
    dataLocal["true_mi"] = true_mi
    dataLocal["samples"] = samples

    # Parametros constructivos de la red
    dataLocal["capas"] = red.hiddenLayers
    dataLocal["neuronas"] = red.neurons
    dataLocal["funcion_activacion"] = ACT_FUNC

    # Parametros de entrenamiento
    dataLocal["LR"] = LR
    dataLocal["LR_patience"] = LR_PATIENCE
    dataLocal["LR_factor"] = LR_FACTOR
    dataLocal["minibatch_size"] = minibatch_size
    dataLocal["last_epoch"] = red.last_epoc()
    dataLocal["validation_avg"] = VALIDATION_AVG
    dataLocal["stop_patience"] = STOP_PATIENCE

    # Resultados
    dataLocal["estimador1"] = estimador1[0]
    dataLocal["estimador1_epoca"] = estimador1[1]
    dataLocal["estimador2"] = estimador2[0]
    dataLocal["estimador2_epoca"] = estimador2[1]
    dataLocal["estimador3"] = estimador3[0]
    dataLocal["estimador3_epoca"] = estimador3[1]

    return dataLocal


def main():
    mines = []
    cantidad_total = len(valores_neuronas)*len(valores_capas)*len(valores_muestras)
    counter = 0

    data = {}

    for index1, neuronas in enumerate(valores_neuronas):
        for index2, capas in enumerate(valores_capas):
            for index3, samples in enumerate(valores_muestras):
                # TODO: cambiar este for. Que se llame a correr epocas aplicado a ray
                #  y que se guarden ls IDs (fijarse el codigo de Feli)
                for _ in range(REA):
                    mines.append(Mine2(capas, neuronas, ACT_FUNC, cuda=cuda,
                                       validation_average=VALIDATION_AVG, stop_patience=STOP_PATIENCE))
                realizaciones = [correr_epocas(mine, int(samples)) for mine in mines]
                for rea in realizaciones:
                    for key in rea.keys():
                        if key not in data.keys():
                            data[key] = []
                        data[key].append(rea[key])
                # toc = time.time()
                mines.clear()
                # mostrar grado de avance
                counter += 1
                progress = counter/cantidad_total * 100
                print(f"Progress: {progress} %", flush=True)

    # Pasamos el dataframe a un csv
    data_df = pd.DataFrame(data)
    data_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)


def generate_aux_datafile():
    # TODO: tambien agregar os parametros que son constantes y
    #  comunes a todas las realizaciones
    sim_iterables = {
        "rhos": rhos,
        "act_funcs": act_funcs,
        "valores_capas": valores_capas,
        "valores_neuronas": valores_neuronas,
        "valores_muestras": valores_muestras,
    }
    auxFile_name = f"{OUTDIR}/sim_iterables.txt"
    if not os.path.exists(auxFile_name):
        with open(auxFile_name, 'w', encoding='utf-8') as auxFile:
            for key in sim_iterables.keys():
                print(f"{key} : {sim_iterables[key]}", file=auxFile)


if __name__ == '__main__':

    generate_aux_datafile()

    inicio = datetime.now()
    print("inicio:", inicio, flush=True)

    main()

    final = datetime.now()
    print("inicio:", inicio, flush=True)
    print("final:", final, flush=True)

