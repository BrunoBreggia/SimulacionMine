#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Simulacion para probar los efectos de los distintos parámetros de red en
la estimación de información mutua de señales aleatorias con distribución gaussiana.
Se usará la implementación tradicional de MINE (Mutual information neural estimator).

--------------------------------------------------------------------------------------
[SimConfig]
Sim_filename='Exp_03'
Sim_variables={'RHO_IDX':[0,1,2],'ACT_IDX':[0,1,2]}
Sim_realizations={'REA':1}
Sim_name='E03'
Sim_hostname='cluster-fiuner'
[endSimConfig]
[SlurmConfig]
#SBATCH --mail-user=bruno.breggia@uner.edu.ar
#SBATCH --partition=internos
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --tasks-per-node=48
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
from copy import deepcopy

# ############# variables de simulacion  ##############
# Indices fijados por simconfig
RHO_IDX = 0
ACT_IDX = 0
REA = 1
subREA = 2
# TODO: preguntar a Feli que eran los subREA

# variables de archivo
RHO = [0.0, 0.5, 0.98][RHO_IDX]
ACT_FUNC = ["relu", "Lrelu", "elu"][ACT_IDX]
LR = 1e-3  # 0.5
TRAIN_PERCENT = 0.8
MINIBATCH_PERCENT = 0.10  # porcenaje respecto del total del dataset
MAX_EPOCAS = 15_000
LR_PATIENCE = 250
LR_FACTOR = 0.5
VALIDATION_AVG = 100
STOP_PATIENCE = 1000

# variables a iterar en archivo
valores_capas = [1, 2, 3]
valores_neuronas = [50, 100, 200]
valores_muestras = [1e3, 3e3, 5e3, 10e3]

# donde va a correr la simulacion
cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
# cuda = "cpu"

# directorio con resultados
sim = "sim03"
path = os.getcwd()
OUTDIR = path + "/outData"
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    print("created folder : ", OUTDIR, flush=True)
else:
    print(OUTDIR, "folder already exists.", flush=True)
output_file = f"{OUTDIR}/{sim}_R{RHO_IDX}_A{ACT_IDX}_R{subREA}.csv"  # TODO: por que aparece subREA aca

# Diccionario para los datos de cada realizacion -> convertir a dataframe
data = {}
#     "rho": [],
#     "true_mi": [],
#     "samples": [],
#     "LR": [],
#     "capas": [],
#     "neuronas": [],
#     "minibatch_size": [],
#     "ultima_epoca": []
# }


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
# ray.init(num_cpus=subREA)


# @ray.remote
def correr_epocas(red: Mine2, samples: int):
    dataLocal = {}

    (x, z), true_mi = signal_generator(correlation_rho=RHO, samples=samples)
    minibatch_size = samples * MINIBATCH_PERCENT

    red.fit(x, z, train_percent=TRAIN_PERCENT, minibatch_size=minibatch_size, learning_rate=LR,
            num_epochs=MAX_EPOCAS, random_partition=True, patience=LR_PATIENCE, scaling_factor=LR_FACTOR)

    estimador1, estimador2, estimador3 = red.estimacion_mi()

    dataLocal["rho"] = RHO
    dataLocal["true_mi"] = true_mi
    dataLocal["samples"] = samples
    dataLocal["LR"] = LR
    dataLocal["LR_patience"] = LR_PATIENCE
    dataLocal["LR_factor"] = LR_FACTOR
    dataLocal["capas"] = red.hiddenLayers
    dataLocal["neuronas"] = red.neurons
    dataLocal["minibatch_size"] = minibatch_size
    dataLocal["last_epoch"] = red.last_epoc
    dataLocal["validation_avg"] = VALIDATION_AVG
    dataLocal["stop_patience"] = STOP_PATIENCE

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

    for index1, neuronas in enumerate(valores_neuronas):
        for index2, capas in enumerate(valores_capas):
            for index3, samples in enumerate(valores_muestras):
                for _ in range(subREA):
                    mines.append(Mine2(capas, neuronas, ACT_FUNC,
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


if __name__ == '__main__':
    inicio = datetime.now()
    print("inicio:", inicio, flush=True)

    main()

    final = datetime.now()
    print("inicio:", inicio, flush=True)
    print("final:", final, flush=True)

