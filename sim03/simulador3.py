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
Sim_name='E02'
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

# ############# variables de simulacion  ##############
# Indices fijados por simconfig
RHO_IDX = 0
ACT_IDX = 0
REA = 1
subREA = 2
# TODO: preguntar a Feli que eran los subREA

# variables de archivo
rho = [0.0, 0.5, 0.98][RHO_IDX]
actFunc = ["relu", "Lrelu", "elu"][ACT_IDX]
lr = 1e-3  # 0.5
train_percent = 0.8
minibatch_percent = 0.125  # porcenaje respecto del total del batch de entrenamiento
max_epocas = 15_000

# variables a iterar en archivo
valores_capas = [1, 2, 3]
valores_neuronas = [50, 100, 200]
valores_muestras = [1e3, 3e3, 5e3, 10e3]

# donde va a correr la simulacion
# cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda = "cpu"

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
data = {
    "rho": [],
    "true_mi": [],
    "samples": [],
    "LR": [],
    "capas": [],
    "neuronas": [],
    "minibatches": [],
    "ultima_epoca": []
}

# ##############  Generacion de señales aleatorias  ###############
# defino la media
mu = np.array([0, 0])
# defino covarianza
cov_matrix = np.array([[1, rho], [rho, 1]])
# Genero la señal
samples = 5000  # TODO: a esto volverlo variable. Capaz creo funcion para generar señales aleatorias
joint_samples_train = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=(samples, 1))
X_samples = joint_samples_train[:, :, 0]
Z_samples = joint_samples_train[:, :, 1]
# Convert to tensors
x = torch.from_numpy(X_samples).float().to(device=cuda)
z = torch.from_numpy(Z_samples).float().to(device=cuda)
# Entropia mediante formula
true_mi = -0.5 * np.log(np.linalg.det(cov_matrix))

# ############## inicializo ray ##############
# ray.init(num_cpus=subREA)

# @ray.remote
def correr_epocas(red: Mine2, epocas: list, n_eval: int):
    dataLocal = {}
    dataLocal["rho"] = rho
    dataLocal["true_mi"] = true_mi
    dataLocal["samples"] = samples
    dataLocal["LR"] = lr
    dataLocal["capas"] = capas
    dataLocal["neuronas"] = red.neurons
    dataLocal["minibatches"] = red.minibatches
    for epoca in epocas:
        print("Comienza el entrenaminto")
        red.run_epochs(x, z, epoca, viewProgress=False)
        print("Termina el entrenaminto")
        # testing = [red.estimate_mi(x, z)]
        prom = red.estimate_mi(x, z)
        dataLocal[f"{epoca} epocas"] = prom
    return dataLocal


def main():
    mines = []
    cantidad_total = len(neuronas)*len(minibatches)
    for i_n, neurona in enumerate(neuronas):
        for i_m, minibatch in enumerate(minibatches):
            for rea in range(subREA):
                mines.append(Mine2(capas, neurona, lr, minibatch, cuda="cpu"))
            tic = time.time()
            #process_ids = [correr_epocas.remote(mine, epocas, 1000) for mine in mines]
            #realizaciones = ray.get(process_ids)
            realizaciones = [correr_epocas(mine, epocas, 1000) for mine in mines]
            for realizacion in realizaciones:
                for key in data.keys():
                    data[key].append(realizacion[key])
            toc = time.time()
            mines.clear()
            # mostrar grado de avance
            print("Progress:", (i_n+1)*(i_m+1)*100/cantidad_total, "%", flush=True)

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

