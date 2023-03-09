#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Simulacion para probar los efectos de los distintos parámetros de red en
la estimación de información mutua de señales aleatorias con distribución gaussiana.
Se usará la implementación tradicional de MINE (Mutual information neural estimator).

--------------------------------------------------------------------------------------
[SimConfig]
Sim_filename='Exp_02'
Sim_variables={'RHO_IDX':[0,1,2],'CAPAS_IDX':[0,1,2]}
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

# variables de simulacion
RHO_IDX = 1
CAPAS_IDX = 1
REA = 1
subREA = 48

# variables de archivo
rho = [0.1, 0.5, 0.9][RHO_IDX]
capas = [1, 2, 3][CAPAS_IDX]
lr = 0.5
minibatches = [1, 10, 100]
epocas = [1_000, 5_000, 10_000, 50_000]
neuronas = [30, 60, 90]
# cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda = "cpu"
# output file
sim = "sim02"
path = os.getcwd()
OUTDIR = path + "/outData"
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    print("created folder : ", OUTDIR, flush=True)
else:
    print(OUTDIR, "folder already exists.", flush=True)
output_file = f"{OUTDIR}/{sim}_R{RHO_IDX}_C{CAPAS_IDX}_R{subREA}.csv"

# Obtencion de datos de simulacion
# media
mu = np.array([0, 0])
# covarianza
cov_matrix = np.array([[1, rho], [rho, 1]])
# Genero la señal
samples = 5000
joint_samples_train = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=(samples, 1))
X_samples = joint_samples_train[:, :, 0]
Z_samples = joint_samples_train[:, :, 1]
# Convert to tensors
x = torch.from_numpy(X_samples).float().to(device=cuda)
z = torch.from_numpy(Z_samples).float().to(device=cuda)
# Entropia mediante formula
true_mi = -0.5 * np.log(np.linalg.det(cov_matrix))

# inicializo ray
ray.init(num_cpus=subREA)

# diccionario para los datos de cada realizacion -> convertir a dataframe
data = {}
data["rho"] = []
data["true_mi"] = []
data["samples"] = []
data["LR"] = []
data["capas"] = []
data["neuronas"] = []
data["minibatches"] = []
for epoca in epocas:
    data[f"{epoca} epocas"] = []

@ray.remote
def correr_epocas(red: Mine2, epocas: list, n_eval: int):
    dataLocal = {}
    dataLocal["rho"] = rho
    dataLocal["true_mi"] = true_mi
    dataLocal["samples"] = samples
    dataLocal["LR"] = lr
    dataLocal["capas"] = capas
    dataLocal["neuronas"] = red.neurons
    dataLocal["minibatches"] = red.minibatches
    for i in range(len(epocas)):
        if i == 0:
            red.run_epochs(x, z, epocas[i], viewProgress=False)
        else:
            red.run_epochs(x, z, epocas[i]-epocas[i-1], viewProgress=False)
        testing = [red.estimate_mi(x, z) for _ in range(n_eval)]
        prom = np.mean(testing)
        dataLocal[f"{epoca} epocas"] = prom
    return dataLocal


def main():
    mines = []
    cantidad_total = len(neuronas)*len(minibatches)
    counter = 0
    for i_n, neurona in enumerate(neuronas):
        for i_m, minibatch in enumerate(minibatches):
            counter += 1
            for rea in range(subREA):
                mines.append(Mine2(capas, neurona, lr, minibatch, cuda="cpu"))
            tic = time.time()
            process_ids = [correr_epocas.remote(mine, epocas, 1000) for mine in mines]
            realizaciones = ray.get(process_ids)
            for realizacion in realizaciones:
                for key in data.keys():
                    data[key].append(realizacion[key])
            toc = time.time()
            mines.clear()
            # mostrar grado de avance
            print("Progress:", counter/cantidad_total*100, "%", flush=True)

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

