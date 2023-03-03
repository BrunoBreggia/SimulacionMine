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
from mine2 import Mine2

# variables de simulacion
RHO_IDX = 1
CAPAS_IDX = 0
REA = 48  # 16 30 48

# variables de archivo
rho = [0.1, 0.5, 0.9][RHO_IDX]
capas = [1, 2, 3][CAPAS_IDX]
lr = 0.5
minibatches = [1, 10,]# 100]
epocas = [1_000, 5_000, ]#10_000, 50_000]
neuronas = [30, 60, 90]  # [NEURONAS_IDX]
# cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda = "cpu"
output_file = 'simulation_data.csv'

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
ray.init(num_cpus=REA)


@ray.remote
def correr_epocas(red: Mine2, epocas: list, n_eval: int):
    data = {}
    data["rho"] = [rho]
    data["true_mi"] = [true_mi]
    data["samples"] = [samples]
    data["LR"] = [lr]
    data["capas"] = [capas]
    data["neuronas"] = [neuronas]
    data["minibatches"] = [red.minibatches]
    for epoca in epocas:
        red.run_epochs(x, z, epoca, viewProgress=False)
        # prom = 0
        # for i in range(n_eval):
        #     prom += (red.estimate_mi(x, z) - prom)/(i+1)
        testing = [red.estimate_mi(x, z) for _ in range(n_eval)]
        prom = np.mean(testing)
        data[f"{epoca} epocas"] = [prom]
    data_df = pd.DataFrame(data)
    data_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)


if __name__ == '__main__':
    mines = []
    for neurona in neuronas:
        for minibatch in minibatches:
            for rea in range(REA):
                mines.append(Mine2(capas, neurona, lr, minibatch, cuda="cpu"))
            tic = time.time()
            process_ids = [correr_epocas.remote(mine, epocas, 1000) for mine in mines]
            ray.get(process_ids)
            toc = time.time()
            mines.clear()
            # print(f"Simulacion terminada. Tiempo: {toc - tic}")
    # print("Simulaciones terminadas")
