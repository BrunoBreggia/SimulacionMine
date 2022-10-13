#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Simulacion para probar los efectos de los distintos parámetros de la red neuronal en
la estimación de la información mutua de señales aleatorias con distribución gaussiana.
Se usará la implementación tradicional de MINE (Mutual information neural estimator) y
una versión con Exponential Moving Avrage (EMA) para remover el bias en MINE tradicional.

--------------------------------------------------------------------------------------
[SimConfig]
Sim_filename='Exp_01'
Sim_variables={'C':[0,1,2],'T':[0,1]}
Sim_realizations={'R':4}
Sim_name='E01'
Sim_hostname='jupiter'
[endSimConfig]
[SlurmConfig]
#SBATCH --mail-user=bruno.breggia@uner.edu.ar
#SBATCH --partition=internos
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
[endSlurmConfig]
'''
import json

import torch

from mine.mine import Mine
import numpy as np
import torch.optim as opt
# import matplotlib.pyplot as plt
import random
from mine.modelos import StandardModel
from mine.mine_ema import MineEMA
from copy import deepcopy
# import sys
import pandas as pd
from datetime import datetime
import os

sim = "sim01"
params = {"sim": sim}

C = 1  # indice de cantidad de capas
T = 1  # indice de tipo de mine
R = 1  # indice de rho
REA = 30

CAPA = [3, 4, 5]
TIPO = ['Mine', 'MineEMA']
RHO = [0, 0.5, 0.7, 0.98]

params['capas'] = CAPA
params['tipos'] = TIPO
params['rhos'] = RHO

capas = CAPA[C]
tipo = TIPO[T]
rho = RHO[R]

params['capa'] = capas
params['tipo'] = tipo
params['data_rho'] = rho
params['realizaciones'] = REA

# Seed initialization
SEED = random.randint(1, 5000)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

params['seed'] = SEED
params['data_mu'] = [0, 0]

# torch.cuda.manual_seed_all(0)
# torch.manual_seed(0)
# np.random.seed(0) # always using same numpy seed
# random.seed(0)

epocas = [2 ** i for i in range(8, 12)]
Muestras = [2 ** i for i in range(9, 15)]
Neuronas = [25, 50, 100, 200]
Activacion = ["relu", "leakyrelu", "gelu", "elu"]
Batch_frac = [1/4, 1/2, 1]

params['epocas'] = epocas
params['muestras'] = Muestras
params['neuronas'] = Neuronas
params['activacion'] = Activacion
params['batch_frac'] = Batch_frac

# ----------------------------------------------------------------------------
# output file
path = os.getcwd()
OUTDIR = path + "/outData"
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    print("created folder : ", OUTDIR, flush=True)
else:
    print(OUTDIR, "folder already exists.", flush=True)
oname = f"{OUTDIR}/{sim}_C{C}_T{T}_R{R}.json"

def generacion_de_datos(samples):
    """ 
    Genero señal aleatoria con distribución Gaussiana 
    """
    # media
    mu = np.array([0, 0])
    # covarianza
    cov_matrix = np.array([[1, rho], [rho, 1]])

    # Genero la señal
    joint_samples_train = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=(samples, 1))
    X_samples = joint_samples_train[:, :, 0]
    Z_samples = joint_samples_train[:, :, 1]
    # Entropia mediante formula
    true_mi = -0.5 * np.log(np.linalg.det(cov_matrix))

    return X_samples, Z_samples, true_mi


def test_model(muestras, neuronas, activacion, batch_size, epocas):
    """
    Funcion que probará una instancia de MINE con los datos pasados como
    argumentos.

        muestras: entero, cantidad de muestras de la señal de prueba
        neuronas: entero, cantidad de neuronasen las capas ocultas de la red
        activacion: string, nombre de la funcion de activacion a utilizar para la red
        batch_size: entero, tamaño de los batches a crear durante el entrenamiento.
        epocas: lista de enteros, con los valores de épocas contra los cuales 
                se pretende evaluar la red.

    """
    X_train, Z_train, mi = generacion_de_datos(muestras)
    dimX = X_train.shape[1]
    dimZ = Z_train.shape[1]
    d_in = dimX + dimZ
    # batch_size = 1024
    modelo = StandardModel(nLayers=capas, in_dimension=d_in, neurons=neuronas, activationFunc=activacion)

    if tipo == "Mine":
        model_mi = Mine(modelo)
    elif tipo == "MineEMA":
        model_mi = MineEMA(modelo)
    else:
        raise TypeError(f"No existe el modelo {tipo}")

    # Se escoje optimizador de tipo ADAM
    # learning_rate = 1e-4
    optimizer_mi = opt.Adam(model_mi.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.001)

    # Return estimations for diferent amounts of epochs
    output_train, output_test = model_mi.train_model(X_train, Z_train, optimizer=optimizer_mi, batch_size=batch_size, epochs=epocas, disableTqdm=True)
    return output_train, output_test, mi


def main():
    counter = 0
    total = len(Muestras)*len(Neuronas)*len(Activacion)*len(Batch_frac)
    dataframes = []

    for muestras in Muestras:
        for neuronas in Neuronas:
            for activacion in Activacion:
                for frac_batch in Batch_frac:

                    data = {
                        'epoca': None,
                        'muestras': muestras,
                        'neuronas': neuronas,
                        'activacion': activacion,
                        'batch': int(frac_batch * muestras),
                        'capas': capas,
                        'mine': tipo,
                        'rho': RHO[R],
                        'im_entrenamiento': [],
                        'im_testeo': [],
                        'im_verdadera': None
                    }
                    dfs = [deepcopy(data) for _ in range(len(epocas))]

                    for realizacion in range(REA):
                        # print(flush=True)
                        # print(f"Entrenando red con {muestras} muestras, con {neuronas} neuronas y {activacion} (batch de {frac_batch} del total de muestras)", flush=True)
                        # print(f"Con {capas-2} capas ocultas, y de modelo de tipo {tipo}", flush=True)
                        est_train, est_eval, mi = test_model(muestras, neuronas, activacion, int(frac_batch * muestras), epocas)
                        # print(flush=True)
                        # print(f"Informacion mutua real: {mi}", flush=True)

                        for i, df in enumerate(dfs):
                            df['epoca'] = epocas[i]
                            df['im_entrenamiento'].append(est_train[i])
                            df['im_testeo'].append(est_eval[i])
                            df['im_verdadera'] = mi

                    counter += 1
                    print(f'Avance: {counter/total*100:.2f}%', flush=True)

                    for df in dfs:
                        dataframes.append(pd.DataFrame.from_dict([df]))

    registro = pd.concat(dataframes, ignore_index=True)
    return registro


if __name__ == '__main__':
    inicio = datetime.now()
    print("inicio:", inicio, flush=True)

    data_df = main()

    final = datetime.now()
    print("inicio:", inicio, flush=True)
    print("final:", final, flush=True)

    # Metodo con formato json
    # escritura
    sim = {
        'params': params,
        'data': data_df.to_dict(),
    }
    with open(oname, 'w') as outfile:
        json.dump(sim, outfile)
    ## lectura
    # with open(oname, 'r') as infile:
    #     file_json = json.load(infile)
    # data_df = pd.DataFrame.from_dict(file_json['data'])
    # params_df = pd.DataFrame.from_dict([file_json['params']])
    #
    # print(data_df)
    # print(params_df)

    # a = pd.read_json(oname)

