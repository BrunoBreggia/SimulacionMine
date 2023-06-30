#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Simulacion para probar los efectos de los distintos parámetros de red en
la estimación de información mutua de señales aleatorias con distribución gaussiana.
Se usará la implementación tradicional de MINE (Mutual information neural estimator).

--------------------------------------------------------------------------------------
[SimConfig]
Sim_filename='Exp04'
Sim_variables={'CYCLE_IDX':[0,1,2]}
Sim_realizations={'DUMMY':1}
Sim_name='E04'
Sim_hostname='cluster-fiuner'
[endSimConfig]
[SlurmConfig]
#SBATCH --mail-user=bruno.breggia@uner.edu.ar
#SBATCH --partition=internos
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
##SBATCH --partition=debug
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=8G
##SBATCH --gres=gpu:1
[endSlurmConfig]
'''
import numpy as np
import torch
import os
import ray
import time
import pandas as pd
from datetime import datetime
import itertools
from mine.mine2 import Mine2
from ProcesamientoCamargo.generar_senial import obtener_senial

# ############# variables de simulacion  ##############
# Iterables de la simulacion
ciclos = ["swing", "stance", "full"]
valores_capas = [1, 2, 3]
valores_neuronas = [50, 100, 200]
foots = ["rtoe", "ltoe"]
angles = ["rknee", "lknee"]

# Indices fijados por simconfig
CYCLE_IDX = 1
DUMMY = 1
REA = 24

# Constantes de la red
ACT_FUNC = "relu"
LR = 1e-3  # 0.001
TRAIN_PERCENT = 80  # porcentaje respecto del total del dataset
MINIBATCH_PERCENT = 10  # porcentaje respecto del total del dataset
MAX_EPOCH = 15_000
LR_PATIENCE = 250
LR_FACTOR = 0.5
VALIDATION_AVG = 100
STOP_PATIENCE = 1000

# Constantes de la señal
CYCLE = ciclos[CYCLE_IDX]
NORM = False

# Donde va a correr la simulacion
# CUDA = "cpu"
CUDA = "cuda:0" if torch.cuda.is_available() else "cpu"

# Directorio con resultados
sim = "sim04"
path = os.getcwd()
OUTDIR = path + "/outData"
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    print("created folder : ", OUTDIR, flush=True)
else:
    print(OUTDIR, "folder already exists.", flush=True)
output_file = f"{OUTDIR}/{sim}_{CYCLE}.csv"


# ##############  Inicializo ray  ###############
# ray.init(num_cpus=REA)

# @ray.remote(num_cpus=1, num_gpus=0.125, max_calls=1)
@ray.remote
def entrenar_red(x, z, foot_side, angle_description, angle_side,
                 neuronas: int, capas: int):
    # Instancio la red
    red = Mine2(capas, neuronas, ACT_FUNC, cuda=CUDA,
                validation_average=VALIDATION_AVG, stop_patience=STOP_PATIENCE)

    # Entreno la red
    minibatch_size = int(len(x) * MINIBATCH_PERCENT / 100)
    red.fit(x, z, train_percent=TRAIN_PERCENT, minibatch_size=minibatch_size, learning_rate=LR,
            num_epochs=MAX_EPOCH, random_partition=True, patience=LR_PATIENCE, scaling_factor=LR_FACTOR)

    # Obtengo las estimaciones
    estimador1, estimador2, estimador3 = red.estimacion_mi()

    # # Almacenamiento de datos de la realizacion # #
    dataLocal = {}

    # Parametros de la señal
    dataLocal["samples"] = len(x)
    dataLocal["ciclo"] = CYCLE
    dataLocal["foot"] = foot_side
    dataLocal["angle_description"] = angle_description
    dataLocal["angle_side"] = angle_side

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
    dataLocal["max_epoch"] = MAX_EPOCH
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

    cantidad_total = len(valores_neuronas)*len(valores_capas)
    counter = 0

    data = {}

    for foot, angle in itertools.product(foots, angles):
        # Obtencion de datos
        signal = obtener_senial("DatosCamargo_nogc/AB06_mine_excluded_nogc.mat", foot, angle, CYCLE, norm=NORM)
        x = signal.foot_height
        z = signal.angle
        foot_side = signal.foot_side
        angle_description = signal.angle_description
        angle_side = signal.angle_side
        x_id, z_id = ray.put(x), ray.put(z)

        # Instanciacion de la red
        for neuronas in valores_neuronas:
            for capas in valores_capas:
                rea_ids = []
                for index3 in range(REA):
                    # Paralelization line
                    rea_ids.append(entrenar_red.remote(x_id, z_id,
                                                       foot_side, angle_description, angle_side,
                                                       neuronas, capas))
                # Confluence point
                data.clear()
                for rea_data in ray.get(rea_ids):
                    for key in rea_data.keys():
                        if key not in data.keys():
                            data[key] = []
                        data[key].append(rea_data[key])
                # mostrar grado de avance
                counter += 1
                progress = counter/cantidad_total * 100
                print(f"Progress: {progress} %", flush=True)

    # Pasamos el dataframe a un csv
    data_df = pd.DataFrame(data)
    data_df.to_csv(output_file, mode='w', header=True, index=False)


def generate_aux_datafile():
    datos_comunes = {
        "ciclo": CYCLE,
        "funcion_activacion": ACT_FUNC,
        "LR": LR,
        "LR_patience": LR_PATIENCE,
        "LR_factor": LR_FACTOR,
        "minibatch_percentage": MINIBATCH_PERCENT,
        "max_epoch": MAX_EPOCH,
        "validation_avg": VALIDATION_AVG,
        "stop_patience": STOP_PATIENCE,
        "cuda": CUDA
    }

    sim_iterables = {
        "valores_capas": valores_capas,
        "valores_neuronas": valores_neuronas,
        "ciclos": ciclos,
        "foots": foots,
        "angles": angles
    }

    auxFile_name = f"{OUTDIR}/sim_parameters.txt"

    # if not os.path.exists(auxFile_name):
    with open(auxFile_name, 'w', encoding='utf-8') as auxFile:

        # Escribo paramteros fijados
        print("Los parametros fijados en esta corrida son: ", file=auxFile, flush=True)
        for key in datos_comunes.keys():
            print(f"{key} : {datos_comunes[key]}", file=auxFile)

        # Escribo parametros variables de la simulacion
        print("\nLos parametros variables en esta simulacion son: ", file=auxFile, flush=True)
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

