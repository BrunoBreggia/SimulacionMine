#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Evaluacion de la informacion mutua entre las senales biomecanicas de la altura del pie
y apertura angular de las articulaciones del miembro inferior. Probado para todos los
sujetos de la base de datos de Camargo. Se emplea la version 2 de Mine, probada en sim03,
con tres capas, 50 neuronas en cada una, y funcion de activacion ReLU.
Se prueba con senales de marcha RAPIDA.
--------------------------------------------------------------------------------------
[SimConfig]
Sim_filename='Exp06'
Sim_variables={'CYCLE_IDX':[0,1,2,3]}
Sim_realizations={'SUJETO_IDX':22}
Sim_name='E06'
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

import torch
import os
import ray
import pandas as pd
from datetime import datetime
import itertools
from mine.mine2 import Mine2
import generador_datos as gd

# ############# variables de simulacion  ##############
# Iterables de la simulacion
ciclos = ["full", "swing", "stance", "nods"]
file_list = ['AB06_fast.mat',
             'AB07_fast.mat',
             'AB08_fast.mat',
             'AB09_fast.mat',
             'AB10_fast.mat',
             'AB11_fast.mat',
             'AB13_fast.mat',
             'AB12_fast.mat',
             'AB14_fast.mat',
             'AB15_fast.mat',
             'AB16_fast.mat',
             'AB17_fast.mat',
             'AB18_fast.mat',
             'AB19_fast.mat',
             'AB20_fast.mat',
             'AB21_fast.mat',
             'AB23_fast.mat',
             'AB24_fast.mat',
             'AB25_fast.mat',
             'AB27_fast.mat',
             'AB28_fast.mat',
             'AB30_fast.mat',
             ]
foots = ["rtoe", "ltoe"]
angles = ['rankle',
          'rsubt',
          'rknee',
          'rhip_addu',
          'rhip_flex',
          'rhip_rot',
          'lankle',
          'lsubt',
          'lknee',
          'lhip_addu',
          'lhip_flex',
          'lhip_rot',
          ]

# Indices fijados por simconfig
CYCLE_IDX = 1
SUJETO_IDX = 1
REA = 48

# Constantes de la red
ACT_FUNC = "relu"
NEURONAS = 50
CAPAS = 3
LR = 1e-3  # 0.001
TRAIN_PERCENT = 80  # porcentaje respecto del total del dataset
MINIBATCH_PERCENT = 10  # porcentaje respecto del total del dataset
MAX_EPOCH = 15_000
LR_PATIENCE = 250
LR_FACTOR = 0.5
VALIDATION_AVG = 100
STOP_PATIENCE = 1000

# Constantes de la senal
CYCLE = ciclos[CYCLE_IDX]
SUJETO = file_list[SUJETO_IDX - 1]
NORM = False

# Donde va a correr la simulacion
# CUDA = "cpu"
CUDA = "cuda:0" if torch.cuda.is_available() else "cpu"

# Directorio con resultados
sim = "sim06"
path = os.getcwd()
OUTDIR = path + "/outData"
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)
    print("created folder : ", OUTDIR, flush=True)
else:
    print(OUTDIR, "folder already exists.", flush=True)
output_file = f"{OUTDIR}/{sim}_{CYCLE}_{SUJETO[:4]}.csv"


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

    # Parametros de la senal
    dataLocal["samples"] = len(x)
    dataLocal["ciclo"] = CYCLE
    dataLocal["sujeto"] = SUJETO
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

    cantidad_total = len(foots)*len(angles)
    counter = 0

    data = {}

    for foot, angle in itertools.product(foots, angles):
        # Obtencion de datos
        signal = gd.obtener_senial("../../DatosCamargo_nogc/fast/" + SUJETO, foot, angle, CYCLE, norm=NORM)
        # foot height
        fh = torch.from_numpy(signal.foot_height).type(torch.FloatTensor)
        # articular angle
        ang = torch.from_numpy(signal.angle).type(torch.FloatTensor)
        # other info
        foot_side = signal.foot_side
        angle_description = signal.angle_description
        angle_side = signal.angle_side

        # Reshape Tensors
        fh = fh.reshape((len(fh), 1))
        ang = ang.reshape((len(ang), 1))
        x_id, z_id = ray.put(fh), ray.put(ang)

        # Instanciacion de la red
        rea_ids = []
        for realization in range(REA):
            # Paralelization line
            rea_ids.append(entrenar_red.remote(x_id, z_id,
                                               foot_side, angle_description, angle_side,
                                               NEURONAS, CAPAS))
        # Confluence point
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


if __name__ == '__main__':

    inicio = datetime.now()
    print("inicio:", inicio, flush=True)

    main()

    final = datetime.now()
    print("inicio:", inicio, flush=True)
    print("final:", final, flush=True)
