from mine.numerical_estimation import numerical_estimator
import generador_datos as gd

SUJETO = "AB07_mine_excluded_nogc.mat"
signal = gd.obtener_senial("../DatosCamargo_nogc/" + SUJETO, "rtoe", "rknee", "full", norm=False)

mi = numerical_estimator(signal.foot_height, signal.angle, bins=100)  # 50


print(mi)
