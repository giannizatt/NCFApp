#
# NCFApp
# New Customers Forecast Application
# Programma creato per il forecast dei clienti
# 

import numpy             as np   # libreria per la gestione dei numeri
import pandas            as pd   # libreria per la gestione del file csv
import seaborn           as sb   # libreria per la costruzione di grafici
import matplotlib.pyplot as plt  # libreria ausiliaria per la costruzione di grafici
import Tkinter           as tk   # libreria per l'interfaccia grafica


# Carica il dataframe dei dati dal file csv

df_sstor = pd.read_csv('NCFAdati.csv',sep=';')

# Crea il dataframe con i totali di BDT

df_totbdt = pd.DataFrame()
df_totbdt['MESE'] = df_sstor['MESE']
df_totbdt['FL_BDT'] = df_sstor['FL_IMP']+df_sstor['FL_PER']+df_sstor['FL_AZR']+df_sstor['FL_RET']
df_totbdt['CP_BDT'] = df_sstor['CP_IMP']+df_sstor['CP_PER']+df_sstor['CP_AZR']+df_sstor['CP_RET']
df_totbdt['ST_BDT'] = df_sstor['ST_IMP']+df_sstor['ST_PER']+df_sstor['ST_AZR']+df_sstor['ST_RET']
