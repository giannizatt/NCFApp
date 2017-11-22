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

df_seriestoriche = pd.read_csv('NCFAdati.csv')

