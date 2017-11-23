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

# Funzione che espone l'ultimo consuntivo caricato
def Lastcons():
    df_mesi = df_sstor['MESE']
    return max(df_mesi)

# Funzione che prepara un grafico e lo salva in una figura
def SimplePlot(SeriePandas=None,LabelofSerie=None):
    plt.plot(SeriePandas,label=LabelofSerie)
    plt.legend()
    return plt.figure()


# Frame principale

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.canvas1 = tk.Canvas(self,width=600,height=150)
        self.canvas1.grid()
        self.imglogo = PIL.Image.open('Logo.jpg')
        self.quitButton = tk.Button(self, text='Quit', background='darkred', command=self.quit)
        self.quitButton.grid()
        
    def quit(self):
        self.destroy()
        return 

app = Application()
app.master.title('New Customers Forecast Application - rel. 1.0 - (C) Gianni Zattoni')
app.mainloop()

