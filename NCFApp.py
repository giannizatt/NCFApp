#
# NCFApp
# New Customers Forecast Application
# Programma creato per il forecast dei clienti
# 

global release
global alpha
global mesi_forecast
release = '1.0'
alpha   = 0.05
mesi_forecast = 24

import numpy             as np 
import pandas            as pd 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Tkinter import *
import tkMessageBox

from PIL import ImageTk, Image

from statsmodels.tsa.stattools    import acf, pacf, adfuller, arma_order_select_ic
from statsmodels.tsa.seasonal     import seasonal_decompose
from statsmodels.tsa.arima_model  import ARIMA, ARIMAResults
from statsmodels.stats.diagnostic import acorr_ljungbox

# Declinazioni
global Decl
Decl = {}
Decl['FL'] = 'FLUSSO LORDO'
Decl['CP'] = 'CLIENTI PERSI'
Decl['FN'] = 'FLUSSO NETTO'
Decl['ST'] = 'STOCK'
Decl['IMP'] = 'IMPRESE'
Decl['PER'] = 'PERSONAL'
Decl['AZR'] = 'AZIENDE RETAIL'
Decl['RET'] = 'PRIVATI RETAIL'
Decl['TOTRET'] = 'TOTALE RETAIL'
Decl['BDT'] = 'TOTALE BDT'
Decl['CONS'] = 'CONSUNTIVO'
Decl['FORE'] = 'FORECAST prospect'
Decl['PRED'] = 'FORECAST predict'

            
class Application:
    ultimo_consuntivo = object
    index_forecast    = object
    FlagForecast      = False
    Tipomisura       = ('FL','CP','FN')
    Tipomisura_file  = ('FL','CP')
    Tipomisura_fore  = ('FL','CP')
    Tipocliente      = ('IMP','PER','AZR','RET','TOTRET','BDT')
    Tipocliente_file = ('IMP','PER','AZR','RET')
    Tipocliente_fore = ('IMP','PER','AZR','RET')
    Tiposerie        = ('CONS','PRED','FORE')
    
    def __init__(self,master):
        
        self.df_cons, self.df_pred, self.df_fore = self.PreparaFrame()
        self.ultimo_consuntivo = max(self.df_cons.index)
        self.index_forecast    = self.df_fore.index
        
        self.frame1 = Frame(master)
        self.frame1.pack(side='top')
        
        self.img_logo = ImageTk.PhotoImage(Image.open("logobanca.png"))
        self.panel_logo = Label(self.frame1,image = self.img_logo)
        self.panel_logo.pack(side = "left", fill = "both", expand = "yes")
        
        self.frame2 = Frame(master)
        self.frame2.pack(side='top')
        
        self.label2bis = Label(self.frame2, text=' ')
        self.label2bis.pack(side='top', fill='both', expand=True)           
        
        self.label2 = Label(self.frame2,font=('Verdana',20,'bold'))
        self.label2.pack(side='bottom', fill='both', expand=True)           
        
        self.frame3a = Frame(master, width=900)
        self.frame3a.pack(side='top')
        
        self.imgFL = Figure(figsize=(4,3), dpi=100)
        self.canvasFL = FigureCanvasTkAgg(self.imgFL, self.frame3a)
        self.canvasFL.show()
        self.canvasFL.get_tk_widget().pack(side='left', fill='both', expand=True)
        
        self.imgCP = Figure(figsize=(4,3), dpi=100)
        self.canvasCP = FigureCanvasTkAgg(self.imgCP, self.frame3a)
        self.canvasCP.show()
        self.canvasCP.get_tk_widget().pack(side='left', fill='both', expand=True)
  
        self.imgFN = Figure(figsize=(4,3), dpi=100)
        self.canvasFN = FigureCanvasTkAgg(self.imgFN, self.frame3a)
        self.canvasFN.show()
        self.canvasFN.get_tk_widget().pack(side='left', fill='both', expand=True)
                
        self.frame4 = Frame(master, width=900)
        self.frame4.pack(side='top')
        # Crea il gruppo dei pulsanti radio
        self.radiobase1 = StringVar()
        self.radiobase1.trace('w',self.ChangeGraph)
        self.radiobase2 = StringVar()
        self.radiobase2.trace('w',self.ChangeGraph)
        for code in self.Tipocliente :
            Radiobutton(self.frame4,
                        background='darkgray',foreground='black',
                        text=Decl[code], variable=self.radiobase1,
                        value=code, indicatoron=0, height=3, width=18).pack(anchor=W,side='left')
        for code in self.Tiposerie :
            Radiobutton(self.frame4,
                        background='darkblue',foreground='gray',
                        text=Decl[code], variable=self.radiobase2,
                        value=code, indicatoron=0, height=3, width=18).pack(anchor=W,side='right')
              

        self.frame4b = Frame(master)
        self.frame4b.pack(side='top')

        self.label4b1 = Label(self.frame4b, text = ' ')
        self.label4b1.pack(side='top', fill = "both", expand = "yes")

        self.label4b2 = Label(self.frame4b,font=('Verdana',16),text='LOG ')
        self.label4b2.pack(side='left', fill='both', expand=True)   
        self.Scroll  = Scrollbar(self.frame4b)
        self.Log     = Text(self.frame4b, height=5, width=160)
        self.Scroll.pack(side='right', fill='y')
        self.Log.pack(side='left', fill='y')
        self.Scroll.config(command=self.Log.yview)
        self.Log.config(yscrollcommand=self.Scroll.set)
            
        self.frame5 = Frame(master)
        self.frame5.pack(side='top')
        self.label5a = Label(self.frame5,text = ' ')
        self.label5a.pack(fill = "both", expand = "yes")
        
        self.frame6 = Frame(master)
        self.frame6.pack(side='top')
        
        self.label6a = Label(self.frame6, anchor='w', width = 60,font = ('Verdana',11,'bold'),
                             text = ' Release '+release+' - dicembre 2017 - (C) Gianni Zattoni')
        self.label6a.pack(side = 'left',fill = "both", expand = "yes")
        
        
        self.pulsante6b = Button(self.frame6, text='Elaborazione FORECAST', width = 20, height=3, font=('bold')) 
        self.pulsante6b["background"] = "lightblue"
        self.pulsante6b.bind('<Button-1>',self.ButtonForecast)
        self.pulsante6b.pack(side='right')  

        #self.pulsante6a = Button(self.frame6, text='Documentazione', width = 15, height=3)
        #self.pulsante6a["background"] = "lightcyan"
        #self.pulsante6a.bind('<Button-1>',self.ButtonDocumentazione)
        #self.pulsante6a.pack(side='right')      

        self.label6b = Label(self.frame6, anchor='e', width = 45, font = ('Verdana',11,'bold'),
                             text = ' Ultimo consuntivo caricato '+str(self.ultimo_consuntivo)[0:-12]+'    ')
        self.label6b.pack(side = 'right', fill = "both", expand = "yes")
                
        self.frame7 = Frame(master)
        self.frame7.pack(side='top')
        self.label7a = Label(self.frame7,text = ' ')
        self.label7a.pack(fill = "both", expand = "yes")
    
        self.Wlog('Lanciata applicazione NCFApp - release '+release)
        self.radiobase1.set('BDT')
        self.radiobase2.set('CONS')
        #self.df_cons = self.Loadcsv()
        #self.label6b['text']=' Ultimo consuntivo caricato '+str(self.ultimo_consuntivo)[0:-12]+'    '
        
    def Wlog(self,textrow) :
        self.Log.insert(END,textrow+'\n')
    
    def Reducto(self,timeserie) :
        # Funzione che taglia la coda della serie storica e tiene i valori piu recenti
        # Le serie vengono tagliate ai 4 anni piu recenti e ...
        # + 12 mesi per la differenziazione stagionale e ...
        # + 1 mese per la differenziazione de-trend
        if len(timeserie)>61 :
            tms_object = timeserie[-61:]
        elif len(timeserie)>49 :
            tms_object = timeserie[-49:]
        elif len(timeserie)>37 :
            tms_object = timeserie[-37:]
        return tms_object
    
    def PreparaFrame(self) : 
        dfp = pd.read_csv('NCFAdati.csv',sep=',',index_col=0, parse_dates=True, infer_datetime_format=True)
        dframecons = pd.DataFrame(index = dfp.index)
        for txt1 in self.Tipomisura_file  :
            for txt2 in self.Tipocliente_file :
                dframecons[txt1+'_'+txt2] = self.Reducto(dfp[txt1+'_'+txt2])
        for txt in self.Tipocliente_file :
            dframecons['FN_'+txt] = dframecons['FL_'+txt] - dframecons['CP_'+txt]
        for txt in self.Tipomisura :
            dframecons[txt+'_TOTRET'] = dframecons[txt+'_AZR'] + dframecons[txt+'_RET']
            dframecons[txt+'_BDT']    = dframecons[txt+'_IMP'] + dframecons[txt+'_PER'] + dframecons[txt+'_AZR'] + dframecons[txt+'_RET']
        self.ultimo_consuntivo = max(dframecons.index)
        
        index_forecast = pd.date_range(self.ultimo_consuntivo, periods=1+mesi_forecast, freq='M')
        index_forecast = index_forecast[1:]

        dframepred = pd.DataFrame(columns=dframecons.columns, index = dframecons.index)
        dframefore = pd.DataFrame(columns=dframecons.columns, index = index_forecast)
        return dframecons, dframepred, dframefore
    
    def ButtonForecast(self,evento) :
        self.ForecastManager()
        pass

    #def ButtonDocumentazione(self,evento) :
    #    pass
    
    def RefreshGraph(self, tipocliente, tiposerie):
        self.Wlog('Cambiamento grafici visualizzati in corso ... '+Decl[tiposerie]+' - '+Decl[tipocliente])
            
        self.label2['text'] = Decl[tiposerie]+' - '+Decl[tipocliente]
        
        header = 'FL_'+tipocliente
        self.imgFL.clear()
        ax = self.imgFL.add_subplot(111,title=Decl['FL'])
        if tiposerie == 'CONS' :
            ax.plot(self.df_cons[header])
            ax.plot(self.df_cons[header].rolling(window=12,center=False).mean(),color='purple')
        if tiposerie == 'PRED' :
            ax.plot(self.df_cons[header])
            ax.plot(self.df_pred[header])
        if tiposerie == 'FORE' :
            ax.plot(self.df_cons[header][mesi_forecast:])
            ax.plot(self.df_fore[header],color='red')
        self.canvasFL.draw()
        
        header = 'CP_'+tipocliente
        self.imgCP.clear()
        ax = self.imgCP.add_subplot(111,title=Decl['CP'])
        if tiposerie == 'CONS' :
            ax.plot(self.df_cons[header])
            ax.plot(self.df_cons[header].rolling(window=12,center=False).mean(),color='purple')
        if tiposerie == 'PRED' :
            ax.plot(self.df_cons[header])
            ax.plot(self.df_pred[header])
        if tiposerie == 'FORE' :
            ax.plot(self.df_cons[header][mesi_forecast:])
            ax.plot(self.df_fore[header],color='red')
        self.canvasCP.draw()
        
        header = 'FN_'+tipocliente
        self.imgFN.clear()
        ax = self.imgFN.add_subplot(111,title=Decl['FN'])
        if tiposerie == 'CONS' :
            ax.plot(self.df_cons[header])
            ax.plot(self.df_cons[header].rolling(window=12,center=False).mean(),color='purple')
        if tiposerie == 'PRED' :
            ax.plot(self.df_cons[header])
            ax.plot(self.df_pred[header])
        if tiposerie == 'FORE' :
            ax.plot(self.df_cons[header][mesi_forecast:])
            ax.plot(self.df_fore[header],color='red')
        self.canvasFN.draw()
        
    def ChangeGraph(self,*args):
        tipocliente = self.radiobase1.get()
        tiposerie   = self.radiobase2.get()
        if (tiposerie=='PRED' or tiposerie=='FORE') and self.FlagForecast==False :
            self.Wlog("I grafici di forecast possono essere richiesti solo dopo che il forecast e' stato calcolato")
        else :
            self.RefreshGraph(tipocliente,tiposerie)
    
    def ForecastManager(self) :
        self.Wlog('Avviata procedura di FORECAST')
        self.FlagForecast = True
        sequences = []
        for txt1 in self.Tipomisura_fore : 
            for txt2 in self.Tipocliente_file :
                sequences.append(txt1+'_'+txt2)
        for TMS in sequences :
            TimeSerie = pd.Series(self.df_cons[TMS])
            self.df_fore[TMS], self.df_pred[TMS] = self.Forecast(TimeSerie)
        self.Wlog('Terminata procedura di FORECAST')
        for txt in self.Tipocliente_file :
            self.df_fore['FN_'+txt] = self.df_fore['FL_'+txt] - self.df_fore['CP_'+txt]
            self.df_pred['FN_'+txt] = self.df_pred['FL_'+txt] - self.df_pred['CP_'+txt]
        self.Wlog('Ricalcolati flussi netti di Forecast')
        for txt in self.Tipomisura :
            self.df_fore[txt+'_TOTRET'] = self.df_fore[txt+'_AZR'] + self.df_fore[txt+'_RET']
            self.df_pred[txt+'_TOTRET'] = self.df_pred[txt+'_AZR'] + self.df_pred[txt+'_RET']
            self.df_fore[txt+'_BDT']    = self.df_fore[txt+'_IMP'] + self.df_fore[txt+'_PER'] + self.df_fore[txt+'_AZR'] + self.df_fore[txt+'_RET']
            self.df_pred[txt+'_BDT']    = self.df_pred[txt+'_IMP'] + self.df_pred[txt+'_PER'] + self.df_pred[txt+'_AZR'] + self.df_pred[txt+'_RET']
        self.Wlog('Ricalcolati Totali Retail e BdT di Forecast')
        self.df_fore.to_csv('NCFAfore.csv',sep=';')
        self.Wlog('Scaricato Forecast su file csv : NCFAfore.csv')
        dfcons = self.df_cons.copy()
        dfcons['ORIGIN']=0
        dffore = self.df_fore.copy()
        dffore['ORIGIN']=1
        dfcomp = pd.concat([dfcons,dffore])
        dfcomp.to_csv('NCFAcomp.csv',sep=';')
        self.Wlog('Scaricati dati completi su file csv : NCFAcomp.csv')

    def Forecast(self, TMS_orig = None):
        # Funzione che data una serie ne calcola il forecast
        # l'argomento e' la serie storica originale, 
        def test_stationarity(tms = None) :
            # Dickey - Fuller test
            # H0 : la serie non e' stazionaria
            # ritorna il p-value
            dftest = adfuller(tms, autolag='AIC')
            self.Wlog("Test Dickey-Fuller per verifica stazionarieta'")
            self.Wlog('Serie temporale     : '+tms.name)
            self.Wlog("H0 - ipotesi nulla  : La serie non e' stazionaria")
            self.Wlog('p_value riscontrato : {:05.4f}'.format(dftest[1]))
            if dftest[1]>alpha :
                self.Wlog("L'ipotesi nulla non si rifiuta - La serie NON e' stazionaria")
            else:
                self.Wlog("L'ipotesi nulla viene rifiutata - La serie e' stazionaria")
            return dftest[1]
        
        def test_whitenoise(tms = None) :   
            # Test Ljung-box sui residui per verifica White noise
            # (H0 : i residui sono WN)
            testWN = acorr_ljungbox(tms,lags=12)
            p_value_min = min(testWN[1])
            self.Wlog("Test Ljung-Box per verifica WHITE NOISE")
            self.Wlog('Serie temporale     : '+tms.name)
            self.Wlog("H0 - ipotesi nulla  : La serie e' white noise")
            self.Wlog('p_value riscontrato : {:05.4f}'.format(p_value_min))
            if p_value_min>alpha :
                self.Wlog("L'ipotesi nulla non si rifiuta - La serie e' White Noise")
            else:
                self.Wlog("L'ipotesi nulla viene rifiutata - La serie NON e' White Noise")
            return p_value_min
    
        self.Wlog('Iniziata procedura di FORECAST per la serie temporale '+TMS_orig.name)
        self.Wlog('Numero di osservazioni presenti      : '+str(len(TMS_orig)))
        #Elimina eventuali spazi vuoti
        TMS_orig = TMS_orig.dropna()
        self.Wlog('Numero di osservazioni post-cleaning : '+str(len(TMS_orig)))
        # differenziazione per la stagionalita'
        TMS_diff_seas = TMS_orig - TMS_orig.shift(12) #effetto stagionalita' eliminato
        self.Wlog('Ricalcolata serie temporale depurata degli effetti stagionali')
        TMS_effetto_trend = TMS_diff_seas / 12        #effetto trend mensile
        self.Wlog("Ricalcolata serie temporale depurata dell'effetto trend")
        #trend mensile medio sull'ultimo anno
        effetto_trend = np.mean(TMS_effetto_trend[-12:].values)
        self.Wlog('Effetto trend mensile (media degli ultimi 12 mesi)'+str(effetto_trend))
    
        decomposition = seasonal_decompose(TMS_orig,two_sided=False)
        TMS_trend     = decomposition.trend.dropna()
        TMS_seas      = decomposition.seasonal.dropna()
        TMS_resid     = decomposition.resid.dropna()
        TMS_resid.name = TMS_orig.name + "_residual"
        self.Wlog('Decomposizione stagionale e di trend terminata')
        self.Wlog('Inizio ricerca del numero ottimale di differenziazioni per giungere alla stazionarieta')
        # Ricerca il numero di differenziazioni sino a giungere alla stazionarieta'
        max_diff  = 5
        TMS_for_test = TMS_resid
        rank_diff = 0
        self.Wlog("Test per la verifica della stazionarieta' sulla serie destagionalizzata e detrendizzata")
    
        while (test_stationarity(TMS_for_test) > alpha) and (rank_diff < max_diff) :
            rank_diff += 1
            TMS_for_test = TMS_for_test - TMS_for_test.shift()
            TMS_for_test = TMS_for_test.dropna()
            TMS_for_test.name = TMS_resid.name + '('+str(rank_diff)+')'
        
        self.Wlog('Numero di differenziazioni da attuare per rendere la serie stazionaria : '+str(rank_diff))
        if rank_diff==0 : self.Wlog('--- valore ammissibile ma piuttosto raro')
        if rank_diff==1 : self.Wlog('--- valore standard e diffuso')
        if rank_diff==2 : self.Wlog('--- Warning! possono esserci trend anomali (almeno quadratici)')
        if rank_diff>2  : self.Wlog('--- Warning! valori anomali - serie temporale imprevedibile, anche se converge rischio di inferenza distorta')
    
        # se la serie e' gia' stazionaria prova a vedere se si tratta di un white noise
        if rank_diff == 0 :
            if test_whitenoise(TMS_resid) > alpha : 
                whitenoise = True
            else :
                whitenoise = False
        else :
            whitenoise = False
        
        if whitenoise :
            self.Wlog('Serie temporale uguale al white noise - nessuna ulteriore informazione catturabile')
            TMS_predict = pd.Series(0,index=df_cons.index)
            TMS_forecast = pd.Series(0,index=df_fore.index)
        else :
            autoregressive      = TMS_resid
            autoregressive_diff = TMS_for_test
            criteria_ic = arma_order_select_ic(autoregressive_diff)
            rank_AR = criteria_ic.bic_min_order[0]
            rank_MA = criteria_ic.bic_min_order[1]
            self.Wlog('Modello autoregressivo stimato : ARIMA('+str(rank_AR)+','+str(rank_diff)+','+str(rank_MA)+')')
            results_AR = ARIMA(autoregressive,order=(rank_AR,rank_diff,rank_MA)).fit(trend='c',disp=-1)
    
            TMS_predict = results_AR.fittedvalues
            forecast    = results_AR.forecast(steps=mesi_forecast)
            TMS_forecast    = pd.Series(forecast[0],index=self.index_forecast)
            forecast_stderr = pd.Series(forecast[1],index=self.index_forecast)
   
        # FORECAST
        self.Wlog('Preparazione delle serie storiche future in corso ...')
        new_TMS_predict = TMS_predict + TMS_trend + TMS_seas

        new_TMS_trend = pd.Series(effetto_trend, index = self.index_forecast)
        new_TMS_trend[0] = new_TMS_trend[0] + TMS_trend[-1]
        new_TMS_trend = new_TMS_trend.cumsum()
    
        mask_seas = TMS_seas[-12:]
        new_TMS_seas = pd.Series(0,index = self.index_forecast)
        index_base = 0
        ranges = range(0,len(self.index_forecast))
        for index_base in ranges :
            index_mask = index_base % 12
            new_TMS_seas[index_base] = mask_seas[index_mask]
    
        # Serie di forecast finale
    
        new_TMS_forecast = TMS_forecast + new_TMS_trend + new_TMS_seas
        self.Wlog('Terminata procedura di FORECAST per la serie temporale '+TMS_orig.name)
        return new_TMS_forecast, new_TMS_predict    

root = Tk()
App = Application(root)
root.title('New Customers Forecast Application - rel. '+release+' - (C) Gianni Zattoni')
root.mainloop()


#print 'Results of Dickey-Fuller Test:'
#dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#for key,value in dftest[4].items():
#dfoutput['Critical Value (%s)'%key] = value
#print dfoutput
#    for text, mode in MODES:
#        b = Radiobutton(master, text=text,
#                        variable=v, value=mode)
#        b.pack(anchor=W)
#Radiobutton(master, text="One", variable=v, value=1).pack(anchor=W)
#Radiobutton(master, text="Two", variable=v, value=2).pack(anchor=W)
    #plt.subplot(411)
    #plt.plot(TMS_orig, label='Original')
    #plt.legend(loc='best')
    #plt.subplot(412)
    #plt.plot(TMS_trend, label='Trend')
    #plt.legend(loc='best')
    #plt.subplot(413)
    #plt.plot(TMS_seas,label='Seasonality')
    #plt.legend(loc='best')
    #plt.subplot(414)
    #plt.plot(TMS_resid, label='Residuals')
    #plt.legend(loc='best')
    #plt.tight_layout()
    #decompose_graph = plt.figure()
# rappresentazione di ACF e PACF
        #lag_acf = acf(TMS_for_test, nlags=15)
        #lag_pacf = pacf(TMS_for_test, nlags=15, method='ols')
        #Plot ACF: 
        #plt.subplot(121) 
        #plt.plot(lag_acf)
        #plt.axhline(y=0,linestyle='--',color='gray')
        #plt.axhline(y=-1.96/np.sqrt(len(TMS_for_test)),linestyle='--',color='gray')
        #plt.axhline(y=1.96/np.sqrt(len(TMS_for_test)),linestyle='--',color='gray')
        #plt.title('Autocorrelation Function')
        #Plot PACF:
        #plt.subplot(122)
        #plt.plot(lag_pacf)
        #plt.axhline(y=0,linestyle='--',color='gray')
        #plt.axhline(y=-1.96/np.sqrt(len(TMS_for_test)),linestyle='--',color='gray')
        #plt.axhline(y=1.96/np.sqrt(len(TMS_for_test)),linestyle='--',color='gray')
        #plt.title('Partial Autocorrelation Function')
        #plt.tight_layout()
        #acfpacf_graph = plt.figure()
        #if graphic : plt.show()
  #, NavigationToolbar2TkAgg       
