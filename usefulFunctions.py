# -*- coding: utf-8 -*-
#Useful little functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#hide root window
root = Tk()
root.withdraw()

#open file
filename =  askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
print("Loaded file: " + filename)

#Get dataframe from csv
dataFrame = pd.read_csv(filename,sep=';',header=3,index_col=0,parse_dates=True)
dataFrame.iloc[:,0] = dataFrame.iloc[:,0].interpolate()

#plot original signal
plt.figure(figsize=(15,7))
dataFrame.iloc[:,0].plot()
plt.title('Original data')
plt.ylabel('beam loss current')

#create hanning window
hann = np.hanning(len(dataFrame.beamLossCurrent.values))
Y = np.fft.fft(hann*dataFrame.iloc[:,0].values)

#get sampling frequency
td = dataFrame.index[1]-dataFrame.index[0]
sampFreq = 1/(td.microseconds/1000000)
N = int(len(Y)/2+1)

#create x-axis vector
X = np.linspace(0,sampFreq/2,N, endpoint=True)

#plot FFT
plt.figure(figsize=(15,7))
plt.plot(X,2.0*np.abs(Y[:N])/N)
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('beam loss current')

#plot in seconds
Xp = 1.0/X
plt.figure(figsize=(15,7))
plt.plot(Xp,2.0*np.abs(Y[:N])/N)
plt.xlabel('Period ($s$)')
plt.ylabel('beam loss current')