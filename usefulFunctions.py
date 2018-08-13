# -*- coding: utf-8 -*-
#Useful little functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#def getFFT():
 #hide root window
root = Tk()
root.withdraw()

#open file
filename = askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
print("Loaded file: " + filename)

#Get data from csv
dataFrame = pd.read_csv(filename,sep=';',header=3,index_col=False,usecols=[0,1])
x = dataFrame.iloc[:,0]
x = pd.to_datetime(x)
y = dataFrame.iloc[:,1]

#plot original signal
plt.figure(figsize=(12,5))
plt.plot(x,y)
plt.title('Original data')
plt.xlabel('time [s]')
plt.show()

#create hanning window
hann = np.hanning(len(y.values))
Y_f = np.fft.fft(hann*y.values)

#get sampling frequency
td = x[1]-x[0]
sampFreq = 1/(td.microseconds/1000000)
N = int(len(Y_f)/2+1)

#create x-axis vector
X_f = np.linspace(0,sampFreq/2,N, endpoint=True)

#plot FFT
plt.figure(figsize=(12,5))
plt.plot(X_f,2.0*np.abs(Y_f[:N])/N)
plt.xlabel('Frequency ($Hz$)')
plt.show()