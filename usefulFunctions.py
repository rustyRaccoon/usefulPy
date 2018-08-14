# -*- coding: utf-8 -*-
#Useful little functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pdb #use with: pdb.set_trace()

###########################################################
#debug stuff

#/debug stuff
###########################################################
    
def loadFile():
    #hide root window
    root = Tk()
    root.withdraw()
    
    #open file
    filename = askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    signal = pd.read_csv(filename,sep=';',skiprows=3,nrows=1) #read only column names
    columns = signal.columns.tolist() #get columns
    columns = columns[:len(columns)-1]
    signal = pd.read_csv(filename,sep=';',header=3,usecols=columns) #Read csv
    signal.index = pd.to_datetime(signal['dataTS']) #Set dateTime as index
    del signal['dataTS'] #not needed anymore, so delete
    print("Loaded file: " + filename)
    return signal

def getFFT(signal): #signal needs to be a dataframe with timestamps as indices
    t = signal.index
    
    #plot original signal
    plt.figure(figsize=(12,5))
    plt.plot(t,signal)
    plt.title('Original data')
    plt.xlabel('time [s]')
    plt.show()
    
    #create hanning window
    hann = np.hanning(len(signal.values))
    Y_f = np.fft.fft(hann*signal.values)
    
    #get sampling frequency
    td = t[1]-t[0]
    sampFreq = 1/(td.microseconds/1000000)
    N = int(len(Y_f)/2+1)
    
    #create x-axis vector
    X_f = np.linspace(0,sampFreq/2,N, endpoint=True)
    
    #plot FFT
    plt.figure(figsize=(12,5))
    plt.plot(X_f,2.0*np.abs(Y_f[:N])/N)
    plt.xlabel('Frequency ($Hz$)')
    plt.show()
    
def getSpectrum(signal):
    spectrum = np.abs(np.fft.ifft(signal))
    spectrum = spectrum[:len(spectrum)-int(len(spectrum)/2)]
    
    i = 0
    limit = 0.01*np.max(spectrum)
    
    while (spectrum[i] > limit or spectrum[i] == 0) and i < len(spectrum)-1:
        i+=1
    
    spectrum = spectrum[:i]
    N1 = len(spectrum)
    f = np.arange(N1)
    plt.figure
    plt.plot(f,spectrum[:N1])
    plt.xlim(0,N1)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.show()

def lowPass(signal,cutoff,timestep = None,x0 = None):
    tau = 1/cutoff
    if timestep is None:
        td = signal.index[1]-signal.index[0]
        delta_t = td.microseconds/1000000
    else:
        delta_t = timestep

    alpha = delta_t/tau
    y = np.zeros_like(signal.values)
    yk = signal.values[0] if x0 is None else x0
    for k in range(len(y)):
        yk += alpha * (signal.values[k]-yk)
        y[k] = yk    
    return y

def nonZeroAvg(x):
    avgSum = 0
    avgN = 0
    for i in range(0,len(x)):
        if x[i] != 0:
            avgSum += x[i]
            avgN+=1
    return avgSum/avgN