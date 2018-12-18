# Import all the stuff!!
import os
import sys
import re
import Py2PDF2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pdb  # Use with: pdb.set_trace()
from tkinter import Tk
from gtts import gTTS
from tkinter import filedialog
# -----------------------------------------------------------------------


functionList = ["findExtension",
                "getFFT",
                "getSpectrum",
                "lowPass",
                "nonZeroAvg",
                "playPDF"]
print("\nAvailable functions: ")
for function in functionList:
    print("\t" + function)
# -----------------------------------------------------------------------


class Functions(object):
    # Find all files with extension
    def findExtension(*args):
        """
        Finds all the files with a given extension in a given directory
        and all sub-directories. User can then search the list further
        or delete all the files in one fell swoop.

        Needed imports:
            import os
        args:
            none
        returns:
            none
        """
        while True:
            path = input("Enter file path: ")
            extension = input("Enter file extension: ")
            directoryList = []  # List of directories
            masterFileList = []  # List ALL found files
            masterFileDict = {}  # Dictionary to relate filenames to dirs

            # Find all subdirectories in the specified directory
            for x in os.walk(path):
                directoryList.append(x[0])
            # print(__directoryList)

            print("Found the following: ")

            # Work through all directories and subdirectories to find
            # files with the specified extension
            for directory in directoryList:
                fileList = []  # List for all found files in CURRENT directory

                # Find the files
                for file in os.listdir(directory):
                    if file.endswith("." + extension):
                        fileList.append(file)
                        masterFileList.append(file)
                        masterFileDict[file] = directory

                # Display the files
                if fileList:
                    print("%s:" % directory)
                    for item in fileList:
                        print("\t%s" % item)

            # Ask what to do next
            decision = input("What now?\n"
                             "\t[s]earch list"
                             "\t[d]elete all"
                             "\t[R]estart\t[e]xit\n")

            # User wants to search the list
            if decision.lower() == 's':
                foundIn = []
                searchTerm = input("What are you looking for?\n")

                for item in masterFileList:
                    if searchTerm.lower() in str(item).lower():
                        foundIn.append(item)
                if foundIn:
                    print("Found in the following directories:")
                    for item in foundIn:
                        print("\t%s: %s" % (masterFileDict[item], item))

            # User wants to delete all the files with specified extension
            elif decision.lower() == 'd':
                sure = input("Are you sure? [y/n]\n")

                if sure.lower() == 'y':
                    tries = 3

                    while tries > 0:
                        deletePassword = input("Enter the password that "
                                               "unlocks it all: ")

                        if deletePassword == 'm@st3rk3y':
                            for item in masterFileList:
                                os.remove(masterFileDict[item] + "\\" + item)
                            break
                        else:
                            tries -= 1
                            print("Sorry, wrong password. "
                                  "%i more tries\n" % tries)

            # User wants to go again...
            elif decision.lower() == 'r':
                continue

            # User wants to quit
            elif decision.lower() == 'e':
                break

            # User is a fucktard
            else:
                print("Dude...srsly. How hard can it be to press the "
                      "correct keys?")
    # -------------------------------------------------------------------

    def getFFT(*args):  # Signal shoudld be a dataframe with timestamp index
        """
        Calculates the FFT of a given signal (as parameter or .csv import)
        and plots it.

        Needed imports:
            import matplotlib.pyplot as plt
            import numpy as np
            from tkinter import Tk
            from tkinter import filedialog
        args:
            (optional) signal to process
        returns:
            none
        """

        if len(sys.argv) == 1:
            print("Please choose file for .csv import")
            root = Tk()
            root.withdraw
            filePath = filedialog.askopenfilename()

            if(os.path.exists(filePath)):
                pass
            else:
                print("File does not exist")
                root.destroy()

            column = input("Which column is the data in (zero-indexed)?")
            skipRow = input("Which line does the data start at?")
            signal = pd.read_csv(filePath,
                                 skiprows=skipRow,
                                 usecols=column)
        else:
            signal = args[0]

        t = signal.index

        # Plot original signal
        plt.figure(figsize=(12, 5))
        plt.plot(t, signal)
        plt.title('Original data')
        plt.xlabel('time [s]')
        plt.show()

        # Create hanning window
        hann = np.hanning(len(signal.values))
        Y_f = np.fft.fft(hann*signal.values)

        # Get sampling frequency
        td = t[1]-t[0]
        sampFreq = 1/(td.microseconds/1000000)
        N = int(len(Y_f)/2+1)

        # Create x-axis vector
        X_f = np.linspace(0,
                          sampFreq/2,
                          N,
                          endpoint=True)

        # Plot FFT
        plt.figure(figsize=(12, 5))
        plt.plot(X_f, 2.0*np.abs(Y_f[:N])/N)
        plt.xlabel('Frequency ($Hz$)')
        plt.show()
    # -------------------------------------------------------------------

    def getSpectrum(*args):
        """
        Plots the spectrum for a given signal (as parameter or .csv import)
        and plots it.

        Needed imports:
            import matplotlib.pyplot as plt
            import numpy as np
            from tkinter import Tk
            from tkinter import filedialog
        args:
            (optional) signal to process
        returns:
            none
        """

        if len(sys.argv) == 1:
            print("Please choose file for .csv import")
            root = Tk()
            root.withdraw
            filePath = filedialog.askopenfilename()

            if(os.path.exists(filePath)):
                pass
            else:
                print("File does not exist")
                root.destroy()

            column = input("Which column is the data in (zero-indexed)?")
            skipRow = input("Which line does the data start at?")
            signal = pd.read_csv(filePath,
                                 skiprows=skipRow,
                                 usecols=column)
        else:
            signal = args[0]

        spect = np.abs(np.fft.ifft(signal))
        spect = spect[:len(spect)-int(len(spect)/2)]

        i = 0
        limit = 0.01*np.max(spect)

        while (spect[i] > limit or spect[i] == 0) and i < len(spect)-1:
            i += 1

        spect = spect[:i]
        N1 = len(spect)
        f = np.arange(N1)
        plt.figure
        plt.plot(f, spect[:N1])
        plt.xlim(0, N1)
        plt.xlabel('frequency')
        plt.ylabel('amplitude')
        plt.show()
    # -------------------------------------------------------------------

    def lowPass(signal, cutoff, timestep=None, x0=None):
        """
        Shoves a given signal through a low pass filter.

        Needed imports:
            import numpy as np
        args:
            signal ... signal to be processed
            cutoff ... cutoff frequency of the filter
            timestep ... delta_T (default = calculated from data)
            x0 ... start value for processing (default = first data point)
        returns:
            signal passed through a low pass filter
        """

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
    # -------------------------------------------------------------------

    def nonZeroAvg(x):
        """
        Calculates the average of all non-zero values of a collection

        Needed imports:
            none
        args:
            collection to calculate the average for
        returns:
            average of all non-zero values
        """
        avgSum = 0
        avgN = 0

        for i in range(0, len(x)):
            if x[i] != 0:
                avgSum += x[i]
                avgN += 1

        return avgSum/avgN
    # -------------------------------------------------------------------

    def playPDF():
        """
        Reads a pdf file aloud if the user is to lazy to read it

        Needed imports:
            import os
            import sys
            import Py2PDF2
            from gtts import gTTS
            from tkinter import Tk
            from tkinter import filedialog
        args:
            none
        returns:
            none
        """
        root = Tk()
        root.withdraw
        filePath = filedialog.askopenfilename()

        if(os.path.exists(filePath)):
            pass
        else:
            print("File does not exist")
            root.destroy()
            sys.exit()

        f = open(filePath, 'rb')

        # Get the no of pages in pdf
        pdfFile = Py2PDF2.PdfFileReader(f)
        no_of_pages = pdfFile.getNumPages()

        # Iterate all the pages using regex to filter only words and numbers
        # Concatenate the words in each page
        string_words = ''
        for pageno in range(no_of_pages):
            page = pdfFile.getPage(pageno)
            content = page.extractText()
            textonly = re.findall(r'[a-zA-Z0-9]+', content)

            for word in textonly:
                string_words = string_words + ' ' + word

        # Convert the string of words to mp3 file
        print(string_words)
        tts = gTTS(text=string_words, lang='en')
        savePath = os.path.expanduser("~/Desktop")
        tts.save(os.path.join(savePath, "listen_pdf.mp3"))
# -----------------------------------------------------------------------


chosenFunction = input("\nWhich one? ")
method = getattr(Functions, chosenFunction)
method()
