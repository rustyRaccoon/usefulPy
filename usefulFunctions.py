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

# Lists the functions and allows you to call one
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
            # List of directories
            directoryList = []
            # List ALL found files
            masterFileList = []
            # Dictionary to relate filenames to dirs
            masterFileDict = {}

            # Find all subdirectories in the specified directory
            for x in os.walk(path):
                directoryList.append(x[0])

            print("Found the following: ")

            # Work through all directories and subdirectories to find
            # files with the specified extension
            for directory in directoryList:
                # List for all found files in current directory
                fileList = []

                # Find the files
                for file in os.listdir(directory):
                    # If extension matches...
                    if file.endswith("." + extension):
                        # ... add the file to the list
                        fileList.append(file)
                        masterFileList.append(file)
                        masterFileDict[file] = directory

                # Display the directories and files
                if fileList:
                    print("%s:" % directory)
                    for item in fileList:
                        print("\t%s" % item)

            # Ask what to do next
            decision = input("What now?\n"
                             "\t[s]earch list"
                             "\t[d]elete all"
                             "\t[R]estart"
                             "\t[e]xit\n")

            # User wants to search the list
            if decision.lower() == 's':
                # List for found files
                foundIn = []
                searchTerm = input("What are you looking for?\n")

                # Loop through all files and see if we can find some
                # matching ones
                for item in masterFileList:
                    # If file is found, add it to the list
                    if searchTerm.lower() in str(item).lower():
                        foundIn.append(item)

                # Let's see if we even have items in the list
                if foundIn:
                    print("Found in the following directories:")
                    # Loop through files and use them as keys for the
                    # dict we created earlier
                    for item in foundIn:
                        print("\t%s: %s" % (masterFileDict[item], item))

            # User wants to delete all the files with specified extension
            elif decision.lower() == 'd':
                # Request confirmation as this is pretty permanent
                sure = input("Are you sure? [y/n]\n")

                if sure.lower() == 'y':
                    # We don't want everyone to be able to do this,
                    # so I set up a password to proceed. True, if anyone
                    # looks at the code they have the password but I'm
                    # too lazy to set up super fancy encrypted files that
                    # store passwords and whatnot.
                    tries = 3

                    while tries > 0:
                        delPass = input("Enter password:")

                        if delPass == 'm@st3rk3y':
                            for item in masterFileList:
                                os.remove(masterFileDict[item] + "\\" + item)
                            break
                        else:
                            tries -= 1
                            print("Sorry, wrong. "
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

        # Let's check if there were any args supplied. If not...
        if len(sys.argv) == 1:
            # ... ask the user to choose an input file via filedialog
            print("Please choose file for .csv import")
            # Create root object...
            root = Tk()
            # ... but hide the root window
            root.withdraw
            # Open filedialog
            filePath = filedialog.askopenfilename()

            # Check if file even exists
            if(os.path.exists(filePath)):
                pass
            else:
                print("File does not exist")
                root.destroy()

            # Ask how the csv is structured and which column to extract
            print("Note that the signal indices should be the timestamps")
            column = input("Which column is the data in (zero-indexed)?")
            skipRow = input("Which line does the data start at?")
            signal = pd.read_csv(filePath,
                                 skiprows=skipRow,
                                 usecols=column)
        else:
            # ... otherwise just use the supplied arg
            signal = args[0]

        # Time axis is just the indices. Hopefully they are the actual
        # times. If not, this will fail horribly.
        t = signal.index

        # Plot original signal
        plt.figure(figsize=(12, 5))
        plt.plot(t, signal)
        plt.title('Original data')
        plt.xlabel('time [s]')
        plt.show()

        # Create hanning window
        hann = np.hanning(len(signal.values))
        # Multiply it with the signal to mix them
        Y_f = np.fft.fft(hann*signal.values)

        # Get sampling frequency from indices
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

        # Let's check if there were any args supplied. If not...
        if len(sys.argv) == 1:
            # ... ask the user to choose an input file via filedialog
            print("Please choose file for .csv import")
            # Create root object...
            root = Tk()
            # ... but hide the root window
            root.withdraw
            # Open filedialog
            filePath = filedialog.askopenfilename()

            # Check if file even exists
            if(os.path.exists(filePath)):
                pass
            else:
                print("File does not exist")
                root.destroy()

            # Ask how the csv is structured and which column to extract
            column = input("Which column is the data in (zero-indexed)?")
            skipRow = input("Which line does the data start at?")
            signal = pd.read_csv(filePath,
                                 skiprows=skipRow,
                                 usecols=column)
        else:
            # ... otherwise just use the supplied arg
            signal = args[0]

        # Comput ifft from signal
        spect = np.abs(np.fft.ifft(signal))
        # Dump second half since it's just gonna be mirrored
        spect = spect[:len(spect)-int(len(spect)/2)]

        # Just as a running variable to count
        i = 0
        # Set a limit for when to cut off. 1% of max value should be fine
        limit = 0.01*np.max(spect)

        # Loop through spectrum collection and see when we reach the limit
        while (spect[i] > limit or spect[i] == 0) and i < len(spect)-1:
            i += 1

        # Dump everything after that
        spect = spect[:i]
        # Get number of useable samples from what we just counted
        N1 = len(spect)
        # Get frequencies to use as x-axis
        f = np.arange(N1)

        # Plot the spectrum
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

        # If no timestep was supplied by the user, try to calculate it
        # from the signal
        if timestep is None:
            td = signal.index[1]-signal.index[0]
            delta_t = td.microseconds/1000000
        else:
            delta_t = timestep

        # Calculate alpha (smoothing factor)
        alpha = delta_t/tau
        # Create zero-array with shape and dimension of signal
        y = np.zeros_like(signal.values)
        # If start value is not supplied, use first value of input signal
        # Otherwise use the supplied one, obviously
        yk = signal.values[0] if x0 is None else x0

        # Loop through the zero-array and fill it with values calculated
        # from time discrete low pass series equation:
        # y[i] = y[i-1] + a * (x[i] - y[i-1])
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

        # Prepare variables, yay
        avgSum = 0
        avgN = 0

        # Loop through the whole collection
        for i in range(0, len(x)):
            # If value is not zero, add it and increase counter
            if x[i] != 0:
                avgSum += x[i]
                avgN += 1

        # Calculate average of the just now acquired values
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

        # Create root object...
        root = Tk()
        # ... but hide the root window
        root.withdraw
        # Open filedialog
        filePath = filedialog.askopenfilename()

        # Check if the file even exists
        if(os.path.exists(filePath)):
            pass
        else:
            print("File does not exist")
            root.destroy()
            sys.exit()

        # Open for binary reading
        f = open(filePath, 'rb')

        # Create a PDF object from the file
        pdfFile = Py2PDF2.PdfFileReader(f)
        # Get the number of pages
        numPages = pdfFile.getNumPages()

        # Create empty string to hold out words later
        stringWords = ''
        # Iterate over all the pages
        for pageNo in range(numPages):
            # Get page object
            page = pdfFile.getPage(pageNo)
            # Get the text of current page
            content = page.extractText()
            # Dump everything that is not words or numbers
            textOnly = re.findall(r'[a-zA-Z0-9]+', content)

            # Add everything to our holding string and add spaces between
            # words or numbers
            for word in textOnly:
                stringWords = stringWords + ' ' + word

        # Print the whole text (mostly for checking what will be read)
        print(stringWords)
        # Convert the string of words to mp3 file
        tts = gTTS(text=stringWords, lang='en')
        # Get desktop path
        savePath = os.path.expanduser("~/Desktop")
        # Save mp3 file to desktop
        tts.save(os.path.join(savePath, "listen_pdf.mp3"))
# -----------------------------------------------------------------------


chosenFunction = input("\nWhich one? ")
# Allows the user to call functions from normal text input
method = getattr(Functions, chosenFunction)
method()
