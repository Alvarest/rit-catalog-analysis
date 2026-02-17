import requests 
import pandas as pd 
import os 
import tarfile 
import matplotlib.pyplot as plt 
import shutil 
import numpy as np

from rit_catalog_parser import *

# Functions to extract the data of the simulations
def saveProps(row):
    dirName = getDirName(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)
    outfile = os.path.join(results_path, "data.tsv")

    with open(outfile, 'w') as output:
        cols = list(row)
        for i, value in enumerate(row):
            output.write(cols[i] + "\t" + str(value) + "\n")
            
def getMode(filename):
    if filename.startswith("rPsi4") and filename.endswith(".asc"):
        l = int(filename[7])
        if filename[10] == "-":
            m = - int(filename[11])
        else:
            m = int(filename[10])
        return [l, m]
    raise Exception("The file " + filename + " doesn't contain any data.")

def getPsi4Data(row, mode):
    downloadMergeData(row)
    filename = "rPsi4_l" + str(mode[0]) + "_m" + str(mode[1]) + "_rInf.asc"
    data_path = os.path.join(tmp, getDirName(row))
    dataInFile = open(os.path.join(data_path, filename), 'r')
    names = ['time', 'real', 'imag', 'ampl', 'phse', 'omeg']
    simData = pd.read_table(dataInFile, sep = '\t', names = names, index_col = False, skiprows = 4)

    return simData
    
def filename_to_Psi4Data(row, filename):
    data_path = os.path.join(tmp, getDirName(row))
    dataInFile = open(os.path.join(data_path, filename), 'r')
    names = ['time', 'real', 'imag', 'ampl', 'phse', 'omeg']
    simData = pd.read_table(dataInFile, sep = '\t', names = names, index_col = False, skiprows = 4)

    return simData

def getInitialMasses(row):
    downloadMetaData(row)
    data_path = os.path.join(metadata_path, getMetaName(row))

    with open(data_path, 'r') as indat:
        for line in indat:
            if line.startswith("initial-mass"):
                if line[12] == "1":
                    m1 = float(line.split("= ")[1])
                elif line[12] == "2":
                    m2 = float(line.split("= ")[1])

    return m1, m2

# Functions for plotting simulations' data               
def plotTable(table, l, m, colx, coly, fileName = ""):
    x = table[colx]
    y = table[coly]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, 'b-', label = "l = " + str(l) + ", m = " + str(m))
    ax.set_xlabel("Time")
    ax.set_ylabel("Real value of the signal")
    plt.legend()
    plt.tight_layout()
    if fileName:
        plt.savefig(fileName, dpi=200, bbox_inches = 'tight')
        print("Figure saved to: " + fileName)
    else:
        plt.show()
    plt.close()

def select_label(axis_name):
    if axis_name == "time":
        return "Time [s]"
    elif axis_name == "real":
        return "Real value of the signal"
    elif axis_name == "imag":
        return "Imaginary value of the signal"
    elif axis_name == "ampl":
        return "Amplitude of the signal"
    elif axis_name == "phse":
        return "Phase of the signal"
    else:
        return "Axis Label"

def plotTableInAx(table, l, m, colx, coly, ax):
    x = table[colx]
    y = table[coly]

    ylabel = select_label(coly)

    ax.plot(x, y, 'b-', label = "l = " + str(l) + ", m = " + str(m))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title("Mode (" + str(l) + "," + str(m) + ")")


def plotAroundMax(row, xaxis = "time", yaxis = "real", thrshSize = 10):
    dirName = getDirName(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)
    
    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) 
    print(dirName)

    xlabel = select_label(xaxis)
    ylabel = select_label(yaxis)

    maxMode = getPsi4Data(row, [2,2])
    maxi = maxMode.max()["ampl"]
    
    for file in dataFiles:
        print("Plotting file " + file, end="\r")
        table = filename_to_Psi4Data(row, file)
        peak = table[
            (maxMode >= maxi / thrshSize) & 
            (table["time"] > table.max()["time"]/2) 
        ] 
        l, m = getMode(file)
        peak.plot(x = xaxis, y = yaxis, xlabel = xlabel, ylabel = ylabel, 
                  label = "Mode (" + str(l) + "," + str(m) + ")")
        plt.legend()
        plt.savefig(os.path.join(results_path, file[:-4] + ".svg"), dpi=200, bbox_inches = 'tight')
        plt.close()  

# The row data must be downloaded.
# There must be 21 mode files.
def plotAllModes(row, yaxis="real", filename = ""):
    dirName = getDirName(row)
    
    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)
    
    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) 
    
    fig, axs = plt.subplots(7, 3, figsize = (10, 24), sharey = True)
    fig.suptitle("Simulation " + row["number"][4:] + " with same y axis", fontsize = 20)
    
    for file in dataFiles:
        if file.endswith(".asc"):
            [l,m] = getMode(file)
            table = getPsi4Data(row, [l,m]) 
            n = l*l - 4 + l + m # Ordering modes
            i = int(n/3)
            j = n % 3
            plotTableInAx(table, l, m, 'time', yaxis, axs[i, j])
    
    # Adjust layout and save figure
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if filename:
        plt.savefig(os.path.join(results_path, filename), dpi=200, bbox_inches = 'tight')
    else:
        plt.show()
    plt.close()
    print("Figure created and saved successfully")

# Modes has to be a list of lists [l,m]
def plotAmplitudes(row, modes, filename = "amplitudes.pdf"):
    dirName = getDirName(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)
    
    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) 

    fig, ax = plt.subplots(figsize = (7, 5))
    ax.set_title("Amplitudes for simulation " + row["number"][4:])
    plt.margins(x=0, y=0.01)
    
    modes = np.array(modes) # Dimensión (2,x)
    for mode in modes:
        table = getPsi4Data(row, mode)
        ax.plot(table["time"], table["ampl"], label = "Mode (" + str(mode[0]) + "," + str(mode[1]) + ")")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude of the signal")

    xend = list(table.iloc[-1:]["time"])[0]+1
    plt.xlim(right = xend)
    fig.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, filename), dpi=200, bbox_inches = 'tight')
    plt.close()  

def plotAmplitudesAroundMax(row, modes, filename = "amplitudes.pdf", thrshSize = 10):
    dirName = getDirName(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)
    
    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) #
    
    # Creamos la figure con sus subfigures
    fig, ax = plt.subplots(figsize = (7, 5))
    plt.margins(x=0, y=0.01)
    
    modes = np.array(modes) # Dimensión (2,x)
    tables = [getPsi4Data(row, mode) for mode in modes]
    maxs = np.array([table.max()["ampl"] for table in tables])

    peaks = [
        table[
            (tables[maxs.argmax()]["ampl"] >= tables[maxs.argmax()].max()["ampl"] / thrshSize) & 
            (table["time"] > table.max()["time"]/2) 
        ] 
        for table in tables] 
    
    for i,peak in enumerate(peaks):
        ax.plot(peak["time"], peak["ampl"], label = "Mode (" + str(modes[i][0]) + "," + str(modes[i][1]) + ")")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude of the signal")

    xend = list(peak.iloc[-1:]["time"])[0]+1
    plt.xlim(right = xend)
    fig.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_path, filename), dpi=200, bbox_inches = 'tight')
    plt.close()  

def plotRelativeAmplitude(row, mode, ax, color = "r"):
    table = getPsi4Data(row, mode)
    ref = getPsi4Data(row, [2,2])

    imax = ref["ampl"].idxmax()
    ax.plot(table["time"] - table["time"][imax], table["ampl"] / ref["ampl"], color = color)

def getAmplitudeRate(row, ref_mode = [2,2], file = "data-results/amplitudeRates.csv"):
    dirName = getDirName(row)
    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) 
    
    modes = [tuple(getMode(file)) for file in dataFiles]
    values = {mode: getPsi4Data(row, mode).max()["ampl"] for mode in modes}
    ref = values[tuple(ref_mode)]

    values = [[mode[0], mode[1], values[mode]/ref] for mode in modes]
    with open(file, 'a') as output:
        for value in values:
            output.write(str(row) + "," + str(value[0]) + "," + str(value[1]) + "," + str(value[2]) + "\n")

# Numerical integration functions 
def integrate(datax, datay, end, start = 0):
    area = 0
    i = start
    while i < end:
        area += (datax[i+1] - datax[i]) * datay[i+1]
        i += 1
    area += 1/2 * (datax[1] - datax[0]) * datay[0]
    area -= 1/2 * (datax[end] - datax[end-1]) * datay[end]

    return area

def integration_step(integral, datax, datay, i):
    return integral + 1/2 * (datax[i+1] - datax[i]) * (datay[i] + datay[i+1])

# Functions to compute the luminosity of the merger
def sumOfSquares(arr):
    result = 0
    for value in arr:
        result += value*value 

    return result

def getLuminosity(row):
    dirName = getDirName(row)
    downloadMergeData(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)   
    results_file = os.path.join(results_path, "luminosity.csv")

    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) 

    tables = [] 
    for file in dataFiles:
        if file.endswith(".asc"):
            tables.append(filename_to_Psi4Data(row, file))
    realIntegrals = [0 for table in tables]
    imagIntegrals = [0 for table in tables]
        
    with open(results_file, 'w') as outLum:
        outLum.write("Time,Luminosity\n")

    sum_of_modes = 0
    i = 1
    length = len(tables[0]["time"]) - 1
    
    while i < length:
        for j, table in enumerate(tables):
                datax = table["time"]
                realDatay = table["real"]
                imagDatay = table["imag"]
                realIntegrals[j] += 0.5 * (datax[i] - datax[i-1]) * (realDatay[i] + realDatay[i-1])
                imagIntegrals[j] += 0.5 * (datax[i] - datax[i-1]) * (imagDatay[i] + imagDatay[i-1])

        sum_of_modes = sumOfSquares(realIntegrals)
        sum_of_modes += sumOfSquares(imagIntegrals)
        # From natural units to IS to ergs/s 
        luminosity = sum_of_modes / (16*np.pi)  * 2.99792458**5 * 1e40 / 6.67430 * 1e11 * 1e7 
        
        time = datax[i]
        with open(results_file, 'a') as outLum:
            outLum.write(str(time))
            outLum.write(",")
            outLum.write(str(luminosity))
            outLum.write("\n")
        i += 1

    return i 

def getMaxLuminosity(row, force = False):
    dirName = getDirName(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)   
    luminosity_path = os.path.join(results_path, "luminosity.csv")
    results_file = os.path.join(results_path, "maxLuminosity.dat")

    if not os.path.isfile(luminosity_path) or force:
        getLuminosity(row)

    luminosity = pd.read_csv(luminosity_path, index_col = 0)
    maxLum = str(luminosity.max())
    idxMaxLum = str(luminosity.idxmax())

    with open(results_file, 'w') as output:
        output.write(maxLum)
        output.write("\n")
        output.write(idxMaxLum)

    return maxLum

# Functions to compute the Stokes Parameter
def getStokes(row):
    dirName = getDirName(row)

    results_path = os.path.join(results, dirName)
    os.makedirs(results_path, exist_ok=True)   
    results_file = os.path.join(results_path, "stokes.csv")
 
    data_path = os.path.join(tmp, dirName)
    dataFiles = os.listdir(data_path) 

    tables = [] 
    for file in dataFiles:
        if file.endswith(".asc"):
            tables.append(filename_to_Psi4Data(row, file))

    result = 0
    for nTable,table in enumerate(tables):

        print("Table " + str(nTable+1) + "/21", end="\r")
        
        time = table["time"]
        real = table["real"]
        imag = table["imag"]
            
        realIntegrals = [0]
        imagIntegrals = [0]
        for i in range(len(time)-1):  
            realIntegrals.append(integration_step(realIntegrals[i], time, real, i))
            imagIntegrals.append(integration_step(imagIntegrals[i], time, imag, i))
        integrals = np.array([[realIntegrals[i] for i in range(len(time))], [imagIntegrals[i] for i in range(len(time))]]) # Dimensión (2, x)
        factors = np.array([[imag[i] for i in range(len(time))], [-real[i] for i in range(len(time))]]) # Dimension (2, x)
        integrals = (factors * integrals).sum(0)
        result = result + integrate(time, integrals, len(time)-1)
            
    result = result * 2 * np.pi
 
    with open(results_file, 'w') as output:
        output.write(str(row["number"]) + "," + str(result) + "\n")

    return result

# Testing if the merging blackholes have aligned spins
def bh_aligned(row):
    x1 = np.array([row["chi_1x"], row["chi_1y"], row["chi_1z"]])
    x2 = np.array([row["chi_2x"], row["chi_2y"], row["chi_2z"]])
    mod_x1 = np.linalg.norm(x1)
    mod_x2 = np.linalg.norm(x2)
    if mod_x1 == 0 or mod_x2 == 0:
        return False
    else:
        return (np.array_equal(x1/mod_x1, x2/mod_x2) or np.array_equal(x1/mod_x1, -x2/mod_x2))

def test_are_aligned(x1, x2):
    mod_x1 = np.linalg.norm(x1)
    mod_x2 = np.linalg.norm(x2)
    if mod_x1 == 0 or mod_x2 == 0:
        return False
    else:
        return (np.array_equal(x1/mod_x1, x2/mod_x2) or np.array_equal(x1/mod_x1, -x2/mod_x2))

# Computing chi_eff and chi_p
def getChi_eff(row):
    m1, m2 = getInitialMasses(row)
    chi1, chi2 = [row.loc["chi_1z"], row.loc["chi_2z"]]

    return (m1*chi1 + m2*chi2)/(m1+m2)

def getChi_p(row):
    chi1 = np.sqrt(np.power(row.loc["chi_1x"], 2) + np.power(row.loc["chi_1y"], 2)) 
    chi2 = np.sqrt(np.power(row.loc["chi_2x"], 2) + np.power(row.loc["chi_2y"], 2)) 
    q = row.loc["q"]

    return np.max([chi1, (2 + 1.5*q)/(q*q) * chi2])

