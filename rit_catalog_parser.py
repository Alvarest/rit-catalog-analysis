from lxml import html 
import requests 
import pandas as pd 
import os 
import tarfile 
import matplotlib.pyplot as plt 
import shutil 
import numpy as np

link = "https://ccrgpages.rit.edu/~RITCatalog/" # The website containing the table with the simulations of the database
cwd = os.getcwd()
dbdir = os.path.join(cwd, "../")
tmp = os.path.join(cwd, "source-data")
results = os.path.join(cwd, "results")
data_results = os.path.join(cwd, "data-results")
metadata_path = os.path.join(cwd, "metadata")

def parse_catalog(link):
    if link.startswith(('https://')):
        response = requests.get(link)
        response.raise_for_status() 
        html_content = response.content
    else:
        with open(link, 'r', encoding='utf-8') as file:
            html_content = file.read()
    
    tree = html.fromstring(html_content)
    table = tree.xpath('//table[@id="example"]')[0] 
    
    headers = []
    for th in table.xpath('.//th'):
        headers.append(th.text_content().strip())
    headers = headers[:len(headers)//2] # Porque se repite al inicio y al final
    headers[2] = "medatadata_link"
    headers.insert(3, "psi4_link")
    headers.insert(4, "strain_link")
    
    data = []
    for row in table.xpath('.//tr')[1:]: # The header has been already read 
        row_data = []
        for i, td in enumerate(row.xpath('.//td')):
            # In the second column we have the three links to the data of the simulation
            if i == 2: 
                # First the metadata link
                metadata_link = td.xpath(".//a[1]/@href") 
                if metadata_link:
                    row_data.append(metadata_link[0])  
                else:
                    row_data.append("")  

                # Second the psi4 data link
                psi4_link = td.xpath('.//a[2]/@href')  # Gets the 2nd <a> tag's href
                if psi4_link:
                    row_data.append(psi4_link[0])  
                else:
                    row_data.append("")  

                # And at last the strain data link
                strain_link = td.xpath('.//a[2]/@href')  # Gets the 2nd <a> tag's href
                if strain_link:
                    row_data.append(strain_link[0])
                else:
                    row_data.append("")   
            elif i > 3 and i < 21:
                row_data.append(float(td.text_content().strip())) # The rest of the columns can be taken as they are  
            else:
                row_data.append(td.text_content().strip())  
        if row_data: # This condition is here to avoid empty rows (just in case)  
            data.append(row_data)
    database = pd.DataFrame(data, columns = headers)
    database = database.rename(columns={
        "Cat. Num.Show names": "number",
        "Type": "type",
        "Simp. Prop. Dist.": "simpPropDist",
        "Init. Coord. Sep.": "initCoordSep",
        "q=m1/m2": "q",
        "χ1x": "chi_1x",
        "χ1y": "chi_1y",
        "χ1z": "chi_1z",
        "χ2x": "chi_2x",
        "χ2y": "chi_2y",
        "χ2z": "chi_2z",
        "Ecc.": "ecc",
        "Merger Time": "mergerTime",
        "Cycles 2,2": "cycles2,2",
        "χrem": "chi_rem",
        "Recoil (km/s)": "recoil",
        "Peak Lumin. (ergs/s)": "peakLumin",
        "Bibtex Keys": "bibtex"})

    return database

# Save the database to a csv file in the local memory. This allows the user to avoid
# parsing the data again from the website.
def save_csv(catalog, dbdir = dbdir, csvfile = "rit_catalog.csv"):
    database.to_csv(os.path.join(dbdir, csvfile))


def getDirName(row):
    return row['psi4_link'][5:-7]

def getFileName(row):
    return row['number'] + ".tar.gz" 

def getMetaName(row):
    return row["number"] + ".dat"
    
def downloadMergeData(row, verbose = False):
    file = getFileName(row)
    data_path = os.path.join(tmp, getDirName(row))

    if not os.path.exists(data_path):
        r = requests.get(link + row['psi4_link'], allow_redirects = True)
        os.makedirs(data_path, exist_ok = True)
        open(os.path.join(tmp, file), 'wb').write(r.content)
        
        tar = tarfile.open(os.path.join(tmp,file), 'r:gz')
        tar.extractall(path = tmp, filter='data')
        names = tar.getnames()
        dirname = names[0].split("/")
        
        if names[0] != getDirName(row):
            source_dir = os.path.join(tmp, names[0])
            os.makedirs(data_path, exist_ok = True)
            datafiles = os.listdir(source_dir)
            for datafile in datafiles:
                os.rename(os.path.join(source_dir, datafile), os.path.join(data_path, datafile))
            shutil.rmtree(os.path.join(tmp, dirname[0]))
            
        tar.close()
        if verbose: print("File " + file + " extracted successfully!")

        os.remove(os.path.join(data_path, "Metadata"))
        os.remove(os.path.join(tmp,file)) 
        if verbose: print("File " + file + " removed successfully!")
    
    return data_path

def removeMergeData(row, verbose = True):
    dir_to_remove = getDirName(row)
    data_path = os.path.join(tmp, dir_to_remove)

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        if verbose: print("Directory " + dir_to_remove + " removed successfully!")
    else:
        if verbose: print("Directory " + dir_to_remove + " doesn't exists")

def downloadMetaData(row):
    file = getFileName(row)
    data_path = os.path.join(metadata_path, getMetaName(row))

    if not os.path.exists(data_path):
        r = requests.get(link + row['metadata_link'], allow_redirects = True)
        open(data_path, 'wb').write(r.content)
    return data_path
