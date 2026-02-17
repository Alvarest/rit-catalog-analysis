# RIT Binary Black Hole Catalog Analysis Toolkit
This repository provides a Python-based toolkit for programatic interaction with the 
[RIT binary black hole simulation catalog](https://ccrg.rit.edu/content/data/rit-waveform-catalog).
The catalog contains a large set of binary black hole mergers simulations, spanning a wide range of 
physical parameters. These simulations are invaluable for interpreting real gravitational wave
observations.

Nevertheless, there is currently no official API, making it challenging for researchers and students
to efficently analyze or visualize the data. This repository aims to help with that. 

## Key Features
- **Efficent Data Access** to fetch and explore catalog data directly using **Pandas DataFrames**.
- **Easy parameter calculation** such as luminosity and spin.
- **Visualization tools** that makes it easier to visualize the simulations.

## Dependencies
This toolkit requires the following Python's packages:
- Pandas
- Numpy
- Matplotlib
- lxml

## Usage
To use this toolkit just download or clone the repository. You can import any function from the 
python files as you would normally do, for example:
```python
from rit_catalog_parser import parse_catalog
from merger_analysis import getLuminosity
```
You can find a more elaborate example of how to use this functions in the Notebook `example.ipynb`.
