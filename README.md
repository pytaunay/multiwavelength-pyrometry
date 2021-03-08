[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4021039.svg)](https://doi.org/10.5281/zenodo.4021039)


### Description
Set of files to compute the temperature and emissivity of a material based on 
the measured radiance at a large number of wavelengths.

The core algorithm is located in the ./algorithm folder. 
The ./article folder contains the files necessary to reproduce the figures and
 content of the associated  publication. 

### How to use
To run the scripts that are located in the ./article folder, either: 
* generate a symbolic link to the ./algorithm folder, or 
* copy the script into the root folder (where this README is located). 

To run the examples from Duvaut 1995 and Wen 2011, also generate a symlink to 
the ./data folder. If executing from the command line, add 
```python
plt.show()
```
at the end of each script so that the figures are displayed. 

Here is an example (starting from the root folder): 
```bash
cd article 
ln -s ../algorithm algorithm 
ln -s ../data data 
python3 duvaut1995.py 
```

### How to cite
Please use the associated publication:

P.-Y. C. R. Taunay, E. Y. Choueiri, "Multi-wavelength Pyrometry Based on 
Robust Statistics and Cross-Validation of Emissivity Model," Review of
Scientific Instruments, 91, 114902, 2020.

```
@article{Taunay2020,
	author = {Taunay, Pierre-Yves C. and Choueiri, Edgar Y.},
	doi= {10.1063/5.0019847},
	journal= {Review of Scientific Instruments},
	pages= {114902},
	title= {Multi-wavelength pyrometry based on robust statistics and cross-validation of emissivity model},
	volume= {91},
	year= {2020}
}
```

Pierre-Yves Taunay, 2021
