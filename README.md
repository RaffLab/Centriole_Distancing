# Centriole_Distancing


## Getting Started

These instructions will provide you with a copy of the project up and running on your local machine for development and testing purposes. It was developed in Linux Mint 17.3 Cinnamon using an Anaconda Python 2.7 distribution.

### Installation

The easiest way to get started is to download and install [Anaconda](https://www.anaconda.com/distribution/) for your OS. This will have the entire scipy stack namely scikit-image, scipy, numpy and scikit-learn this project depends on. Further you will also require:

* tqdm : `sudo pip install tqdm` (for progress bar)
* tifffile : `sudo conda install -c conda-forge tifffile` (for multi-page .tif writing capability)
* opencv : `sudo conda install -c conda-forge opencv`
* keras : `sudo conda install keras==2.0.8` (only tested on this ver. others may work)
* tensorflow : `sudo conda install tensorflow-gpu==1.2.0` (only test on this ver. others may work)

After installing these prerequisites. The package can be installed using pip by running the following in the cloned GitHub folder:
```
sudo pip setup.py install
```

### Example Usage
The easiest way to use the library of functions is to see provided example pipelines under 'Examples/' folder. The recommended way is to use config files to set the experimental parameters. Inside 'Examples' there are three scripts:

Script Name | Function
------------| -------------
`run_centriole_detection.py` | Given a z-stack single timepoint image, detects all centrioles present and exports the detections and cropped patches.
`run_centriole_distancing.py` | Given cropped patches, run distancing using the trained CNN models
`export_distancing_statistics.py` | Exports individual distancing statistics into files for each embryo suitable for GraphPad plotting.

### Training Your Own Models


### Documentation

Documentation for the library can be found at . Offline html version can be found by opening docs/_build/html/index.html with any web browser. 

### Citing

If you find this library useful please cite the following:


### Author and Maintainer
**Felix Y. Zhou** @ Ludwig Institute for Cancer Research, University of Oxford 

email: felixzhou1@gmail.com

## Acknowledgements
- Mustafa Aydogan
- Laura Hankins
- Raff Lab
- Ludwig Institute for Cancer Research

## License
This project is licensed under a MIT LICENSE