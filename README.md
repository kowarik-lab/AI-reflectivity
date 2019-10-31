# AI-reflectivity
`AI-reflectivity` is a code based on artificial neural networks trained with simulated reflectivity data that quickly predicts film parameters from experimental X-ray reflectivity curves. In addition to downloading the code you can try the program without installing anything by clicking on the binder link below. 

This project has a common root with (ML-reflectivity)[https://github.com/schreiber-lab/ML-reflectivity] and evolved in parallel. Both are linked to the follwoing publication:
Fast Fitting of Reflectivity Data of Growing Thin Films Using Neural Networks A. Greco, V. Starostin, C. Karapanagiotis, A. Hinderhofer, A. Gerlach, L. Pithan, S. Liehr, F. Schreiber,  S. Kowarik (2019). J. Appl. Cryst. 

## How to use AI-reflectivity
Currently there are two ways to work with library:
1) plain python scripts
2) jupyter notebooks (demo available by clicking on the mybinder link below)

So far there is only a demo notebook that uses an already trained network to predict the relevant thin film properties.

### Live demo
For an online live demonstration using a pre-trained network have a look at
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kowarik-labs/AI-reflectivity/master?filepath=notebooks%2Fprediction_demo.ipynb)
You can upload your own files and try the neural network we trained for XRR data of organic molecular layers (20 - 300 Angstrom thick) on silicon with native oxide.

### Run the code locally
It is encouraged to use a conda environment with conda-forge as additional channel. All dependencies are listed in `requirements.txt`. To create a new enviroment on a system where conda is installed just pick a name for the new environement run the following commands in the anaconda prompt (windows) or in any terminal (linux, mac). If you prefere to use the Anaconda Navigator just have a look in `requirements.txt` and install the required packages from there.
```
   $ conda create --name YOUR_ENV_NAME --file requirements.txt --channel defaults --channel conda-forge
   $ conda activate YOUR_ENV_NAME
```

### How to train AI
Please run first `generate_training_data.py` and `training.py` after that.
```
   $ cd scripts
   $ python generate_training_data.py YOUR_CONFIG_FILE
   $ python training.py YOUR_CONFIG_FILE
```

### How to get a Fit

#### using a plain python script
```
   $ cd scripts
   $ python prediction.py YOUR_CONFIG_FILE
```

#### using jupyter notebook
Please run [jupyter](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html) inside the `notebooks` directory and use `prediction_demo.ipynb`

### Config file entries
TODO: Explain structure of config file

### Some comments about the data structure
- config relative to config file
- labels and data:
   - TODO: this notation comes from tensorflow .... 

## How to cite this project
- Cite the related Journal of Applied Crystallography publication once has been publlished
    
   Fast Fitting of Reflectivity Data of Growing Thin Films Using Neural Networks A. Greco, V. Starostin, C. Karapanagiotis, A. Hinderhofer, A. Gerlach, L. Pithan, S. Liehr, F. Schreiber,  S. Kowarik (2019). J. Appl. Cryst. 

- Cite via zenodo.org (TODO: add DOI once available)

## Authors
- Alessandro Greco (Institut für Angewandte Physik, University of Tübingen)
- Vladimir Starostin (Institut für Angewandte Physik, University of Tübingen)
- Christos Karapanagiotis (Institut für Physik, Humboldt Universität zu Berlin)
- Alexander Hinderhofer (Institut für Angewandte Physik, University of Tübingen)
- Alexander Gerlach (Institut für Angewandte Physik, University of Tübingen)
- Linus Pithan (ESRF The European Synchrotron)
- Sascha Liehr (Bundesanstalt für Materialforschung und -prüfung (BAM))
- Frank Schreiber (Institut für Angewandte Physik, University of Tübingen)
- Stefan Kowarik (Bundesanstalt für Materialforschung und -prüfung (BAM) and Institut für Physik, Humboldt Universität zu Berlin)

## Main Dependencies 
- tensorflow
- keras

##Developers Corner
- We use `black` for unified code appearance
