# AI-reflectivity
`AI-reflectivity` is a code based on artificial neural networks trained with simulated reflectivity data that quickly predicts film parameters from experimental X-ray reflectivity curves.

## How to use AI-reflectivity
Currently there are two ways to work with library:
1) plain python scipts
2) jupyter notebooks

So far there is only a demo notebook that uses an already trained network to predict the relevant thin film properties.

### Live demo
TODO link with mybinder.org

### Run the code locally
It is encuraged to use a conda environment with conda-forge as additional channel. All dependencies are listed in `requirements.txt` and may e.g. be installed via
```
   $ conda create --name <env> --file requirements.txt
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
Pleas run (jupyter)[https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html] inside the `notebooks` directory and use `prediction_demo.ipynb`

### Config file entries
TODO: Explain structure of config file

### Some comments about the data structure
- config relative to config file
- labels and data:
   - TODO: this notation comes from tensorflow .... 

## How to cite this project
- Cite the related Journal of Applied Crystallography publication once has been accepted (TODO: add link once available )
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
- relf1D  TODO: Where do we depend on this package?

##Developers Corner
- We use `black` for unified code appearance
