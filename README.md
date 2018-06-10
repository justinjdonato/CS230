
This project is based on the starter code referecned here: (https://cs230-stanford.github.io/project-starter-code.html).

The code is best executed by running the statoil_train.py file from an IDE. In the interest of time it has not been tested against the python cmd interface.

All files with the file name format of statoil_*.py have been authored by Justin Donato. Other files may have been reworked or extended.

All configurations (experiments) were run out of the configuration file: 

\statoil_experiments\base_model\params.json not the experiments folder approach.

Brief description

- statoil_build_dataset.py: extracts data from the json files supplied by kaggle and rewirites to the training, dev and test csv files
- statoil_train.py: the master file that controls execution, by initiating the data input and executing the model
- statoil_input_fn: read data from csv, apply normalization and augmentation
- statoil_model_fn: read the experiment config data, create the model and execute the experiment
- statoil_utils.py: utilities used for augmentation