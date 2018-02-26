# EL-long-tail-phenomena
Systematic study of long tail phenomena in the task of entity linking

### 1. Description of the data structure

#### 1.1. Notebooks

There are three main notebooks that carry out the reported analysis:
* `Load.ipynb` loads the evaluation datasets, runs the entity linking systems on them, and stores the results in the `bin/` folder.
* `Data Analysis.ipynb` performs analysis of the datasets' properties.
* `System Analysis.ipynb` performs analysis of the system performance in relation to the data properties.

#### 1.2. Python files

There are several Python files with utility functions that are used in these notebooks:
* `classes.py` contains definition of the main classes we deal with in this project. Above all, these are: EntityMention and NewsItem.
* `dataparser.py` contains functions to load data in different formats, and produce class objects as instances.
* `load_utils.py` contains additional functions that are helpful with the data loading process.
* `analysis_utils.py` contains functions for manipulation and conversion of the data, for analysis, and for evaluation.
* `plot_utils.py` has several main plotting functions.

#### 1.3. Directories

* `bin/` contains the news item objects as loaded from raw data, and their processed versions by each of the systems.
* `data/` contains the input datasets in their original format.
* `debug/` contains additional debugging logs or files.
* `img/` stores the plot images (in PNG format) created in this project as an output.
