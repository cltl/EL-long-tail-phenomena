# EL-long-tail-phenomena
This repository contains the entire code for the analysis reported in the following COLING 2018 paper:

```
  @inproceedings{ilievski2018systematic,
        title = {Systematic Study of Long Tail Phenomena in Entity Linking},
        author={Ilievski, Filip and Vossen, Piek and Schlobach, Stefan},
        booktitle={proceedings of COLING},
        year = {2018}}
```

### 1. Description of the data structure

#### 1.1. Notebooks

There are four main notebooks that carry out the reported analysis:
* `Load.ipynb` loads the evaluation datasets, runs the entity linking systems on them, and stores the results in the `bin/` folder.
* `Data Analysis.ipynb` performs analysis of the datasets' properties.
* `System Analysis (Micro).ipynb` performs analysis of the system performance measured with micro F1-score in relation to the data properties.
* `System Analysis (Macro).ipynb` performs additional analysis of system performance, but now with macro F1-score as a metric.

#### 1.2. Python files

There are several Python files with utility functions that are used in these notebooks:
* `classes.py` contains definition of the main classes we deal with in this project. Above all, these are: EntityMention and NewsItem.
* `dataparser.py` contains functions to load data in different formats, and produce class objects as instances.
* `load_utils.py` contains additional functions that are helpful with the data loading process.
* `analysis_utils.py` contains functions for manipulation and conversion of the data, for analysis, and for evaluation.
* `plot_utils.py` has several plotting functions.

#### 1.3. Directories

* `bin/` contains the news item objects as loaded from raw data, and their processed versions by each of the systems.
* `data/` contains the input datasets in their original format.
* `debug/` contains additional debugging logs or files.
* `img/` stores the plot images (in PNG format) created in this project as an output.

### 2. Before running the project

#### 2.1. Dependencies

We use the usual Python modules for statistical analysis, data manipulation, and plotting: scipy, numpy, collections, matplotlib, seaborn, pickle, collections, pandas, rdflib, and lxml.

We also use the Redis database and its Python client to cache some data.

This project has been coded and tested on a computer with Python v3.6.

Hence, please make sure that:
* **You can run a Python version 3.6 or similar**
* **You have installed the modules specified above.**
* **You have installed Redis and its [Python client](https://pypi.python.org/pypi/redis) on your machine**

#### 2.2. Preparing complementary DBpedia data: PageRank values, redirects, and disambiguation values

We pre-load three types of DBpedia data in a Redis database for quick access during the analysis. These are:
* PageRank values from [http://people.aifb.kit.edu/ath/](http://people.aifb.kit.edu/ath/).
* disambiguation links and redirect links from [http://wiki.dbpedia.org/downloads-2016-04](http://wiki.dbpedia.org/downloads-2016-04).

For replication purposes, please simply run the bash script `prepare_redis_data.sh` to: download these files, place them in the correct place locally, and cache them in Redis. Note that this script takes some time to execute (around 1.5-2 hours on my laptop).

#### 2.3. Download evaluation data: N3 corpus and AIDA

The N3-collection of datasets is publicly available [on github](https://github.com/dice-group/n3-collection). For your convenience, we prepared a script that downloads this collection and creates the directory assumed by the analysis notebooks. You can run this script without arguments, as follows:

`sh prepare_n3_data.sh`

The second data collection, AIDA, is unfortunately not publicly available. Please get in touch with the paper authors if you are unable to obtain this data.

**To enable easier replicability and work around the proprietary dataset AIDA, we provide the .bin files that are created in the first step in this project by the `Load.ipynb` notebook. This allows users to avoid the obstacle of a private/paid dataset, re-runing the EL linking systems and storing the data on disk themselves, while still enabling them to run the entire analysis that supports the paper findings and inspect the underlying data. Note that these .bin files contain only the set of entities with their mentions, thus still preserving the commercial aspect of AIDA.**

### 3. Running the project

Section 4 of the paper is based on the notebook `Data Analysis.ipynb`. This notebook uses the dataset `.bin` objects as downloaded from github or pre-cached by the `Load.ipynb` notebook.

Section 5 is based mostly on the notebook `System Analysis (Micro).ipynb`, and complemented by some findings from the notebook `System Analysis (Macro).ipynb`. These two notebooks rely on the `.bin` objects that are downloaded from github or pre-cached by the `Load.ipynb` notebook, containing the system disambiguation output by all systems on the datasets.

Feel welcome to rerun the notebook to validate and inspect their results. All analysis notebooks run really quick (within a minute). The `Load.ipynb` notebook takes longer to run (around 2 hours in total) and expects one non-public dataset, AIDA (as discussed in 2.3), hence perhaps it is a good idea not to make use of the provided `.bin` files.

The green markers inside the notebooks help the reader relate the analysis in the notebook to the plots in the paper. For example, in `Data analysis` part 6) PageRank distribution of instances, we have a pointer "Section 4.2 of the paper".

### 4. Contact

For any questions or comments, please contact Filip via e-mail: `f.ilievski@vu.nl`.
