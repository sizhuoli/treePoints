# treePoints

A python package to build a tree database from remote sensing data.


![Build Status](https://img.shields.io/badge/build-passing-green)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)
![PyPI version](https://img.shields.io/pypi/v/treepoints)
![PyPI page](https://pypi.org/project/treepoints/)


## Table of Contents

- [Installation](#installation)
- [Demo Usage](#demo-usage)
- [Features](#features)
- [Acknowledgements](#acknowledgements)


## Installation

To install the package [treePoints](https://pypi.org/project/treepoints/), follow the steps below:

### Step 1: Create a Specific Conda Environment

1. First, ensure that Conda is installed on your system.
2. Download the required conda environment file `environment.yml` from this GitHub [repository](https://github.com/sizhuoli/treePoints).
3. Create the Conda environment using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

### Step 2: Activate the Conda Environment

After creating the Conda environment, activate it using the following command:

```bash
conda activate tf2151full_treepoints
```

### Step 3: Install the Package

#### Option 1: Install the Package from PyPI

* ( install the package directly, see deployment structure in [repository](https://github.com/sizhuoli/treePoints))

```
pip install treepoints
```

#### Option 2: Install the Package from a cloned [repository](https://github.com/sizhuoli/treePoints):

* ( clone to run demo test examples )

```bash
git clone https://github.com/sizhuoli/treePoints.git
cd treePoints
pip install .
```


## Demo Usage

See [repository](https://github.com/sizhuoli/treePoints).

```bash
python test_example/demo_predict.py
```

Set configs in test_example/config/hyperps.yaml


## Features
..

## Acknowledgements
..