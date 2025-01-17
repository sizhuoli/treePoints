# treePoints

A python package to build a tree databased from remote sensing data.

## Installation

To install the package, follow instructions below.

### Step 1: Create a Specific Conda Environment

1. First, ensure that Conda is installed on your system. You can download Conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
2. Download the environment file `environment.yml` from this repository.
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
```bash
pip install .
```


## Step 4: Demo Usage



```bash
python test_example/demo_project.py
```

Set configs in /config/hyperps.yaml