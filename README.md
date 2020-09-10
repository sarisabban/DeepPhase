# DeepPhase
 Protein structure classification from crystal structure

## Requirements:
1. Get this script:

`git clone https://github.com/sarisabban/DeepPhase.git`

2. Install the required dependencies:

`module load cuda/10.0.130 anaconda3`

`conda create -n MLenv -y`

`source activate MLenv`

`conda install hdf5 pandas==1.0.5 scikit-learn==0.23.1 numpy==1.16.6 tensorflow-gpu==2.2.0 keras-gpu==2.3.1`

`source deactivate MLenv`

## Description:
This project is an attempt to classify proteins (Helical or Non-helical) from their X-Ray crystal reflection data.

### Compile Dataset:
1. If you are going to compile the dataset yourself, use the following command to install conda and the required packages:

`python setup.py`

**NOTE:** This command will result in some perminant changes to your operating system setup to ensure the correct running of the required libraries.

2. Download a list of PDB IDs from RCSB.org, make sure the all the IDs are in a single line separated by a comma as per the RCSB standard.

3. Compile the dataset using the following command:

`python crystal.py class`

4. Vectorise and serialise the dataset. The dataset is too large to be loaded anywhere, therefore this command will build the tensors, normalise them or one-hot encode them, then serialise them, that way the dataset can be loaded. Each tensor will be a separate file.

`python crystal.py vectorise_class`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete.

### Run Training:

***THIS PROJECT IS STILL A WORK IN PROGRESS...***
