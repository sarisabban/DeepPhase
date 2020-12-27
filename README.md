# DeepPhase
 Protein structure classification from crystal structure

## Requirements
1. Get this script:

`git clone https://github.com/sarisabban/DeepPhase.git`

2. Install the required dependencies:

`module load cuda/10.0.130 anaconda3`

`conda create -n MLenv -y`

`source activate MLenv`

`conda install hdf5 pandas==1.0.5 scikit-learn==0.23.1 numpy==1.16.6 tensorflow-gpu==2.2.0 keras-gpu==2.3.1`

`source deactivate MLenv`

## Description
DeepClass: This project is an attempt to classify proteins (Helical or Non-helical) from their X-Ray crystal reflection data.

DeepPhase: This project is an attempt to calculate the phase of each reflection point.

## How To Use
### Compile Dataset:
1. If you are going to compile the dataset yourself, use the following command to install conda and the required packages:

`python setup.py`

**NOTE:** This command will result in some perminant changes to your operating system setup to ensure the correct running of the required libraries.

2. Download a list of PDB IDs from RCSB.org, make sure the all the IDs are in a single line separated by a comma as per the RCSB standard.

3. Compile the dataset using one of the following commands:

`python crystal.py --Dataset PDBIDs.txt` or `python crystal.py -D PDBIDs.txt`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete.

4. Vectorise and serialise the dataset. The dataset is too large to be loaded anywhere, therefore this command will build the tensors, normalise, standerdise, or one-hot encode them, then serialise them. That way the dataset can be loaded using less RAM memory. 

To cover more points they will be randomly samples, each file will contain randomly samples NUMBER_OF_POINTS, to cover more points we can repeat the sampling for NUMBER_OF_SAMPLES. Each tensor will end up in a separate file

For DeepClass (protein classification dataset):
`python crystal.py --Vectorise class FILENAME.csv NUMBER_OF_POINTS NUMBER_OF_SAMPLES class` or `python crystal.py -V class FILENAME.csv 15000 3`

For DeepPhase (reflection point phase prediction dataset):
`python crystal.py --Vectorise phase FILENAME.csv NUMBER_OF_POINTS NUMBER_OF_SAMPLES` or `python crystal.py -V phase FILENAME.csv 15000 3`

### Run Training:

TRAINING CODE AND COMMAND

COMPILED DATASET DOWNLOAD LINK

TRAINED WEIGHTS DOWNLOAD LINK

***THIS PROJECT IS STILL A WORK IN PROGRESS...***
