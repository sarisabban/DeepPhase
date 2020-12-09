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

To compile the dataset for DeepClass

`python crystal.py --Class PDBIDs.txt` or `python crystal.py -C PDBIDs.txt`

To compile the dataset for DeepPhase

`python crystal.py --Phase PDBIDs.txt` or `python crystal.py -P PDBIDs.txt`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete.

4. Vectorise and serialise the dataset. The dataset is too large to be loaded anywhere, therefore this command will build the tensors, normalise them or one-hot encode them, then serialise them. That way the dataset can be loaded using less RAM memory. Each tensor will be a separate file. Since there are a large number of reflection points and a large variation in the number of reflection points between files, these commands can also allow you to choose the top reflection points sorted by E-value. Any gaps in the dataset will be padded with zeros.

To vectorise and serialise all the reflection points for DeepClass:

`python crystal.py --VecClass FILENAME.csv all` or `python crystal.py -vC FILENAME.csv all`

To vectorise and serialise randomly sampled 10,000 points and repeat the process 3 times used the following command: 

`python crystal.py --VecClass FILENAME.csv 10000 3` or `python crystal.py -vC FILENAME.csv 10000 3`

To vectorise and serialise all the reflection points for DeepPhase (can only vectorise all points):

`python crystal.py --VecPhase FILENAME.csv` or `python crystal.py -vP FILENAME.csv`

### Run Training:

TRAINING CODE AND COMMAND

COMPILED DATASET DOWNLOAD LINK

TRAINED WEIGHTS DOWNLOAD LINK

***THIS PROJECT IS STILL A WORK IN PROGRESS...***
