# DeepPhase
 Protein structure classification from crystal structure

## Description
**DeepClass**: This project is an attempt to classify proteins (all Helix or all Sheet) from their X-Ray crystal reflection data.

**DeepPhase**: This project is an attempt to calculate the phase of each reflection point.

## How To Use
1. Get this script:

`git clone https://github.com/sarisabban/DeepPhase.git`

### Compile Dataset:
2. If you are going to compile the dataset yourself, use the following command to install conda and the required packages:

`python setup.py`

**NOTE:** This command will result in some perminant changes to your operating system setup to ensure the correct running of the required libraries.

3. Download a list of PDB IDs from RCSB.org, make sure the all the IDs are in a single line separated by a comma as per the RCSB standard.

4. Compile the dataset using one of the following commands:

`python crystal.py --Dataset PDBIDs.txt` or `python crystal.py -D PDBIDs.txt`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete.

5. For **DeepClass** (protein classification dataset):
The dataset has to be segmented depending on the number of reflection points of each example into the following segmens
[less the 1k, 1k-10k, 10k-500k, 500k-1M, 1M+]
500k-1M and 1M+ will be voxelised
`python crystal.py --Voxelise FILENAME.csv`



Vectorise the dataset. The dataset is too large to be loaded anywhere, therefore this command will build the tensors, normalise, standerdise, or one-hot encode them, then serialise them. That way the dataset can be loaded using less RAM memory. 

If you want to serialise the vectorised dataset then replace --Vectorise (-V) with --Serialise (-S).




`python crystal.py --Vectorise TYPE FILENAME.csv NUMBER_OF_POINTS` or `python crystal.py -V DeepClass FILENAME.csv 1024`





















For **DeepPhase** (reflection point phase prediction dataset):
`python crystal.py --Vectorise TYPE FILENAME.csv NUMBER_OF_POINTS` or `python crystal.py -V DeepPhase FILENAME.csv 1024`






***THIS PROJECT IS STILL A WORK IN PROGRESS...***

### Train Network:
* COMPILED DATASET DOWNLOAD LINK

* To train the neural network on the dataset use the following command:

`python network.py`





### Perform Prediction:
* TRAINED WEIGHTS DOWNLOAD LINK
