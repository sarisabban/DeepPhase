# DeepPhase
 Protein structure classification from crystal structure

## Requirements:
1. Get this script:

`git clone https://github.com/sarisabban/DeepPhase.git`

2. Install the required dependencies:

`pip3 install numpy pandas sklearn keras tensorflow`

## Description:
This project is an attempt to classify proteins from their X-Ray crystal reflection data.

### Compile Dataset:
1. If you are going to compile the dataset yourself, use the following command to install conda and the required packages:

`python crystal.py setup`

**NOTE:** This command will result in some perminant changes to your operating system setup to ensure the correct running of the required libraries.

2. Download a list of PDB IDs from RCSB.org, make sure the all the IDs are in a single line separated by a comma as per the RCSB standard.

3. Compile the dataset using the following command:

`python crystal.py class`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete.

### Run Training:
1. Perform training on the dataset using the following command:
`python DeepClass.py`

***THIS PROJECT IS STILL A WORK IN PROGRESS...***
