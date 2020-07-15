# DeepPhase
 Protein Crystal Classification Using Machine Learning

## Requirements:
1. Get this script:

`git clone https://github.com/sarisabban/DeepPhase.git`

2. To just run the neural network install the following packages:

`pip3 install numpy pandas sklearn keras tensorflow`

## Description:
The first part of this project is to classify proteins from their X-Ray crystal reflection data.






### Compile Dataset:
1. If you are going to compile the dataset yourself, use the following command to install conda and the required packages:

`python crystal.py setup`

NOTE: This command will result in some perminant changes to your operating system setup to ensure the correct running of the required libraries.

2. Download a list of PDB IDs from RCSB.org, make sure the all the IDs are in a single line separated by a ',' as per the RCSB standard.

3. Compile the dataset using the following command:

`python crystal.py class`

For the classification dataset

`python crystal.py phase`

For the phase dataset

### Run Training:
1. Perform training on the dataset using the following command:
`python DeepClass.py`


