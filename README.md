# DeepPhase
 Protein structure classification from crystal X-Ray diffraction patterns

## Description
**DeepClass**: This project is an attempt to classify proteins (all Helix or all Sheet) from their X-Ray crystal diffraction patterns.

**DeepPhase**: This project is an attempt to calculate the phase of each reflection point.

## How To Use
1. Get this script:

`git clone https://github.com/sarisabban/DeepPhase.git`

### Compile Dataset:
2. If you are going to compile the dataset yourself, use the following command to install conda and the required packages:

`python setup.py`

**NOTE:** This command will result in some perminant changes to your operating system setup to ensure the correct running of the required libraries.

3. Download a list of PDB IDs from RCSB.org in a file called **IDs.txt**, make sure the all the IDs are in a single line separated by a comma as per the RCSB standard.

4. Compile the dataset using one of the following command:

`python crystal.py --Dataset IDs.txt` or `python crystal.py -D IDs.txt`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete and up to 1TB memory.

If you want to download our dataset it will be provided [here]() (CrystalDataset.csv ~247GB)

5. Serialise the dataset:
For **DeepClass** (protein classification dataset):

`python crystal.py --Serialise TYPE FILENAME.csv NUMBER_OF_POINTS` or `python crystal.py -S DeepClass CrystalDataset.csv 300`

This command will vectorises and serialise the dataset. It will first filter and collect all points between 2.8 < Resolution < 3.0, make the train/valid/tests set splits, then standerdise each set seperatly. finally it will compile all X, Y, Z, R, E features (each point's X, Y, Z coordinates as well as its Resolution and E-value) into tensors, then export each set as a serialised file. You should end up with the following files: X_train.h5, X_valid.h5, X_tests.h5, Y_train.h5, Y_valid.h5, Y_tests.h5 at the end.




























***THIS PROJECT IS STILL A WORK IN PROGRESS...***
***THIS PROJECT IS STILL A WORK IN PROGRESS...***
***THIS PROJECT IS STILL A WORK IN PROGRESS...***

#TODO:
* UPLOAD DATASET and provide LINK
* UPLOAD DEEPCLASS SERIALISED DATASET and provide LINK
* UPLOAD DEEPCLASS WEIGHTS and provide LINK
* For **DeepPhase** (reflection point phase prediction dataset):
* Perform Prediction:

