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

If you want to download our dataset it will be provided here (CrystalDataset.csv ~247GB)









5. Prepare dataset for training:
For **DeepClass** (protein classification dataset):

`python crystal.py --Serialise TYPE FILENAME.csv NUMBER_OF_POINTS` or `python crystal.py -S DeepClass FILENAME.csv 10000`

For the 10K-500K, 500K-1M, and 1M+ segments it is recommended to keep NUMBER_OF_POINTS to 10000 since we found this trains best in DeepClass. The other segments use 100.

If you want to serialise the vectorised the dataset then replace --Vectorise (-V) with --Serialise (-S).




























***THIS PROJECT IS STILL A WORK IN PROGRESS...***
***THIS PROJECT IS STILL A WORK IN PROGRESS...***
***THIS PROJECT IS STILL A WORK IN PROGRESS...***

### Train Network:
* COMPILED DATASET DOWNLOAD LINK

* To train the neural network on the dataset use the following command:

`python network.py`










For **DeepPhase** (reflection point phase prediction dataset):
`python crystal.py --Vectorise TYPE FILENAME.csv NUMBER_OF_POINTS` or `python crystal.py -V DeepPhase FILENAME.csv 1024`



### Perform Prediction:
* TRAINED WEIGHTS DOWNLOAD LINK
