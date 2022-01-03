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

3. Download a list of PDB IDs from RCSB.org in a file called **IDs.txt**, make sure the all the IDs are in a single line separated by a comma as per the RCSB standard.

4. Compile the dataset using one of the following commands:

`python crystal.py --Dataset PDBIDs.txt` or `python crystal.py -D PDBIDs.txt`

**NOTE:** Depending on the number of structures you are using to compile the dataset this may take from several hours to several days to compelete.


--------------------- THIS SECTION HAS CHANGES -----------------------
NOT USING ANY MORE !!!! CHECK !!!!!!



If you want to compile a dataset of computer generated reflections from PDB files and augment each of these example then use the following command:

`python crystal.py --Augment NUMBER` or `python crystal.py -A 10`

Where NUMBER is the number of augments to each example.

to export the augments as .mtz files use the collosing command:
`python crystal.py --Augment NUMBER MTZ` or `python crystal.py -A 10 MTZ`


--------------------- THIS SECTION HAS CHANGES -----------------------
Fix info about voxelisation
Fix info about data generation
Fix info about vectorisation


5. Prepare dataset for training:
For **DeepClass** (protein classification dataset):
The dataset has to be segmented depending on the number of reflection points of each example into the following segmens: less the 1K, 1K-10K, 10K-500K, 500K-1M, and more than 1M. The 500k-1M and 1M+ will be voxelised while the others will not. Run the following command:

`python crystal.py --Voxelise FILENAME.csv` or `python crystal.py -V FILENAME.csv`

The dataset is too large to be loaded anywhere, therefore the segmented (and voxelised) datasets has to be vectorised (or serialised to save this step into files). This command will build the tensors, normalise, standerdise or one-hot encode them, then serialise them. That way the dataset can be loaded using less RAM memory. 

`python crystal.py --Vectorise TYPE FILENAME.csv NUMBER_OF_POINTS` or `python crystal.py -V DeepClass FILENAME.csv 10000`

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
