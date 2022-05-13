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

If you want to use our list of PDB IDs it is provided [here](https://www.dropbox.com/s/1ytior1eu28rbwo/IDs.txt?dl=0) (name: IDs.txt and size: ~227KB)

The dataset will compile the following information for each example: the PDB ID of the structure, its secondary structure classification (Helix or Sheet), the crystal space group, the unit cell, and finally each reflection point's X, Y, Z coordinates as well as each reflection point's Resolution and E-value. The reflection point features are referred to by shorthand here as XYZRE.

If you want to download our compiled dataset it is provided [here]() (name: CrystalDataset.csv.xz and size: ~37GB). You will need to uncompress it using the command `xz -d CrystalDataset.xz` at which point the CrystalDataset.csv dataset size will become ~250GB.

5. Serialise the dataset:
Before training, the dataset needs to be serialised into train/valid/tests sets. This is a separate step to allow the use of a dataset generator to randomly sample features and push them through the neural network:

`python crystal.py --Serialise TYPE FILENAME.csv NUMBER_OF_POINTS` example `python crystal.py -S DeepClass CrystalDataset.csv 300`

This command will vectorise and serialise the dataset. It will first filter and collect all points between 2.8 < Resolution < 3.0 and all structures that have less than 2 million total number of features (X+Y+Z+R+E). It will then make the train/valid/tests set splits and standerdise each set seperatly. Then it will compile all XYZRE features into the X features tensors (for each set) and the Helix and Sheet classes into the Y labels tensors (for each set). Finally it will export each set as a serialised file. You should end up with the following files: X_train.h5, X_valid.h5, X_tests.h5, Y_train.h5, Y_valid.h5, Y_tests.h5 at the end. This computation is RAM heavy and might require up to 1TB RAM.

If you want to download our serialise dataset it is provided [here]() (name: CrystalDataset_serialised.tar.bz2 and size: ~2GB). You will need to uncompress this file using the command `tar -jxvf CrystalDataset_serialised.tar.bz2` at which point the serialised files will have a total size of ~52GB.

The serialised dataset will have the following shapes:

```
X_train.h5 = (13856, 61502, 5)

X_valid.h5 = (4618, 61502, 5)

X_tests.h5 = (4620, 61502, 5)

Y_train.h5 = (13856,)

Y_valid.h5 = (4618,)

Y_tests.h5 = (4620,)
```

6. Training the neural network:
Once you have the serialised dataset you can run the following command to train the PointNet neural network. At the end of a successful training you should get a weights.h5 file. 

`python PointNet.py`

If you want to download our trained weights file it is provided [here]() (name: weights.h5 and size: ~MB).




















***THIS PROJECT IS STILL A WORK IN PROGRESS...***

#TODO:
* UPLOAD PDB IDs text file and provide LINK
* UPLOAD DATASET and provide LINK
* UPLOAD DEEPCLASS SERIALISED DATASET and provide LINK
* UPLOAD DEEPCLASS WEIGHTS and provide LINK
* For **DeepPhase** (reflection point phase prediction dataset):
* Perform Prediction:

