import os 
import sys
import subprocess
import urllib.request

"""
# BASH COMMAND SEQUENCE
sudo ln -s /usr/bin/python3 /usr/bin/python
sudo apt install dssp libglu1-mesa-dev freeglut3-dev mesa-common-dev scons build-essential
mkdir CCTBX
cd ./CCTBX
wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
python bootstrap.py --use-conda --python 38 --nproc=4
bash ./Miniconda3-latest-Linux-x86_64.sh
    yes
    ./miniconda3
    yes
conda create -n Cenv
conda activate Cenv
conda install -c schrodinger pymol -y
conda install -c conda-forge cctbx tqdm keras tensorflow -y
conda install -c anaconda numpy scipy biopython pandas scikit-learn h5py -y
conda install -c open3d-admin open3d -y

# IBEX SETUP
conda create -n POINT
conda activate POINT
conda install -c anaconda numpy scipy scikit-learn h5py keras-gpu tensorflow-gpu -y
conda deactivate
"""

def setup():
	'''
	Installs required dependencies for this script to work
	https://github.com/cctbx/cctbx_project/
	'''
	os.system('sudo ln -s /usr/bin/python3 /usr/bin/python')
	os.system('sudo apt install dssp libglu1-mesa-dev freeglut3-dev mesa-common-dev scons build-essential')
	version = sys.version_info
	V = str(version[0])+str(version[1])
	url = 'https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py'
	os.mkdir('CCTBX')
	os.chdir('./CCTBX')
	urllib.request.urlretrieve(url, 'bootstrap.py')
	os.system('python bootstrap.py --use-conda --python {} --nproc=1'.format(V))
	stdout = subprocess.Popen(
		'bash ./Miniconda3-latest-Linux-x86_64.sh', #bash ./CCTBX/Miniconda3-latest-Linux-x86_64.sh
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		shell=True).communicate(b'\nyes\n./miniconda3\nyes\n')
	stdout = subprocess.Popen(
		'./mc3/bin/conda create -n Cenv -c cctbx-dev -c conda-forge cctbx python={}'.format(V),
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		shell=True).communicate(b'y\n')
	os.chdir('./mc3/bin/')
	os.system('./conda init bash')
	os.system('conda activate Cenv')
	os.system('conda install -c schrodinger pymol -y')
	os.system('conda install tqdm biopython h5py pandas==1.0.5 scikit-learn==0.23.1 numpy==1.16.6 tensorflow==2.2.0 keras==2.3.1 -y')
	os.system('conda install -c open3d-admin open3d -y')
setup()
