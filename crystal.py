import os
import sys
import h5py
import tqdm
import math
import keras
import random
import urllib
import Bio.PDB
import argparse
import statistics
import numpy as np
import open3d as o3d
import urllib.request
from multiprocessing import Pool
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

parser = argparse.ArgumentParser(description='Compiling, vectorising, voxelise, and train on X-ray crystal diffraction datasets')
parser.add_argument('-U', '--Setup'    , action='store_true', help='Compile a datset of protein reflections points, include a text file of PDB IDs')
parser.add_argument('-D', '--Dataset'  , nargs='+',           help='Compile a datset of protein reflections points, include a text file of PDB IDs')
parser.add_argument('-V', '--Vectorise', nargs='+',           help='Vectorise the datset only')
parser.add_argument('-S', '--Serialise', nargs='+',           help='Vectorise and serialise the datset')
parser.add_argument('-A', '--Augment',   nargs='+',           help='Augment a .pdb file to different orientations and generate reflection data')
parser.add_argument('-X', '--Voxelise',  nargs='+',           help='Voxelise the points of in a .csv file')
parser.add_argument('-G', '--Generator', nargs='+',           help='Generate batches of data and push them through the network on the fly')
args = parser.parse_args()

def setup():
	'''
	Installs required dependencies for this script to work
	https://github.com/cctbx/cctbx_project/
	----------------------------------------------------------------------------
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
		'bash ./Miniconda3-latest-Linux-x86_64.sh',
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
	os.system('conda install tqdm biopython h5py scipy pandas==1.0.5 scikit-learn==0.23.1 numpy==1.16.6 tensorflow==2.2.0 keras==2.3.1 -y')
	os.system('conda install -c open3d-admin open3d -y')

class Dataset():
	'''
	Build a .csv dataset for protein X-ray crystal diffraction points
	from .mtz and .pdb files and compile all
	'''
	def __init__(self, PDB_MTZ='MTZ', d=2.5, n=10, augment=False):
		self.PDB_MTZ = PDB_MTZ
		self.d = d
		self.n = n
		self.augment = augment
		if self.PDB_MTZ == 'MTZ': self.n = 1
		elif augment == False: self.n = 1
	def download(self, ID):
		''' Downloads a structure's .mtz and .pdb files. '''
		ID = ID.lower()
		Murl = 'http://edmaps.rcsb.org/coefficients/{}.mtz'.format(ID)
		Purl = 'https://files.rcsb.org/download/{}.pdb'.format(ID)
		if self.PDB_MTZ == 'MTZ':
			urllib.request.urlretrieve(Murl, '{}.mtz'.format(ID))
			urllib.request.urlretrieve(Purl, '{}.pdb'.format(ID))
		elif self.PDB_MTZ == 'PDB':
			urllib.request.urlretrieve(Purl, '{}.pdb'.format(ID))
	def Ref_MTZ(self, filename):
		'''
		Extracts Space group, Unit Cell, X, Y, Z, Resolution,
		E-value, and Phase from .mtz files. More info can be
		found here:
		https://cci.lbl.gov/cctbx_docs/cctbx/cctbx.miller.html
		'''
		from iotbx.reflection_file_reader import any_reflection_file
		hkl_file = any_reflection_file(filename)
		arrays = hkl_file.as_miller_arrays(merge_equivalents=False)
		for a in arrays:
			label = str(a.info()).split(':')[-1].split(',')[-1]
			if label == 'PHIC':
				# Cell Dimentions
				UC = a.unit_cell()
				C = str(UC)[1:-1]
				C = tuple(map(str, C.split(', ')))
				# P1 expand
				P1 = a.expand_to_p1().indices()
				# Resolution
				R = list(UC.d(P1))
				# Space Group
				ms_base = a.customized_copy()
				ms_all = ms_base.complete_set()
				ms=ms_all.customized_copy(space_group_info=a.space_group_info())
				S = str(ms).split()[-1].split(')')[0]
				# Convert miller hkl to polar
				polar_coordinates = list(UC.reciprocal_space_vector(P1))
				X = []
				Y = []
				Z = []
				for x, y, z in polar_coordinates:
					X.append(x)
					Y.append(y)
					Z.append(z)
				# Phase
				P = list(a.expand_to_p1().phases().data())
		for a in arrays:
			label = str(a.info()).split(':')[-1].split(',')[-1]
			if label == 'FOM':
				# E-values in P1 space group
				a.setup_binner(auto_binning=True)
				a.binner()
				e_val = a.quasi_normalize_structure_factors()
				E = list(e_val.expand_to_p1().f_sq_as_f().data())
		return(S, C, X, Y, Z, R, E, P)
	def Ref_PDB(self, pdbstr, export_mtz=False):
		'''
		Generate Space group, Unit Cell, X, Y, Z, Resolution,
		E-value, and Phase from .pdb files. More info can be
		found here:
		https://cci.lbl.gov/cctbx_docs/cctbx/cctbx.miller.html
		'''
		import iotbx.pdb
		xrs = iotbx.pdb.input(source_info=None, lines=pdbstr)\
		.xray_structure_simple()
		a = xrs.structure_factors(d_min=self.d).f_calc()
		# Cell Dimentions
		UC = a.unit_cell()
		C = str(UC)[1:-1]
		C = tuple(map(str, C.split(', ')))
		# Space Group
		S = str(a.customized_copy()).split()[-1].split(')')[0]
		# P1 expand
		P1 = a.expand_to_p1().indices()
		# Resolution
		R = [round(x, 5) for x in list(UC.d(P1))]
		# Phase
		P = [round(x, 5) for x in list(a.expand_to_p1().phases().data())]
		amp = a.amplitudes()
		amp.setup_binner(auto_binning=True)
		amp.binner()
		e_val = amp.quasi_normalize_structure_factors()
		# E-values in P1 space group
		E = list(e_val.expand_to_p1().f_sq_as_f().data())
		E = [round(x, 5) for x in E]
		# Convert miller hkl to polar
		polar_coordinates = list(UC.reciprocal_space_vector(P1))
		X = []
		Y = []
		Z = []
		for x, y, z in polar_coordinates:
			X.append(round(x, 5))
			Y.append(round(y, 5))
			Z.append(round(z, 5))
		# Export as .mtz file
		if export_mtz == True:
			mtz_dataset = a.as_mtz_dataset(column_root_label='FC')
			mtz_object = mtz_dataset.mtz_object()
			mtz_object.write(file_name='temp.mtz')
		return(S, C, X, Y, Z, R, E, P)
	def Augment(self, filename):
		''' Augment a .pdb file's molecular position and unit cell '''
		import pymol
		pymol.cmd.load(filename)
		name = pymol.cmd.get_names()[0]
		# Flip molecule
		x  = random.randint(1, 10)
		y  = random.randint(1, 10)
		z  = random.randint(1, 10)
		xr = random.randint(1, 360)
		yr = random.randint(1, 360)
		zr = random.randint(1, 360)
		pymol.cmd.translate([x, y, z], name)
		pymol.cmd.rotate([1, 0, 0], xr, name)
		pymol.cmd.rotate([0, 1, 0], yr, name)
		pymol.cmd.rotate([0, 0, 1], zr, name)
		pymol.cmd.save('temp.pdb', name)
		# Return molecule
		pymol.cmd.rotate([0, 0, 1], -zr, name)
		pymol.cmd.rotate([0, 1, 0], -yr, name)
		pymol.cmd.rotate([1, 0, 0], -xr, name)
		pymol.cmd.translate([-x, -y, -z], name)
		# Augment crystal unit cell
		with open(filename, 'r') as f:
			for line in f:
				tag = line.strip().split()[0]
				if tag == 'HEADER': header = line
				if tag == 'CRYST1': crystal = line
			L = crystal.strip().split()
			ID        = L[0]
			UCeA      = float(L[1])
			UCeB      = float(L[2])
			UCeC      = float(L[3])
			UCaAlpha  = float(L[4])
			UCaBeta   = float(L[5])
			UCaGamma  = float(L[6])
			Space     = ' '.join(L[7:-1])
			Z         = L[-1]
			UCeA      += random.choice([-1*random.random(), 1*random.random()])
			UCeB      += random.choice([-1*random.random(), 1*random.random()])
			UCeC      += random.choice([-1*random.random(), 1*random.random()])
			#UCaAlpha  += random.choice([-1*random.random(), 1*random.random()])
			#UCaBeta   += random.choice([-1*random.random(), 1*random.random()])
			#UCaGamma  += random.choice([-1*random.random(), 1*random.random()])
			UCeA      = round(UCeA, 3)
			UCeB      = round(UCeB, 3)
			UCeC      = round(UCeC, 3)
			UCaAlpha  = round(UCaAlpha, 3)
			UCaBeta   = round(UCaBeta, 3)
			UCaGamma  = round(UCaGamma, 3)
			crystal = '{:6} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:<10} {:>4}\n'\
			.format(ID, UCeA, UCeB, UCeC, UCaAlpha, UCaBeta, UCaGamma, Space, Z)
		# Export augmented structure
		with open('temp.pdb', 'r') as t: aug = t.readlines()
		augmented = header + crystal
		for i in aug: augmented += i
		os.remove('temp.pdb')
		return(augmented)
	def labels(self, filename):
		structure = Bio.PDB.PDBParser(QUIET=True).get_structure('X', filename)
		dssp = Bio.PDB.DSSP(structure[0], filename, acc_array='Wilke')
		SS = []
		for aa in dssp:
			if   aa[2] == 'G' or aa[2] == 'H' or aa[2] == 'I': ss = 'H'
			elif aa[2] == 'B' or aa[2] == 'E':                 ss = 'S'
			elif aa[2] == 'S' or aa[2] == 'T' or aa[2] == '-': ss = 'L'
			SS.append(ss)
		H = SS.count('H')
		S = SS.count('S')
		L = SS.count('L')
		H_frac = H/len(SS)
		S_frac = S/len(SS)
		L_frac = L/len(SS)
		return(H_frac, S_frac, L_frac)
	def run(self, IDs='IDs.txt', EXP_MTZ=False):
		with open('CrystalDataset.csv', 'a') as F:
			h1 = 'PDB_ID,Class,Space Group,'
			h2 = 'Unit-Cell a,Unit-Cell b,Unit-Cell c,'
			h3 = 'Unit-Cell Alpha,Unit-Cell Beta,Unit-Cell Gamma'
			h4 = ',X,Y,Z,Resolution,E-value,Phase\n'
			head = h1 + h2 + h3 + h4
			F.write(head)
			size = []
			with open(IDs) as f:
				line = f.read().strip().lower().split(',')
				for item in tqdm.tqdm(line):
					try:
						self.download(item)
					except:
						red = '\u001b[31m'
						ret = '\u001b[0m'
						print('{}[-] {} Failed: could not download{}'\
						.format(red, item.upper(), ret))
						continue
					for count in range(self.n):
						if self.PDB_MTZ == 'MTZ':
							try:
								Mfilename = item + '.mtz'
								Pfilename = item + '.pdb'
								S, C, X, Y, Z, R, E, P = self.Ref_MTZ(Mfilename)
								os.remove(Mfilename)
							except:
								red = '\u001b[31m'
								ret = '\u001b[0m'
								print('{}[-] {} Failed: problem compiling{}'\
								.format(red, item.upper(), ret))
								os.remove(Mfilename)
								continue
						if self.PDB_MTZ == 'PDB':
							try:
								Pfilename = item + '.pdb'
								if self.augment == True:
									pdbstr = self.Augment(Pfilename)
								elif self.augment == False:
									with open(Pfilename) as f: pdbstr = f.read()
								try:
									if EXP_MTZ == True:
										S,C,X,Y,Z,R,E,P = self.Ref_PDB(pdbstr,
										export_mtz=True)
										os.rename('temp.mtz', '{}_{}.mtz'\
										.format(Pfilename[:-4], count+1))
									elif EXP_MTZ == False:
										S,C,X,Y,Z,R,E,P = self.Ref_PDB(pdbstr)
								except:
									continue
							except:
								red = '\u001b[31m'
								ret = '\u001b[0m'
								print('{}[-] {} Failed: problem compiling{}'\
								.format(red, item.upper(), ret))
								continue
						H_frac, S_frac, L_frac = self.labels(Pfilename)
						if   H_frac>=0.50 and S_frac==0.00 and L_frac<=0.50:
							label = 'Helix'
						elif H_frac==0.00 and S_frac>=0.50 and L_frac<=0.50:
							label = 'Sheet'
						elif H_frac>=S_frac and H_frac>=L_frac:
							label = 'Mostly Helix'
						elif S_frac>=H_frac and S_frac>=L_frac:
							label = 'Mostly Sheet'
						elif L_frac>=H_frac and L_frac>=S_frac:
							label = 'Mostly Loops'
						else:
							ERROR = 'error in secondary structures'
							print('\u001b[31m[-] {} Failed: {}\u001b[0m'\
							.format(item.upper(), ERROR))
							break
						assert len(X)==len(Y)==len(Z)==len(R)==len(E)==len(P),\
						'\u001b[31m[-] {} Failed: values not equal\u001b[0m'.\
						format(item.upper())
						exp = [S]
						a     = C[0]
						b     = C[1]
						c     = C[2]
						alpha = C[3]
						beta  = C[4]
						gamma = C[5]
						exp.append(a+','+b+','+c+','+alpha+','+beta+','+gamma)
						for x, y, z, r, e, p in zip(X, Y, Z, R, E, P):
							x = str(round(x, 5))
							y = str(round(y, 5))
							z = str(round(z, 5))
							r = str(round(r, 5))
							e = str(round(e, 5))
							p = str(round(p, 5))
							exp.append(x+','+y+','+z+','+r+','+e+','+p)
						example = ','.join(exp)
						TheID = item
						if self.n != 1: TheID = '{}_{}'.format(TheID, count+1)
						F.write(TheID.upper()+','+label+',')
						F.write(example + '\n')
						size.append(len(X))
					os.remove(Pfilename)

def Vectorise(filename='CrystalDataset.csv', max_size='15000', Type='DeepClass',
	fp=np.float32, ip=np.int32):
	'''
	Since the .csv file for this dataset would require larger RAM than what is
	currently available, yet we still need to train a network on as much
	reflection points as possible this function is used to overcome this
	limitation by randomly sampling max_size number of points from each example,
	vectorise these points, normalise/standerdises them, construct the final
	tensors then output the resulting tensors
	'''
	I = np.array([])
	L = np.array([])
	S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
	max_size = int(max_size)
	with open(filename, 'r') as f:
		next(f)
		for line in f:
			line = line.strip().split(',')
			# 1. Isolate PDB IDs and labels
			I = np.append(I, np.array(str(line[0]), dtype=str))
			L = np.append(L, np.array(str(line[1]), dtype=str))
			S.append(np.array(int(line[2]), dtype=ip))
			UCe.append(np.array([float(i) for i in line[3:6]], dtype=fp))
			UCa.append(np.array([float(i) for i in line[6:9]], dtype=fp))
			# 2. Isolate points
			T = line[9:]
			T = [float(i) for i in T]
			# 3. Collect each point values
			NC = [(x, y, z, r, e, p) for x, y, z, r, e, p
				in zip(T[0::6], T[1::6], T[2::6], T[3::6], T[4::6], T[5::6])]
			# 4. Sample points at regular intervals
			T = NC[::len(NC)//max_size][:max_size]
			assert len(T) == max_size, 'Max number of points incorrect'
			T = [i for sub in T for i in sub]
			# 5. Export points
			X.append(np.array(T[0::6], dtype=fp))
			Y.append(np.array(T[1::6], dtype=fp))
			Z.append(np.array(T[2::6], dtype=fp))
			R.append(np.array(T[3::6], dtype=fp))
			E.append(np.array(T[4::6], dtype=fp))
			P.append(np.array(T[5::6], dtype=fp))
	# 6. Build arrays
	I   = np.array(I)
	S   = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X   = np.array(X)
	Y   = np.array(Y)
	Z   = np.array(Z)
	R   = np.array(R)
	E   = np.array(E)
	P   = np.array(P)
	if Type == 'deepclass' or Type == 'DeepClass':
		# 7. One-Hot encoding and normalisation
		''' Y labels '''
		L[L=='Helix'] = 0
		L[L=='Sheet'] = 1
		Class = L.astype(np.int)
		''' X features '''
		categories = [sorted([x for x in range(1, 230+1)])]
		S = S.reshape(-1, 1)
		onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
		S = onehot_encoder.fit_transform(S)  #One-hot encode [Space Groups]
		UCe = (UCe-np.mean(UCe))/np.std(UCe) #Standardise    [Unit Cell Edges]
		UCa = (UCa-np.mean(UCa))/np.std(UCa) #Standardise    [Unit Cell Angle]
		X = (X-np.mean(X))/np.std(X)         #Standardise    [X Coordinates]
		Y = (Y-np.mean(Y))/np.std(Y)         #Standardise    [Y Coordinates]
		Z = (Z-np.mean(Z))/np.std(Z)         #Standardise    [Z Coordinates]
		R = (R-np.mean(R))/np.std(R)         #Standardise    [Resolution]
		E = (E-np.mean(E))/np.std(E)         #Standardise    [E-value]
		# 8. Construct tensors
		Space = S
		UnitC = np.concatenate([UCe, UCa], axis=1)
		Coord = np.array([X, Y, Z, R, E])
		Coord = np.swapaxes(Coord, 0, 2)
		Coord = np.swapaxes(Coord, 0, 1)
		S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
		# 9. Shuffle examples
		Coord, Class, UnitC, Space, I = shuffle(Coord, Class, UnitC, Space, I)
		print('IDs      =', I.shape)
		print('Space    =', Space.shape)
		print('UnitCell =', UnitC.shape)
		print('X Coord  =', Coord.shape)
		print('Y Class  =', Class.shape)
		return(Coord, Class, Space, UnitC, I)
	elif Type == 'deepphase' or Type == 'DeepPhase':
		# 7. One-Hot encoding and normalisation
		''' Y labels '''
		MIN, MAX, BIN = -4, 4, 8 # 8 bins for range -4 to 4
		bins = np.array([MIN+i*((MAX-MIN)/BIN) for i in range(BIN+1)][1:-1])
		P = np.digitize(P, bins)
		Phase = np.eye(BIN)[P] # One-hot encode the bins
		''' X features '''
		categories = [sorted([x for x in range(1, 230+1)])]
		S = S.reshape(-1, 1)
		onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
		S = onehot_encoder.fit_transform(S)  #One-hot encode [Space Groups]
		UCe = (UCe-np.mean(UCe))/np.std(UCe) #Standardise    [Unit Cell Edges]
		UCa = (UCa-np.mean(UCa))/np.std(UCa) #Standardise    [Unit Cell Angle]
		X = (X-np.mean(X))/np.std(X)         #Standardise    [X Coordinates]
		Y = (Y-np.mean(Y))/np.std(Y)         #Standardise    [Y Coordinates]
		Z = (Z-np.mean(Z))/np.std(Z)         #Standardise    [Z Coordinates]
		R = (R-np.mean(R))/np.std(R)         #Standardise    [Resolution]
		E = (E-np.mean(E))/np.std(E)         #Standardise    [E-value]
		# 8. Construct tensors
		Space = S
		UnitC = np.concatenate([UCe, UCa], axis=1)
		Coord = np.array([X, Y, Z, R, E])
		Coord = np.swapaxes(Coord, 0, 2)
		Coord = np.swapaxes(Coord, 0, 1)
		S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
		# 9. Shuffle examples
		Coord, Phase, UnitC, Space, I = shuffle(Coord, Phase, UnitC, Space, I)
		print('IDs      =', I.shape)
		print('Space    =', Space.shape)
		print('UnitCell =', UnitC.shape)
		print('X Coord  =', Coord.shape)
		print('Y Phase  =', Phase.shape)
		return(Coord, Phase, Space, UnitC, I)

def Voxel(filename='Gen.csv', show=False):
	''' Reducing the number of reflection points by voxelisation '''
	with open(filename) as f:
		next(f)
		for line in f:
			line = line.strip().split(',')
			I = line[0]
			L = line[1]
			S = line[2]
			UCe = ','.join(line[3:6])
			UCa = ','.join(line[6:9])
			T = line[9:]
			T = [float(i) for i in T]
			if       len(T)/6 <= 1e3: size=0.001; fn='1k-.csv'      ; vox=False
			if 1e3 < len(T)/6 <= 1e4: size=0.005; fn='1k-10k.csv'   ; vox=False
			if 1e4 < len(T)/6 <= 1e5: size=0.005; fn='10k-100k.csv' ; vox=False
			if 1e5 < len(T)/6 <= 2e5: size=0.008; fn='100k-200k.csv'; vox=True
			if 2e5 < len(T)/6 <= 3e5: size=0.008; fn='200k-300k.csv'; vox=True
			if 3e5 < len(T)/6 <= 4e5: size=0.008; fn='300k-400k.csv'; vox=True
			if 4e5 < len(T)/6 <= 5e5: size=0.008; fn='400k-500k.csv'; vox=True
			if 5e5 < len(T)/6 <= 1e6: size=0.008; fn='500k-1M.csv'  ; vox=True
			if 1e6 < len(T)/6:        size=0.010; fn='1M+.csv'      ; vox=True
			X = T[0::6]
			Y = T[1::6]
			Z = T[2::6]
			R = T[3::6]
			E = T[4::6]
			P = T[5::6]
			if vox == True:
				with open('example.xyz', 'w') as F:
					for x, y, z, r, e, p in zip(X, Y, Z, R, E, P):
						line = '{} {} {} {} {} {}\n'.format(x, y, z, r, e, p)
						F.write(line)
				xyz = o3d.io.read_point_cloud('example.xyz', 'xyzrgb')
				voxel_grid = o3d.geometry.VoxelGrid.\
				create_from_point_cloud(xyz, size)
				with open(fn, 'a') as F:
					start = '{},{},{},{},{}'\
					.format(I, L, S, UCe, UCa)
					F.write(start)
					for v in voxel_grid.get_voxels():
						x = v.grid_index[0]*size
						y = v.grid_index[1]*size
						z = v.grid_index[2]*size
						r = v.color[0]
						e = v.color[1]
						p = v.color[2]
						line = ',{},{},{},{},{},{}'.format(x, y, z, r, e, p)
						F.write(line)
					F.write('\n')
				os.remove('example.xyz')
				print('Point Cloud {:<10,}   Voxelised {:<10,}   Save {}'\
				.format(len(xyz.points), len(voxel_grid.get_voxels()), fn))
			elif vox == False:
				with open(fn, 'a') as F:
					start = '{},{},{},{},{}'.format(I, L, S, UCe, UCa)
					F.write(start)
					for x, y, z, r, e, p in zip(X, Y, Z, R, E, P):
						line = ',{},{},{},{},{},{}'.format(x, y, z, r, e, p)
						F.write(line)
					F.write('\n')
				print('Point Cloud {:<10,}   Not Voxelised       Save {}'\
				.format(len(X), fn))
			if show == True and vox == True:
				o3d.visualization.draw_geometries([xyz])
				o3d.visualization.draw_geometries([voxel_grid])
				X, Y, Z, R, E, P = [], [], [], [], [], []
				for v in voxel_grid.get_voxels():
					x = X.append(v.grid_index[0]*size)
					y = Y.append(v.grid_index[1]*size)
					z = Z.append(v.grid_index[2]*size)
					r = R.append(v.color[0])
					e = E.append(v.color[1])
					p = P.append(v.color[2])
				with open('Centers.xyz', 'w') as F:
					for x, y, z, r, e, p in zip(X, Y, Z, R, E, P):
						line = '{} {} {} {} {} {}\n'.format(x, y, z, r, e, p)
						F.write(line)
				xyz_centers = o3d.io.read_point_cloud('Centers.xyz', 'xyzrgb')
				o3d.visualization.draw_geometries([xyz_centers])
				os.remove('Centers.xyz')

def discover(filename):
	''' Discover dataset parameters '''
	X_mean, Y_mean, Z_mean, R_mean, E_mean = [], [], [], [], []
	X_len, Y_len, Z_len, R_len, E_len = [], [], [], [], []
	X_SD, Y_SD, Z_SD, R_SD, E_SD = [], [], [], [], []
	m = 0
	head = False
	with open(filename) as f:
		# 1. Check and skip header line
		if f.readline(6) == 'PDB_ID':
			head = True
			next(f)
		for line in f:
			# 2. Get data points
			line = line.strip().split(',')
			T = line[9:]
			X = [float(x) for x in T[0::6]]
			Y = [float(y) for y in T[1::6]]
			Z = [float(z) for z in T[2::6]]
			R = [float(r) for r in T[3::6]]
			E = [float(e) for e in T[4::6]]
			# 3. Find mean of each example
			X_mean.append(statistics.mean(X))
			Y_mean.append(statistics.mean(Y))
			Z_mean.append(statistics.mean(Z))
			R_mean.append(statistics.mean(R))
			E_mean.append(statistics.mean(E))
			# 4. Find standard deviation of each example
			X_SD.append(statistics.stdev(X))
			Y_SD.append(statistics.stdev(Y))
			Z_SD.append(statistics.stdev(Z))
			R_SD.append(statistics.stdev(R))
			E_SD.append(statistics.stdev(E))
			# 5. Find length of each example
			X_len.append(len(X))
			Y_len.append(len(Y))
			Z_len.append(len(Z))
			R_len.append(len(R))
			E_len.append(len(E))
			# 6. Count m number of examples
			m += 1
	# 7. Calculate fina dataset mean
	meanX = sum([X_mean[i]*X_len[i] for i in range(m)])/sum(X_len)
	meanY = sum([Y_mean[i]*Y_len[i] for i in range(m)])/sum(Y_len)
	meanZ = sum([Z_mean[i]*Z_len[i] for i in range(m)])/sum(Z_len)
	meanR = sum([R_mean[i]*R_len[i] for i in range(m)])/sum(R_len)
	meanE = sum([E_mean[i]*E_len[i] for i in range(m)])/sum(E_len)
	# 8. Calculate final dataset standard deviation
	listX = [X_SD[i]**2*(X_len[i]-1) + (X_mean[i]-meanX)*X_mean[i]*X_len[i]\
		for i in range(m)]
	listY = [Y_SD[i]**2*(Y_len[i]-1) + (Y_mean[i]-meanY)*Y_mean[i]*Y_len[i]\
		for i in range(m)]
	listZ = [Z_SD[i]**2*(Z_len[i]-1) + (Z_mean[i]-meanZ)*Z_mean[i]*Z_len[i]\
		for i in range(m)]
	listR = [R_SD[i]**2*(R_len[i]-1) + (R_mean[i]-meanR)*R_mean[i]*R_len[i]\
		for i in range(m)]
	listE = [E_SD[i]**2*(E_len[i]-1) + (E_mean[i]-meanE)*E_mean[i]*E_len[i]\
		for i in range(m)]
	sdevX = math.sqrt(sum(listX)/(sum(X_len) - 1))
	sdevY = math.sqrt(sum(listY)/(sum(Y_len) - 1))
	sdevZ = math.sqrt(sum(listZ)/(sum(Z_len) - 1))
	sdevR = math.sqrt(sum(listR)/(sum(R_len) - 1))
	sdevE = math.sqrt(sum(listE)/(sum(E_len) - 1))
	# 9. Generate indexes of lines (examples)
	index = [x for x in range(m)]
	# 10. If there is a dataset file with a header remove first line's index
	if head == True: index = index[1:]
	# 11. Shuffle line (example) indexes
	random.shuffle(index)
	# 12. Slice off train set
	train = index[:math.ceil((m*80)/100)]
	# 13. Slice off validation set
	valid = index[-1*math.floor((m*20)/100):]
	return(	m,
			meanX, meanY, meanZ, meanR, meanE,
			sdevX, sdevY, sdevZ, sdevR, sdevE,
			train, valid)

class DataGenerator(keras.utils.Sequence):
	def __init__(self, filename='CrystalDataset.csv', batch_size=8,
				Set='train', Type='Class', points=10, values=None):
		''' Initialization '''
		self.filename   = filename
		self.Set        = Set
		self.pts        = points
		self.m          = values[0]
		self.meanX      = values[1]
		self.meanY      = values[2]
		self.meanZ      = values[3]
		self.meanR      = values[4]
		self.meanE      = values[5]
		self.sdevX      = values[6]
		self.sdevY      = values[7]
		self.sdevZ      = values[8]
		self.sdevR      = values[9]
		self.sdevE      = values[10]
		self.train      = values[11]
		self.valid      = values[12]
		self.batch_size = batch_size
		self.on_epoch_end()
	def on_epoch_end(self):
		''' Shuffle at end of epoch '''
		if self.Set == 'train':
			self.example_indexes = np.arange(len(self.train))
			number_of_batches = len(self.example_indexes)/self.batch_size
			self.number_of_batches = int(np.floor(number_of_batches))
			np.random.shuffle(self.example_indexes)
			random.shuffle(self.train)
			self.X, self.Y, self.Space, self.UnitC, self.I = self.Vectorise(
				filename=self.filename, max_size=self.pts, Type='Class',
				index=self.train, m=self.m, meanX=self.meanX, meanY=self.meanY,
				meanZ=self.meanZ, meanR=self.meanR, meanE=self.meanE,
				sdevX=self.sdevX, sdevY=self.sdevY, sdevZ=self.sdevZ,
				sdevR=self.sdevR, sdevE=self.sdevE)
		elif self.Set == 'valid':
			self.example_indexes = np.arange(len(self.valid))
			number_of_batches = len(self.example_indexes)/self.batch_size
			self.number_of_batches = int(np.floor(number_of_batches))
			np.random.shuffle(self.example_indexes)
			random.shuffle(self.valid)
			self.X, self.Y, self.Space, self.UnitC, self.I = self.Vectorise(
				filename=self.filename, max_size=self.pts, Type='Class',
				index=self.valid, m=self.m, meanX=self.meanX, meanY=self.meanY,
				meanZ=self.meanZ, meanR=self.meanR, meanE=self.meanE,
				sdevX=self.sdevX, sdevY=self.sdevY, sdevZ=self.sdevZ,
				sdevR=self.sdevR, sdevE=self.sdevE)
		else: print("[+] Set type string incorrect, choose 'train' or 'valid'")
	def __len__(self):
		''' Denotes the number of batches per epoch '''
		return int(np.floor(len(self.X) / self.batch_size))
	def __getitem__(self, index):
		''' Generate one batch of data '''
		batch_indexes = self.example_indexes[index*self.batch_size:\
			(index+1)*self.batch_size]
		batch_x = np.array([self.X[k] for k in batch_indexes])
		batch_y = np.array([self.Y[k] for k in batch_indexes])
		return batch_x, batch_y
	def Vectorise(self, filename='CrystalDataset.csv', max_size='15000',
		Type='DeepClass', fp=np.float32, ip=np.int32, index=[1, 2, 3], m=None,
		meanX=None, meanY=None, meanZ=None, meanR=None, meanE=None,
		sdevX=None, sdevY=None, sdevZ=None, sdevR=None, sdevE=None):
		''' This function randomly samples points from each example '''
		I = np.array([])
		L = np.array([])
		S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
		max_size = int(max_size)
		with open(filename, 'r') as f:
			for pos, line in enumerate(f):
				if pos in index:
					line = line.strip().split(',')
					# 1. Isolate PDB IDs and labels
					I = np.append(I, np.array(str(line[0]), dtype=str))
					L = np.append(L, np.array(str(line[1]), dtype=str))
					S.append(np.array(int(line[2]), dtype=ip))
					UCe.append(np.array([float(i) for i in line[3:6]],dtype=fp))
					UCa.append(np.array([float(i) for i in line[6:9]],dtype=fp))
					# 2. Isolate points
					T = line[9:]
					T = [float(i) for i in T]
					# 3. Collect each point values
					NC = [(x, y, z, r, e, p) for x, y, z, r, e, p
						in zip(T[0::6],T[1::6],T[2::6],T[3::6],T[4::6],T[5::6])]
					# 4. Random sampling of points
					T = [random.choice(NC) for x in range(max_size)]
					assert len(T) == max_size, 'Max number of points incorrect'
					T = [i for sub in T for i in sub]
					# 5. Export points
					X.append(np.array(T[0::6], dtype=fp))
					Y.append(np.array(T[1::6], dtype=fp))
					Z.append(np.array(T[2::6], dtype=fp))
					R.append(np.array(T[3::6], dtype=fp))
					E.append(np.array(T[4::6], dtype=fp))
					P.append(np.array(T[5::6], dtype=fp))
		# 6. Build arrays
		I   = np.array(I)
		S   = np.array(S)
		UCe = np.array(UCe)
		UCa = np.array(UCa)
		X   = np.array(X)
		Y   = np.array(Y)
		Z   = np.array(Z)
		R   = np.array(R)
		E   = np.array(E)
		P   = np.array(P)
		if Type == 'class' or Type == 'Class':
			# 7. One-Hot encoding and normalisation
			''' Y labels '''
			L[L=='Helix'] = 0
			L[L=='Sheet'] = 1
			Class = L.astype(np.int)
			''' X features '''
			categories = [sorted([x for x in range(1, 230+1)])]
			S = S.reshape(-1, 1)
			onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
			S = onehot_encoder.fit_transform(S) #One-hot encode[Space Groups]
			UCe = (UCe-np.mean(UCe))/np.std(UCe)#Standardise   [Unit Cell Edges]
			UCa = (UCa-np.mean(UCa))/np.std(UCa)#Standardise   [Unit Cell Angle]
			X = (X-meanX)/sdevX                 #Standardise    [X Coordinates]
			Y = (Y-meanY)/sdevY                 #Standardise    [Y Coordinates]
			Z = (Z-meanZ)/sdevZ                 #Standardise    [Z Coordinates]
			R = (R-meanR)/sdevR                 #Standardise    [Resolution]
			E = (E-meanE)/sdevE                 #Standardise    [E-value]
			# 8. Construct tensors
			Space = S
			UnitC = np.concatenate([UCe, UCa], axis=1)
			Coord = np.array([X, Y, Z, R, E])
			Coord = np.swapaxes(Coord, 0, 2)
			Coord = np.swapaxes(Coord, 0, 1)
			S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
			# 9. Shuffle examples
			Coord, Class, UnitC, Space, I = shuffle(Coord,Class,UnitC,Space,I)
			return(Coord, Class, Space, UnitC, I)
		elif Type == 'phase' or Type == 'Phase':
			# 7. One-Hot encoding and normalisation
			''' Y labels '''
			MIN, MAX, BIN = -4, 4, 8 # 8 bins for range -4 to 4
			bins = np.array([MIN+i*((MAX-MIN)/BIN) for i in range(BIN+1)][1:-1])
			P = np.digitize(P, bins)
			Phase = np.eye(BIN)[P] # One-hot encode the bins
			''' X features '''
			categories = [sorted([x for x in range(1, 230+1)])]
			S = S.reshape(-1, 1)
			onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
			S = onehot_encoder.fit_transform(S) #One-hot encode[Space Groups]
			UCe = (UCe-np.mean(UCe))/np.std(UCe)#Standardise   [Unit Cell Edges]
			UCa = (UCa-np.mean(UCa))/np.std(UCa)#Standardise   [Unit Cell Angle]
			X = (X-meanX)/sdevX                 #Standardise   [X Coordinates]
			Y = (Y-meanY)/sdevY                 #Standardise   [Y Coordinates]
			Z = (Z-meanZ)/sdevZ                 #Standardise   [Z Coordinates]
			R = (R-meanR)/sdevR                 #Standardise   [Resolution]
			E = (E-meanE)/sdevE                 #Standardise   [E-value]
			# 8. Construct tensors
			Space = S
			UnitC = np.concatenate([UCe, UCa], axis=1)
			Coord = np.array([X, Y, Z, R, E])
			Coord = np.swapaxes(Coord, 0, 2)
			Coord = np.swapaxes(Coord, 0, 1)
			S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
			# 9. Shuffle examples
			Coord, Phase, UnitC, Space, I = shuffle(Coord,Phase,UnitC,Space,I)
			return(Coord, Phase, Space, UnitC, I)

def Vectorise(filename='CrystalDataset.csv', max_size='1600', data='DeepClass'):
	''' Selects points, splits sets, standerdises, then vectorises dataset '''
	I = np.array([])
	L = np.array([])
	S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
	max_size = int(max_size)
	with open(filename, 'r') as f:
		next(f)
		for line in f:
			line = line.strip().split(',')
			# 1. Isolate points
			T = line[9:]
			T = [float(i) for i in T]
			# 2. Collect each point values
			NC = [(x, y, z, r, e, p) for x, y, z, r, e, p
				in zip(T[0::6], T[1::6], T[2::6], T[3::6], T[4::6], T[5::6])]
			# 3. Select only points where 2.8 < R < 3
			NC = [point for point in NC if 2.8 <= point[3] <= 3.0]
			# 4. Sample points at regular intervals to collect max_size
			if len(NC) != 0 and len(NC) >= max_size:
				T = NC[::len(NC)//max_size][:max_size]
				# 5. Isolate PDB IDs and labels
				I = np.append(I, np.array(str(line[0]), dtype=str))
				L = np.append(L, np.array(str(line[1]), dtype=str))
				S.append(np.array(int(line[2]), dtype=np.int32))
				UCe.append(np.array([float(i) for i in line[3:6]]))
				UCa.append(np.array([float(i) for i in line[6:9]]))
			else: continue
			assert len(T) == max_size, 'Max number of points incorrect'
			T = [i for sub in T for i in sub]
			# 6. Export points
			X.append(np.array(T[0::6]))
			Y.append(np.array(T[1::6]))
			Z.append(np.array(T[2::6]))
			R.append(np.array(T[3::6]))
			E.append(np.array(T[4::6]))
			P.append(np.array(T[5::6]))
	# 7. Structure encoding
	''' DeepClass Y labels '''
	L[L=='Helix'] = 0
	L[L=='Sheet'] = 1
	L = L.astype(np.int)
	# 8. Build arrays
	I   = np.array(I)
	S   = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X   = np.array(X)
	Y   = np.array(Y)
	Z   = np.array(Z)
	R   = np.array(R)
	E   = np.array(E)
	P   = np.array(P)
	# 9. Shuffle but maintain order
	L, I, S, UCe, UCa, X, Y, Z, R, E, P = \
	shuffle(L, I, S, UCe, UCa, X, Y, Z, R, E, P)
	# 10. Split train/valid/tests sets
	train_to   = math.floor((len(X)*60)/100)
	valid_from = train_to
	valid_to   = train_to + math.floor((len(X)*20)/100)
	tests_from = valid_to
	L_train = L[:train_to]
	L_valid = L[valid_from:valid_to]
	L_tests = L[tests_from:]
	I_train = I[:train_to]
	I_valid = I[valid_from:valid_to]
	I_tests = I[tests_from:]
	S_train = S[:train_to]
	S_valid = S[valid_from:valid_to]
	S_tests = S[tests_from:]
	UCe_train = UCe[:train_to]
	UCe_valid = UCe[valid_from:valid_to]
	UCe_tests = UCe[tests_from:]
	UCa_train = UCa[:train_to]
	UCa_valid = UCa[valid_from:valid_to]
	UCa_tests = UCa[tests_from:]
	X_train = X[:train_to]
	X_valid = X[valid_from:valid_to]
	X_tests = X[tests_from:]
	Y_train = Y[:train_to]
	Y_valid = Y[valid_from:valid_to]
	Y_tests = Y[tests_from:]
	Z_train = Z[:train_to]
	Z_valid = Z[valid_from:valid_to]
	Z_tests = Z[tests_from:]
	R_train = R[:train_to]
	R_valid = R[valid_from:valid_to]
	R_tests = R[tests_from:]
	E_train = E[:train_to]
	E_valid = E[valid_from:valid_to]
	E_tests = E[tests_from:]
	P_train = P[:train_to]
	P_valid = P[valid_from:valid_to]
	P_tests = P[tests_from:]
	S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
	# 11. Label and feature one-hot encoding standardisation
	''' X features '''
	categories = [sorted([x for x in range(1, 230+1)])]
	onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
	S_train = S_train.reshape(-1, 1)
	S_valid = S_valid.reshape(-1, 1)
	S_tests = S_tests.reshape(-1, 1)
	S_train = onehot_encoder.fit_transform(S_train)
	S_valid = onehot_encoder.fit_transform(S_valid)
	S_tests = onehot_encoder.fit_transform(S_tests)
	UCe_train = (UCe_train-np.mean(UCe_train))/np.std(UCe_train)
	UCe_valid = (UCe_valid-np.mean(UCe_valid))/np.std(UCe_valid)
	UCe_tests = (UCe_tests-np.mean(UCe_tests))/np.std(UCe_tests)
	UCa_train = (UCa_train-np.mean(UCa_train))/np.std(UCa_train)
	UCa_valid = (UCa_valid-np.mean(UCa_valid))/np.std(UCa_valid)
	UCa_tests = (UCa_tests-np.mean(UCa_tests))/np.std(UCa_tests)
	X_train = (X_train-np.mean(X_train))/np.std(X_train)
	X_valid = (X_valid-np.mean(X_valid))/np.std(X_valid)
	X_tests = (X_tests-np.mean(X_tests))/np.std(X_tests)
	Y_train = (Y_train-np.mean(Y_train))/np.std(Y_train)
	Y_valid = (Y_valid-np.mean(Y_valid))/np.std(Y_valid)
	Y_tests = (Y_tests-np.mean(Y_tests))/np.std(Y_tests)
	Z_train = (Z_train-np.mean(Z_train))/np.std(Z_train)
	Z_valid = (Z_valid-np.mean(Z_valid))/np.std(Z_valid)
	Z_tests = (Z_tests-np.mean(Z_tests))/np.std(Z_tests)
	R_train = (R_train-np.mean(R_train))/np.std(R_train)
	R_valid = (R_valid-np.mean(R_valid))/np.std(R_valid)
	R_tests = (R_tests-np.mean(R_tests))/np.std(R_tests)
	E_train = (E_train-np.mean(E_train))/np.std(E_train)
	E_valid = (E_valid-np.mean(E_valid))/np.std(E_valid)
	E_tests = (E_tests-np.mean(E_tests))/np.std(E_tests)
	''' DeepPhase Y labels '''
	MIN, MAX, BIN = -4, 4, 8 # 8 bins for range -4 to 4
	bins = np.array([MIN+i*((MAX-MIN)/BIN) for i in range(BIN+1)][1:-1])
	P_train = np.digitize(P_train, bins)
	P_valid = np.digitize(P_valid, bins)
	P_tests = np.digitize(P_tests, bins)
	P_train = np.eye(BIN)[P_train]
	P_valid = np.eye(BIN)[P_valid]
	P_tests = np.eye(BIN)[P_tests]
	# 12. Construct tensors
	Ident_train = I_train
	Ident_valid = I_valid
	Ident_tests = I_tests
	Class_train = L_train
	Class_valid = L_valid
	Class_tests = L_tests
	Space_train = S_train
	Space_valid = S_valid
	Space_tests = S_tests
	UnitC_train = np.concatenate([UCe_train, UCa_train], axis=1)
	UnitC_valid = np.concatenate([UCe_valid, UCa_valid], axis=1)
	UnitC_tests = np.concatenate([UCe_tests, UCa_tests], axis=1)
	Coord_train = np.array([X_train, Y_train, Z_train, R_train, E_train])
	Coord_train = np.swapaxes(Coord_train, 0, 2)
	Coord_train = np.swapaxes(Coord_train, 0, 1)
	Coord_valid = np.array([X_valid, Y_valid, Z_valid, R_valid, E_valid])
	Coord_valid = np.swapaxes(Coord_valid, 0, 2)
	Coord_valid = np.swapaxes(Coord_valid, 0, 1)
	Coord_tests = np.array([X_tests, Y_tests, Z_tests, R_tests, E_tests])
	Coord_tests = np.swapaxes(Coord_tests, 0, 2)
	Coord_tests = np.swapaxes(Coord_tests, 0, 1)
	Phase_train = P_train
	Phase_valid = P_valid
	Phase_tests = P_tests
	if data == 'DeepClass':
		return( Coord_train, Coord_valid, Coord_tests,
				Class_train, Class_valid, Class_tests)
	elif data == 'DeepPhase':
		return( Coord_train, Coord_valid, Coord_tests,
				Phase_train, Phase_valid, Phase_tests)










''' LATEST EXPERIMENT '''


def Vectorise_last(	filename='CrystalDataset.csv',
				data='DeepClass',
				point_size=None,
				batch_size=None):
	'''
	Vectorises the dataset by spliting it into train/valid/tests sets, filter
	points between 2.8 < R < 3.0, standerdise each set seperatly, compiles them
	into tendors, then export them
	'''
	I = np.array([])
	L = np.array([])
	S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
	with open(filename, 'r') as f:
		next(f)
		for line in f:
			line = line.strip().split(',')
			# 1. Isolate points
			T = line[9:]
			T = [float(i) for i in T]
			# 2. Collect each point values
			NC = [(x, y, z, r, e, p) for x, y, z, r, e, p
				in zip(T[0::6], T[1::6], T[2::6], T[3::6], T[4::6], T[5::6])]
			# 3. Select only points where 2.8 < R < 3
			NC = [point for point in NC if 2.8 <= point[3] <= 3.0]
			# 4. Collect examples if they have points within R range
			if len(NC) != 0:
				T = NC
				# 5. Isolate PDB IDs and labels
				I = np.append(I, np.array(str(line[0]), dtype=str))
				L = np.append(L, np.array(str(line[1]), dtype=str))
				S.append(np.array(int(line[2]), dtype=np.int32))
				UCe.append(np.array([float(i) for i in line[3:6]]))
				UCa.append(np.array([float(i) for i in line[6:9]]))
			else: continue
			T = [i for sub in T for i in sub]
			# 6. Export points
			X.append(T[0::6])
			Y.append(T[1::6])
			Z.append(T[2::6])
			R.append(T[3::6])
			E.append(T[4::6])
			P.append(T[5::6])
	# 7. Structure encoding
	''' DeepClass Y labels '''
	L[L=='Helix'] = 0
	L[L=='Sheet'] = 1
	L = L.astype(np.int)
# 8. Build arrays
	I   = np.array(I)
	S   = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X   = pd.DataFrame(X)
	X   = pd.DataFrame.to_numpy(X)
	Y   = pd.DataFrame(Y)
	Y   = pd.DataFrame.to_numpy(Y)
	Z   = pd.DataFrame(Z)
	Z   = pd.DataFrame.to_numpy(Z)
	R   = pd.DataFrame(R)
	R   = pd.DataFrame.to_numpy(R)
	E   = pd.DataFrame(E)
	E   = pd.DataFrame.to_numpy(E)
	P   = pd.DataFrame(P)
	P   = pd.DataFrame.to_numpy(P)
	# 9. Shuffle but maintain order
	L, I, S, UCe, UCa, X, Y, Z, R, E, P = \
	shuffle(L, I, S, UCe, UCa, X, Y, Z, R, E, P)
	# 10. Split train/valid/tests sets
	train_to   = math.floor((len(X)*60)/100)
	valid_from = train_to
	valid_to   = train_to + math.floor((len(X)*20)/100)
	tests_from = valid_to
	L_train = L[:train_to]
	L_valid = L[valid_from:valid_to]
	L_tests = L[tests_from:]
	I_train = I[:train_to]
	I_valid = I[valid_from:valid_to]
	I_tests = I[tests_from:]
	S_train = S[:train_to]
	S_valid = S[valid_from:valid_to]
	S_tests = S[tests_from:]
	UCe_train = UCe[:train_to]
	UCe_valid = UCe[valid_from:valid_to]
	UCe_tests = UCe[tests_from:]
	UCa_train = UCa[:train_to]
	UCa_valid = UCa[valid_from:valid_to]
	UCa_tests = UCa[tests_from:]
	X_train = X[:train_to]
	X_valid = X[valid_from:valid_to]
	X_tests = X[tests_from:]
	Y_train = Y[:train_to]
	Y_valid = Y[valid_from:valid_to]
	Y_tests = Y[tests_from:]
	Z_train = Z[:train_to]
	Z_valid = Z[valid_from:valid_to]
	Z_tests = Z[tests_from:]
	R_train = R[:train_to]
	R_valid = R[valid_from:valid_to]
	R_tests = R[tests_from:]
	E_train = E[:train_to]
	E_valid = E[valid_from:valid_to]
	E_tests = E[tests_from:]
	P_train = P[:train_to]
	P_valid = P[valid_from:valid_to]
	P_tests = P[tests_from:]
	S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
	# 11. Label and feature one-hot encoding standardisation
	''' X features '''
	categories = [sorted([x for x in range(1, 230+1)])]
	onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
	S_train = S_train.reshape(-1, 1)
	S_valid = S_valid.reshape(-1, 1)
	S_tests = S_tests.reshape(-1, 1)
	S_train = onehot_encoder.fit_transform(S_train)
	S_valid = onehot_encoder.fit_transform(S_valid)
	S_tests = onehot_encoder.fit_transform(S_tests)
	UCe_train = (UCe_train-np.mean(UCe_train))/np.std(UCe_train)
	UCe_valid = (UCe_valid-np.mean(UCe_valid))/np.std(UCe_valid)
	UCe_tests = (UCe_tests-np.mean(UCe_tests))/np.std(UCe_tests)
	UCa_train = (UCa_train-np.mean(UCa_train))/np.std(UCa_train)
	UCa_valid = (UCa_valid-np.mean(UCa_valid))/np.std(UCa_valid)
	UCa_tests = (UCa_tests-np.mean(UCa_tests))/np.std(UCa_tests)
	X_train = (X_train-np.nanmean(X_train))/np.nanstd(X_train)
	X_valid = (X_valid-np.nanmean(X_valid))/np.nanstd(X_valid)
	X_tests = (X_tests-np.nanmean(X_tests))/np.nanstd(X_tests)
	Y_train = (Y_train-np.nanmean(Y_train))/np.nanstd(Y_train)
	Y_valid = (Y_valid-np.nanmean(Y_valid))/np.nanstd(Y_valid)
	Y_tests = (Y_tests-np.nanmean(Y_tests))/np.nanstd(Y_tests)
	Z_train = (Z_train-np.nanmean(Z_train))/np.nanstd(Z_train)
	Z_valid = (Z_valid-np.nanmean(Z_valid))/np.nanstd(Z_valid)
	Z_tests = (Z_tests-np.nanmean(Z_tests))/np.nanstd(Z_tests)
	R_train = (R_train-np.nanmean(R_train))/np.nanstd(R_train)
	R_valid = (R_valid-np.nanmean(R_valid))/np.nanstd(R_valid)
	R_tests = (R_tests-np.nanmean(R_tests))/np.nanstd(R_tests)
	E_train = (E_train-np.nanmean(E_train))/np.nanstd(E_train)
	E_valid = (E_valid-np.nanmean(E_valid))/np.nanstd(E_valid)
	E_tests = (E_tests-np.nanmean(E_tests))/np.nanstd(E_tests)
	''' DeepPhase Y labels '''
	MIN, MAX, BIN = -4, 4, 8
	bins = np.array([MIN+i*((MAX-MIN)/BIN) for i in range(BIN+1)][1:-1])
	P_train = pd.DataFrame(P_train)
	P_valid = pd.DataFrame(P_valid)
	P_tests = pd.DataFrame(P_tests)
	P_train = pd.DataFrame.to_numpy(P_train)
	P_valid = pd.DataFrame.to_numpy(P_valid)
	P_tests = pd.DataFrame.to_numpy(P_tests)
	P_train = np.digitize(P_train, bins)
	P_valid = np.digitize(P_valid, bins)
	P_tests = np.digitize(P_tests, bins)
	P_train = np.eye(BIN)[P_train]
	P_valid = np.eye(BIN)[P_valid]
	P_tests = np.eye(BIN)[P_tests]
	# 12. Construct tensors
	Ident_train = I_train
	Ident_valid = I_valid
	Ident_tests = I_tests
	Class_train = L_train
	Class_valid = L_valid
	Class_tests = L_tests
	Space_train = S_train
	Space_valid = S_valid
	Space_tests = S_tests
	UnitC_train = np.concatenate([UCe_train, UCa_train], axis=1)
	UnitC_valid = np.concatenate([UCe_valid, UCa_valid], axis=1)
	UnitC_tests = np.concatenate([UCe_tests, UCa_tests], axis=1)
	Coord_train = np.array([X_train, Y_train, Z_train, R_train, E_train])
	Coord_train = np.swapaxes(Coord_train, 0, 2)
	Coord_train = np.swapaxes(Coord_train, 0, 1)
	Coord_valid = np.array([X_valid, Y_valid, Z_valid, R_valid, E_valid])
	Coord_valid = np.swapaxes(Coord_valid, 0, 2)
	Coord_valid = np.swapaxes(Coord_valid, 0, 1)
	Coord_tests = np.array([X_tests, Y_tests, Z_tests, R_tests, E_tests])
	Coord_tests = np.swapaxes(Coord_tests, 0, 2)
	Coord_tests = np.swapaxes(Coord_tests, 0, 1)
	Phase_train = P_train
	Phase_valid = P_valid
	Phase_tests = P_tests
	# 13. Export dataset
	if data == 'DeepClass' and point_size == None and batch_size == None:
		return( Coord_train, Coord_valid, Coord_tests,
				Class_train, Class_valid, Class_tests)
	elif data == 'DeepClass' and isinstance(point_size, int)\
		and batch_size == None:
		select = point_size
		P = []
		for p in Coord_train:
			p = p[~np.isnan(p).any(axis=1)]
			if select <= len(p):
				p = p[::len(p)//select][:select]
				p = np.ndarray.tolist(p)
				P.append(p)
			else: continue
		Coord_train = np.array(P)
		P = []
		for p in Coord_valid:
			p = p[~np.isnan(p).any(axis=1)]
			if select <= len(p):
				p = p[::len(p)//select][:select]
				p = np.ndarray.tolist(p)
				P.append(p)
			else: continue
		Coord_valid = np.array(P)
		P = []
		for p in Coord_tests:
			p = p[~np.isnan(p).any(axis=1)]
			if select <= len(p):
				p = p[::len(p)//select][:select]
				p = np.ndarray.tolist(p)
				P.append(p)
			else: continue
		Coord_tests = np.array(P)
		return( Coord_train, Coord_valid, Coord_tests,
				Class_train, Class_valid, Class_tests)
	elif data == 'DeepClass' and isinstance(point_size, int)\
		and isinstance(batch_size, int):
		select = point_size
		P = []
		for p in Coord_train:
			p = p[~np.isnan(p).any(axis=1)]
			if select <= len(p):
				p = [random.choice(p) for x in range(select)]
				P.append(p)
			else: continue
		Coord_train = np.array(P)
		P = []
		for p in Coord_valid:
			p = p[~np.isnan(p).any(axis=1)]
			if select <= len(p):
				p = [random.choice(p) for x in range(select)]
				P.append(p)
			else: continue
		Coord_valid = np.array(P)
		P = []
		for p in Coord_tests:
			p = p[~np.isnan(p).any(axis=1)]
			if select <= len(p):
				p = [random.choice(p) for x in range(select)]
				P.append(p)
			else: continue
		Coord_tests = np.array(P)
		segment = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
		train_example_indexes = list(np.arange(len(Coord_train)))
		train_batch_indexes = segment(train_example_indexes, batch_size)
		valid_example_indexes = list(np.arange(len(Coord_valid)))
		valid_batch_indexes = segment(valid_example_indexes, batch_size)
		tests_example_indexes = list(np.arange(len(Coord_tests)))
		tests_batch_indexes = segment(tests_example_indexes, batch_size)
		x_batches_r = []
		y_batches_r = []
		for i in train_batch_indexes:
			x = np.array([Coord_train[k] for k in i])
			y = np.array([Class_train[k] for k in i])
			x_batches_r.append(x)
			y_batches_r.append(y)
		x_batches_v = []
		y_batches_v = []
		for i in valid_batch_indexes:
			x = np.array([Coord_valid[k] for k in i])
			y = np.array([Class_valid[k] for k in i])
			x_batches_v.append(x)
			y_batches_v.append(y)
		x_batches_t = []
		y_batches_t = []
		for i in tests_batch_indexes:
			x = np.array([Coord_tests[k] for k in i])
			y = np.array([Class_tests[k] for k in i])
			x_batches_t.append(x)
			y_batches_t.append(y)
		return( x_batches_r, y_batches_r,
				x_batches_v, y_batches_v,
				x_batches_t, y_batches_t)
	elif data == 'DeepPhase'and point_size == None and batch_size == None:
		return( Coord_train, Coord_valid, Coord_tests,
				Phase_train, Phase_valid, Phase_tests)
	#=======================================================
	#=======================================================
	#=======================================================
	elif data == 'DeepPhase' and isinstance(point_size, int)\
		and batch_size == None:
		print('NOT YET CODED')
	elif data == 'DeepPhase' and isinstance(point_size, int)\
		and isinstance(batch_size, list):
		print('NOT YET CODED')

Coord_train, Coord_valid, Coord_tests, Class_train, Class_valid, Class_tests = Vectorise()
with h5py.File('X_train.h5','w') as x_train:dset=x_train.create_dataset('default', data=Coord_train)
with h5py.File('X_valid.h5','w') as x_valid:dset=x_valid.create_dataset('default', data=Coord_valid)
with h5py.File('X_tests.h5','w') as x_tests:dset=x_tests.create_dataset('default', data=Coord_tests)
with h5py.File('Y_train.h5','w') as y_train:dset=y_train.create_dataset('default', data=Class_train)
with h5py.File('Y_valid.h5','w') as y_valid:dset=y_valid.create_dataset('default', data=Class_valid)
with h5py.File('Y_tests.h5','w') as y_tests:dset=y_tests.create_dataset('default', data=Class_tests)



class DataGenerator(keras.utils.Sequence):
	''' DataGenerator for Vectorise_last() '''
	def __init__(self, X, Y, batch_size, feature_size):
		''' Initialization '''
		self.X = X
		self.Y = Y
		self.feature_size = feature_size
		self.batch_size = batch_size
		self.on_epoch_end()
	def on_epoch_end(self):
		''' Shuffle at end of epoch '''
		self.example_indexes = np.arange(len(self.X))
		number_of_batches = len(self.example_indexes)/self.batch_size
		self.number_of_batches = int(np.floor(number_of_batches))
		np.random.shuffle(self.example_indexes)
	def __len__(self):
		''' Denotes the number of batches per epoch '''
		return(int(np.floor(len(self.X)/self.batch_size)))
	def __getitem__(self, index):
		''' Generate one batch of data '''
		batch_indexes = self.example_indexes[index*self.batch_size:\
			(index+1)*self.batch_size]
		batch_x = np.array([self.X[k] for k in batch_indexes])
		batch_y = np.array([self.Y[k] for k in batch_indexes])
		x = []
		for example in batch_x:
			example = example[~np.isnan(example).any(axis=1)]
			idx = np.random.choice(len(example),\
				size=self.feature_size, replace=False)
			example = example[idx, :]
			x.append(example)
		batch_x = np.array(x)
		return batch_x, batch_y

''' LATEST EXPERIMENT '''

def main():
	if  args.Setup:
		setup()
	elif  args.Dataset:
		D = Dataset()
		D.run(IDs=sys.argv[2])
	elif args.Vectorise:
		Type = sys.argv[2]
		File = sys.argv[3]
		size = int(sys.argv[4])
		X, Y, S, U, I = Vectorise(filename=File, max_size=size, Type=Type)
	elif args.Serialise:
		Type = sys.argv[2]
		File = sys.argv[3]
		size = int(sys.argv[4])
		X, Y, S, U, I = Vectorise(filename=File, max_size=size, Type=Type)
		I = [n.encode('ascii', 'ignore') for n in I]
		with h5py.File('X.h5','w') as x:dset=x.create_dataset('default',data=X)
		with h5py.File('Y.h5','w') as y:dset=y.create_dataset('default',data=Y)
	elif args.Augment:
		PDB_MTZ = 'PDB'
		d = 2.5
		n = int(sys.argv[2])
		augment = True
		D = Dataset(PDB_MTZ=PDB_MTZ, n=n, d=d, augment=augment)
		try:
			sys.argv[3]=='MTZ'
			D.run(EXP_MTZ=True)
			os.remove('CrystalDataset.csv')
		except:
			D.run()
	elif args.Voxelise:
		Voxel(filename=sys.argv[2], size=sys.argv[3])
	elif args.Generator:
		FN = sys.argv[2]
		Type = sys.argv[3]
		pts = sys.argv[4]
		values = discover(FN)
		with Pool(2) as p:
			train, valid = p.starmap(DataGenerator,\
			[(FN, 32, 'train', Type, pts, values),\
			(FN, 32, 'valid', Type, pts, values)])

if __name__ == '__main__': main()
