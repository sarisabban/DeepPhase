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
import subprocess
import numpy as np
import open3d as o3d
import urllib.request
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
	def Ref_PDB(self, pdbstr):
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
	def run(self, IDs='IDs.txt'):
		with open('CrystalDataset.csv', 'a') as F:
			h1 = 'PDB_ID,Class,Space_Group,'
			h2 = 'Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,'
			h3 = 'Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma'
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
								try: S,C,X,Y,Z,R,E,P = self.Ref_PDB(pdbstr)
								except: continue
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
						else:
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

class DataGenerator(keras.utils.Sequence):
	''' Data generator for the dataset '''
	def __init__(self, filename='CrystalDataset.csv', batch_size=8,
				Set='train', Type='Class', points=10):
		''' Initialization '''
		values = self.discover(filename)
		m     = values[0]
		meanX = values[1]
		meanY = values[2]
		meanZ = values[3]
		meanR = values[4]
		meanE = values[5]
		sdevX = values[6]
		sdevY = values[7]
		sdevZ = values[8]
		sdevR = values[9]
		sdevE = values[10]
		train = values[11]
		valid = values[12]
		pts   = points
		if Set == 'train':
			self.X, self.Y, self.Space, self.UnitC, self.I = self.Vectorise(
				filename=filename, max_size=pts, Type='Class', index=train, m=m,
				meanX=meanX, meanY=meanY, meanZ=meanZ, meanR=meanR, meanE=meanE,
				sdevX=sdevX, sdevY=sdevY, sdevZ=sdevZ, sdevR=sdevR, sdevE=sdevE)
		elif Set == 'valid':
			self.X, self.Y, self.Space, self.UnitC, self.I = self.Vectorise(
				filename=filename, max_size=pts, Type='Class', index=valid, m=m,
				meanX=meanX, meanY=meanY, meanZ=meanZ, meanR=meanR, meanE=meanE,
				sdevX=sdevX, sdevY=sdevY, sdevZ=sdevZ, sdevR=sdevR, sdevE=sdevE)
		else: print("[+] Set type string incorrect, choose 'train' or 'valid'")
		self.batch_size = batch_size
		self.example_indexes = np.arange(len(self.X))
		number_of_batches = len(self.example_indexes)/self.batch_size
		self.number_of_batches = int(np.floor(number_of_batches))
		self.on_epoch_end()
	def on_epoch_end(self):
		''' Shuffle at end of epoch '''
		np.random.shuffle(self.example_indexes)
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
	def discover(self, filename):
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
				# 6. Count number of examples m
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
		# 9. Generate indexes of lines (exmaples)
		index = [x for x in range(m)]
		# 10. If there is a dataset file has header remove first line index
		if head == True: index = index[1:]
		# 11. shuffle line (example) indexes
		random.shuffle(index)
		# 12. Slice off train set
		train = index[:math.ceil((m*80)/100)]
		# 13. Slice off validation set
		valid = index[-1*math.floor((m*20)/100):]
		return(	m, meanX, meanY, meanZ, meanR, meanE,
				sdevX, sdevY, sdevZ, sdevR, sdevE,
				train, valid)
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
		D.run()
	elif args.Voxelise:
		Voxel(filename=sys.argv[2], size=sys.argv[3])
	elif args.Generator:
		FN = sys.argv[2]
		Type = sys.argv[3]
		pts = sys.argv[4]
		train = DataGenerator(filename=FN,
							batch_size=32, Set='train', Type=Type, points=pts)
		valid = DataGenerator(filename=FN,
							batch_size=32, Set='valid', Type=Type, points=pts)

if __name__ == '__main__': main()
