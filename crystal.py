import os
import sys
import h5py
import tqdm
import random
import urllib
import Bio.PDB
import argparse
import numpy as np
from iotbx.reflection_file_reader import any_reflection_file
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

parser = argparse.ArgumentParser(description='Compiling X-ray diffraction datasets')
parser.add_argument('-C' , '--Class'   , nargs='+', help='Compile DeepClass protein classification dataset, include filename of PDB IDs')
parser.add_argument('-P' , '--Phase'   , nargs='+', help='Compile DeepPhase protein phase calculation dataset, include filename of PDB IDs')
parser.add_argument('-vC', '--VecClass', nargs='+', help='Vectorise and serialise the DeepClass protein classification dataset, include maximum number of reflections of leave empty for all reflections')
parser.add_argument('-vP', '--VecPhase', nargs='+', help='Vectorise and serialise the DeepPhase protein phase calculation dataset, include maximum number of reflections of leave empty for all reflections')
args = parser.parse_args()

class ClassData():
	'''
	Build a dataset for protein classification (Helix/Sheet)
	from x-ray diffraction .mtz and .pbd files and compile all
	the reflections
	'''
	def download(self, ID):
		''' Downloads a structure's .mtz and .pdb files '''
		Murl = 'http://edmaps.rcsb.org/coefficients/{}.mtz'.format(ID)
		Purl = 'https://files.rcsb.org/download/{}.pdb'.format(ID)
		urllib.request.urlretrieve(Murl, '{}.mtz'.format(ID))
		urllib.request.urlretrieve(Purl, '{}.pdb'.format(ID))
	def features(self, filename):
		'''
		Extracts required features from .mtz files. More info can be found here:
		https://cci.lbl.gov/cctbx_docs/cctbx/cctbx.miller.html
		'''
		hkl_file = any_reflection_file(filename)
		arrays = hkl_file.as_miller_arrays(merge_equivalents=False)
		for a in arrays:
			label = str(a.info()).split(':')[-1]
			if label == 'FOM':
				# Cell Dimentions
				C = a.unit_cell()
				# P1 expand
				P1 = a.expand_to_p1().indices()
				# Resolution
				R = list(C.d(P1))
				# Space Group
				ms_base = a.customized_copy()
				ms_all = ms_base.complete_set()
				ms=ms_all.customized_copy(space_group_info=a.space_group_info())
				S = str(ms).split()[-1].split(')')[0]
				# E-values in P1 space group
				a.setup_binner(auto_binning=True)
				a.binner()
				e_val = a.quasi_normalize_structure_factors()
				E = list(e_val.expand_to_p1().f_sq_as_f().data())
				# Convert miller hkl to polar P1 space group
				polar_coordinates = list(C.reciprocal_space_vector(P1))
				X = []
				Y = []
				Z = []
				for x, y, z in polar_coordinates:
					X.append(x)
					Y.append(y)
					Z.append(z)
				C = str(C)[1:-1]
				C = tuple(map(str, C.split(', ')))
				return(S, C, X, Y, Z, R, E)
	def labels(self, filename):
		structure = Bio.PDB.PDBParser(QUIET=True).get_structure('X', filename)
		dssp = Bio.PDB.DSSP(structure[0], filename, acc_array='Wilke')
		SS = []
		for aa in dssp:
			if aa[2] == 'G' or aa[2] == 'H' or aa[2] == 'I':   ss = 'H'
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
		with open('temp', 'a') as TheFile:
			size = []
			with open(IDs) as f:
				line = f.read().strip().lower().split(',')
				for item in tqdm.tqdm(line):
					try:
						self.download(item)
					except:
						print('\u001b[31m[-] {} Failed to download\u001b[0m'\
						.format(item.upper()))
						continue
					try:
						Mfilename = item + '.mtz'
						Pfilename = item + '.pdb'
						S,C,X,Y,Z,R,E = self.features(Mfilename)
						H_frac, S_frac, L_frac = self.labels(Pfilename)
						if   H_frac >= 0.50 and S_frac == 0.00 and L_frac <= 0.50:
							label = 'Helix'
						elif H_frac == 0.00 and S_frac >= 0.50 and L_frac <= 0.50:
							label = 'Sheet'
						assert len(X) == len(Y) == len(Z) == len(R) == len(E),\
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
						for x, y, z, r, e in zip(X, Y, Z, R, E):
							x = str(round(x, 5))
							y = str(round(y, 5))
							z = str(round(z, 5))
							r = str(round(r, 5))
							e = str(round(e, 5))
							exp.append(x+','+y+','+z+','+r+','+e)
						example = ','.join(exp)
						TheFile.write(item.upper()+'.mtz'+','+label+',')
						TheFile.write(example + '\n')
						size.append(len(X))
						os.remove(Mfilename)
						os.remove(Pfilename)
					except Exception as e:
						print('\u001b[31m[-] {} Failed: not compiling\u001b[0m'\
						.format(item.upper()))
						continue
			header = ['PDB_ID,Label,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max(size)+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},E-value_{}'\
				.format(i, i, i, i, i))
			header = ''.join(header)
		with open('DeepClass.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(header + '\n')
				for line in t: f.write(line)
		os.remove('temp')

def VectoriseClass_NR(filename='DeepClass.csv',
	max_size='1000', Rmin=3.00, Rmax=10.0, Emin=0.00, Emax=9.00,
	fp=np.float64, ip=np.int64, Pids=False):
	'''
	Since the .csv file cannot be loaded into RAM even that of a supercomputer,
	this function vectorises the dataset normalises it as well as construct the
	final tensors and export the result as a serial. It also allows construction
	of datasets with different point sizes
	'''
	# 1. Find max_size number
	try:
		max_size = int(max_size)
		tick = True
	except:
		with open(filename) as f:
			header = f.readline()
			header = header.strip().split(',')[9:]
			max_size = int(len(header)/5)
		tick = False
	# 2. Find number of rows
	rows = len(open(filename).readlines()) - 1
	# 3. Generate a list of random number of rows
	lines = list(range(1, rows + 1))
	random.shuffle(lines)
	# 4. Open CSV file
	with open(filename, 'r') as File:
		all_lines_variable = File.readlines()
	L = np.array([])
	S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
	I = np.array([])
	for i in lines:
		# 5. Isolate labels and crystal data columns
		line= all_lines_variable[i]
		line= line.strip().split(',')
		I = np.append(I, np.array(str(line[0]), dtype=str))
		L = np.append(L, np.array(str(line[1]), dtype=str))
		S.append(np.array(int(line[2]), dtype=ip))
		UCe.append(np.array([float(i) for i in line[3:6]], dtype=fp))
		UCa.append(np.array([float(i) for i in line[6:9]], dtype=fp))
		# 6. Isolate points data columns
		Pts = line[9:]
		Pts = [float(i) for i in Pts]
		# 7. Sort and turnicate
		if tick == True:
			x = Pts[0::5]
			y = Pts[1::5]
			z = Pts[2::5]
			r = Pts[3::5]
			e = Pts[4::5]
			NC = []
			for xx, yy, zz, rr, ee in zip(x, y, z, r, e):
				if abs(xx)>0.1 and abs(yy)>0.1 and zz>0.1 and Rmin<=rr and abs(ee-1)>0.5 and ee!=0: NC.append((xx, yy, zz, rr, ee))
				#if xx!=0.0 and xx!=-0.0 and yy!=0.0 and yy!=-0.0 and zz>0 and Rmin<=rr<=Rmax and ee!=1.0: NC.append((xx, yy, zz, rr, ee))
				#if Rmin<=rr<=Rmax and Emin<=ee<=Emax: NC.append((xx, yy, zz, rr, ee))
			# 7.5 Sort and choose according to largest R
			Pts = sorted(NC, reverse=True, key=lambda c:c[3])
			Pts = Pts[:max_size]
			Pts = [i for sub in Pts for i in sub]
		# 8. Isolate different points data
		Xp = Pts[0::5]
		Yp = Pts[1::5]
		Zp = Pts[2::5]
		Rp = Pts[3::5]
		Ep = Pts[4::5]
		# 9. Padding
		if len(Xp) < max_size:
			dif = max_size - len(Xp)
			for i in range(dif):
				Xp.append(0.0)
				Yp.append(0.0)
				Zp.append(0.0)
				Rp.append(0.0)
				Ep.append(0.0)
		assert len(Xp) == max_size, 'Max number of points incorrect'
		# 10. Export points
		X.append(np.array(Xp, dtype=fp))
		Y.append(np.array(Yp, dtype=fp))
		Z.append(np.array(Zp, dtype=fp))
		R.append(np.array(Rp, dtype=fp))
		E.append(np.array(Ep, dtype=fp))
	# 11. Build arrays
	assert len(X[0]) == len(X[1]), 'Max number of points incorrect'
	I   = np.array(I)
	S   = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X   = np.array(X)
	Y   = np.array(Y)
	Z   = np.array(Z)
	R   = np.array(R)
	E   = np.array(E)
	# 12. One-Hot encoding and normalisation
	''' Y labels '''
	L[L=='Helix'] = 0
	L[L=='Sheet'] = 1
	y = L.astype(np.int)
	''' X features '''
	categories = [sorted([x for x in range(1, 230+1)])]
	S = S.reshape(-1, 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
	S   = onehot_encoder.fit_transform(S)                # One-hot encode [Space Groups]
	UCe = (UCe-np.amin(UCe))/(np.amax(UCe)-np.amin(UCe)) # Normalise      [Unit Cell Edges]
	UCa = (UCa-np.amin(UCa))/(np.amin(UCa)-np.amin(UCa)) # Normalise      [Unit Cell Angles]
	X   = (X-np.amin(X))/(np.amax(X)-np.amin(X))         # Normalise      [X Coordinates]
	Y   = (Y-np.amin(Y))/(np.amax(Y)-np.amin(Y))         # Normalise      [Y Coordinates]
	Z   = (Z-np.amin(Z))/(np.amax(Z)-np.amin(Z))         # Normalise      [Z Coordinates]
	R   = (R-np.amin(R))/(np.amax(R)-np.amin(R))         # Normalise      [Resolution]
	E   = (E-np.amin(E))/(np.amax(E)-np.amin(E))         # Normalise      [E-value]
	# 13. Construct tensors - final features
	Space = S
	UnitC = np.concatenate([UCe, UCa], axis=1)
	Coord = np.array([X, Y, Z, R, E])
	Coord = np.swapaxes(Coord, 0, 2)
	Coord = np.swapaxes(Coord, 0, 1)
	S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
	print('I =', I.shape)
	print('Y =', y.shape)
	print('Space =', Space.shape)
	print('UnitC =', UnitC.shape)
	print('Coord =', Coord.shape)
	# 14. Serialise tensors
	with h5py.File('Y.hdf5', 'w') as Yh:
		dset = Yh.create_dataset('default', data=y)
	with h5py.File('Space.hdf5', 'w') as Sh:
		dset = Sh.create_dataset('default', data=Space)
	with h5py.File('UnitC.hdf5', 'w') as Uh:
		dset = Uh.create_dataset('default', data=UnitC)
	with h5py.File('Coord.hdf5', 'w') as Ch:
		dset = Ch.create_dataset('default', data=Coord)
	if Pids == True:
		I = [n.encode('ascii', 'ignore') for n in I]
		with h5py.File('IDs.hdf5', 'w') as ii:
			dset = ii.create_dataset('default', data=I)

def VectoriseClass_SD(filename='DeepClass.csv',
	max_size='1000', Rmin=3.00, Rmax=10.0, Emin=0.00, Emax=9.00,
	fp=np.float64, ip=np.int64, Pids=False):
	'''
	Since the .csv file cannot be loaded into RAM even that of a supercomputer,
	this function vectorises the dataset standerdise it as well as construct the
	final tensors and export the result as a serial. It also allows construction
	of datasets with different point sizes
	'''
	# 1. Find max_size number
	try:
		max_size = int(max_size)
		tick = True
	except:
		with open(filename) as f:
			header = f.readline()
			header = header.strip().split(',')[9:]
			max_size = int(len(header)/5)
		tick = False
	# 2. Find number of rows
	rows = len(open(filename).readlines()) - 1
	# 3. Generate a list of random number of rows
	lines = list(range(1, rows + 1))
	random.shuffle(lines)
	# 4. Open CSV file
	with open(filename, 'r') as File:
		all_lines_variable = File.readlines()
	L = np.array([])
	S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
	I = np.array([])
	for i in lines:
		# 5. Isolate labels and crystal data columns
		line= all_lines_variable[i]
		line= line.strip().split(',')
		I = np.append(I, np.array(str(line[0]), dtype=str))
		L = np.append(L, np.array(str(line[1]), dtype=str))
		S.append(np.array(int(line[2]), dtype=ip))
		UCe.append(np.array([float(i) for i in line[3:6]], dtype=fp))
		UCa.append(np.array([float(i) for i in line[6:9]], dtype=fp))
		# 6. Isolate points data columns
		Pts = line[9:]
		Pts = [float(i) for i in Pts]
		# 7. Sort and turnicate
		if tick == True:
			x = Pts[0::5]
			y = Pts[1::5]
			z = Pts[2::5]
			r = Pts[3::5]
			e = Pts[4::5]
			NC = []
			for xx, yy, zz, rr, ee in zip(x, y, z, r, e):
				if abs(xx)>0.1 and abs(yy)>0.1 and zz>0.1 and Rmin<=rr and abs(ee-1)>0.5 and ee!=0: NC.append((xx, yy, zz, rr, ee))
				#if xx!=0.0 and xx!=-0.0 and yy!=0.0 and yy!=-0.0 and zz>0 and Rmin<=rr<=Rmax and ee!=1.0: NC.append((xx, yy, zz, rr, ee))
				#if Rmin<=rr<=Rmax and Emin<=ee<=Emax: NC.append((xx, yy, zz, rr, ee))
			# 7.5 Sort and choose according to largest R
			Pts = sorted(NC, reverse=True, key=lambda c:c[3])
			Pts = Pts[:max_size]
			Pts = [i for sub in Pts for i in sub]
		# 8. Isolate different points data
		Xp = Pts[0::5]
		Yp = Pts[1::5]
		Zp = Pts[2::5]
		Rp = Pts[3::5]
		Ep = Pts[4::5]
		# 9. Padding
		if len(Xp) < max_size:
			dif = max_size - len(Xp)
			for i in range(dif):
				Xp.append(0.0)
				Yp.append(0.0)
				Zp.append(0.0)
				Rp.append(0.0)
				Ep.append(0.0)
		assert len(Xp) == max_size, 'Max number of points incorrect'
		# 10. Export points
		X.append(np.array(Xp, dtype=fp))
		Y.append(np.array(Yp, dtype=fp))
		Z.append(np.array(Zp, dtype=fp))
		R.append(np.array(Rp, dtype=fp))
		E.append(np.array(Ep, dtype=fp))
	# 11. Build arrays
	assert len(X[0]) == len(X[1]), 'Max number of points incorrect'
	I   = np.array(I)
	S   = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X   = np.array(X)
	Y   = np.array(Y)
	Z   = np.array(Z)
	R   = np.array(R)
	E   = np.array(E)
	# 12. One-Hot encoding and normalisation
	''' Y labels '''
	L[L=='Helix'] = 0
	L[L=='Sheet'] = 1
	y = L.astype(np.int)
	''' X features '''
	categories = [sorted([x for x in range(1, 230+1)])]
	S = S.reshape(-1, 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
	S = onehot_encoder.fit_transform(S)        # One-hot encode [Space Groups]
	UCe = (UCe-np.mean(UCe))/np.std(UCe)       # Standardise    [Unit Cell Edges]
	UCa = (UCa-np.mean(UCa))/np.std(UCa)       # Standardise    [Unit Cell Angles]
	X = (X-np.mean(X))/np.std(X)               # Standardise    [X Coordinates]
	Y = (Y-np.mean(Y))/np.std(Y)               # Standardise    [Y Coordinates]
	Z = (Z-np.amin(Z))/(np.amax(Z)-np.amin(Z)) # Normalise      [Z Coordinates]
	R = (R-np.amin(R))/(np.amax(R)-np.amin(R)) # Normalise      [Resolution]
	E = (E-np.amin(E))/(np.amax(E)-np.amin(E)) # Normalise      [E-value]
	# 13. Construct tensors - final features
	Space = S
	UnitC = np.concatenate([UCe, UCa], axis=1)
	Coord = np.array([X, Y, Z, R, E])
	Coord = np.swapaxes(Coord, 0, 2)
	Coord = np.swapaxes(Coord, 0, 1)
	S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
	print('I =', I.shape)
	print('Y =', y.shape)
	print('Space =', Space.shape)
	print('UnitC =', UnitC.shape)
	print('Coord =', Coord.shape)
	# 14. Serialise tensors
	with h5py.File('Y.hdf5', 'w') as Yh:
		dset = Yh.create_dataset('default', data=y)
	with h5py.File('Space.hdf5', 'w') as Sh:
		dset = Sh.create_dataset('default', data=Space)
	with h5py.File('UnitC.hdf5', 'w') as Uh:
		dset = Uh.create_dataset('default', data=UnitC)
	with h5py.File('Coord.hdf5', 'w') as Ch:
		dset = Ch.create_dataset('default', data=Coord)
	if Pids == True:
		I = [n.encode('ascii', 'ignore') for n in I]
		with h5py.File('IDs.hdf5', 'w') as ii:
			dset = ii.create_dataset('default', data=I)

class PhaseData():
	''' Build a dataset for phase calculation from x-ray diffractions '''
	def download(self, filename):
		''' Downloads a structure's .mtz file '''
		url = 'http://edmaps.rcsb.org/coefficients/{}.mtz'.format(filename)
		urllib.request.urlretrieve(url, '{}.mtz'.format(filename))
	def features(self, filename):
		'''
		Extracts required features from .mtz files. More info can be found here:
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
				ms = ms_all.customized_copy(space_group_info=a.space_group_info())
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
				# Calculated Phases
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
	def run(self, IDs='IDs.txt'):
		with open('temp', 'a') as temp:
			size = []
			with open(IDs) as f:
				line = f.read().strip().lower().split(',')
				for item in tqdm.tqdm(line):
					try:
						self.download(item)
					except:
						print('\u001b[31m[-] {} Failed: could not download\u001b[0m'.format(item.upper()))
						continue
					try:
						filename = item + '.mtz'
						S, C, X, Y, Z, R, F, P = self.features(filename)
						assert len(X) == len(Y) == len(Z) == len(R) == len(F) == len(P),\
						'\u001b[31m[-] {} Failed: values not equal\u001b[0m'.format(item.upper())
						exp = [S]
						a     = C[0]
						b     = C[1]
						c     = C[2]
						alpha = C[3]
						beta  = C[4]
						gamma = C[5]
						exp.append(a+','+b+','+c+','+alpha+','+beta+','+gamma)
						for x, y, z, r, f, p in zip(X, Y, Z, R, F, P):
							x = str(round(x, 5))
							y = str(round(y, 5))
							z = str(round(z, 5))
							r = str(round(r, 5))
							f = str(round(f, 5))
							p = str(round(p, 5))
							exp.append(x+','+y+','+z+','+r+','+f+','+p)
						example = ','.join(exp)
						temp.write(item.upper()+'.mtz'+',')
						temp.write(example + '\n')
						size.append(len(X))
						os.remove(filename)
					except:
						print('\u001b[31m[-] {} Failed: problem compiling\u001b[0m'.format(item.upper()))
						continue
			header = ['PDB_ID,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max(size)+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},E-value_{},Phase_{}'\
				.format(i, i, i, i, i, i))
			header = ''.join(header)
		with open('DeepPhase.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(header + '\n')
				for line in t: f.write(line)
		os.remove('temp')

def VectorisePhase(filename='DeepPhase.csv', fp=np.float16, ip=np.int16):
	'''
	Since the .csv file cannot be loaded into RAM even that of a supercomputer,
	this function vectorises the dataset normalises it as well as construct the
	final tensors and export the result as a serial.
	'''
	# 1. Find max_size number
	with open(filename) as f:
		header = f.readline()
		header = header.strip().split(',')[8:]
		max_size = int(len(header)/6)
	# 2. Find number of rows
	rows = len(open(filename).readlines()) - 1
	# 3. Generate a list of random number of rows
	lines = list(range(1, rows + 1))
	random.shuffle(lines)
	# 4. Open CSV file
	with open(filename, 'r') as File: all_lines_variable = File.readlines()
	S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
	for i in lines:
		# 5. Isolate labels and crystal data columns
		line= all_lines_variable[i]
		line= line.strip().split(',')
		# 5.1 Isolate structures < 10,000 reflection
		if len(line[8:]) <= 10000*6:
			S.append(np.array(int(line[1]), dtype=ip))
			UCe.append(np.array([float(i) for i in line[2:5]], dtype=fp))
			UCa.append(np.array([float(i) for i in line[5:8]], dtype=fp))
			# 6. Isolate points data columns
			Pts = line[8:]
			Pts = [float(i) for i in Pts]
			# 7. Isolate different points data
			Xp = Pts[0::6]
			Yp = Pts[1::6]
			Zp = Pts[2::6]
			Rp = Pts[3::6]
			Ep = Pts[4::6]
			Pp = Pts[5::6]
			# 8. Padding
			if len(Xp) < max_size:
				dif = max_size - len(Xp)
				for i in range(dif):
					Xp.append(-0.4)
					Yp.append(-0.4)
					Zp.append(-0.4)
					Rp.append(2.5)
					Ep.append(0.0)
					Pp.append(-3.14159)
			assert len(Xp) == max_size, 'Max number of points incorrect'
			# 9. Export points
			X.append(np.array(Xp, dtype=fp))
			Y.append(np.array(Yp, dtype=fp))
			Z.append(np.array(Zp, dtype=fp))
			R.append(np.array(Rp, dtype=fp))
			E.append(np.array(Ep, dtype=fp))
			P.append(np.array(Pp, dtype=fp))
	# 10. Build arrays
	assert len(X[0]) == len(X[1]), 'Max number of points incorrect'
	S = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	R = np.array(R)
	E = np.array(E)
	P = np.array(P)
	# 11. One-Hot encoding and normalisation
	''' Y labels '''
	mini = np.amin(P)
	maxi = np.amax(P)
	Phase = (P-mini)/(maxi-mini)               # Normalise      [Phase]
	''' X features '''
	categories = [sorted([x for x in range(1, 230+1)])]
	S = S.reshape(-1, 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
	S = onehot_encoder.fit_transform(S)        # One-hot encode [Space Groups]
	UCe = (UCe-np.mean(UCe))/np.std(UCe)       # Standardise    [Unit Cell Edges]
	UCa = (UCa-np.mean(UCa))/np.std(UCa)       # Standardise    [Unit Cell Angles]
	X = (X-np.mean(X))/np.std(X)               # Standardise    [X Coordinates]
	Y = (Y-np.mean(Y))/np.std(Y)               # Standardise    [Y Coordinates]
	Z = (Z-np.amin(Z))/(np.amax(Z)-np.amin(Z)) # Normalise      [Z Coordinates]
	R = (R-np.amin(R))/(np.amax(R)-np.amin(R)) # Normalise      [Resolution]
	E = (E-np.amin(E))/(np.amax(E)-np.amin(E)) # Normalise      [E-value]
	# 12. Construct tensors - final features
	Space = S
	UnitC = np.concatenate([UCe, UCa], axis=1)
	Coord = np.array([X, Y, Z, R, E])
	Coord = np.swapaxes(Coord, 0, 2)
	Coord = np.swapaxes(Coord, 0, 1)
	S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
	# 13. Serialise tensors
	print('Phase =', Phase.shape)
	print('Space =', Space.shape)
	print('UnitC =', UnitC.shape)
	print('Coord =', Coord.shape)
	with h5py.File('Phase.hdf5', 'w') as Ph:
		dset = Ph.create_dataset('default', data=Phase)
	with h5py.File('Space.hdf5', 'w') as Sh:
		dset = Sh.create_dataset('default', data=Space)
	with h5py.File('UnitC.hdf5', 'w') as Uh:
		dset = Uh.create_dataset('default', data=UnitC)
	with h5py.File('Coord.hdf5', 'w') as Ch:
		dset = Ch.create_dataset('default', data=Coord)

def main():
	if  args.Class:
		Cls = ClassData()
		Cls.run(IDs=sys.argv[2])
	elif args.Phase:
		Phs = PhaseData()
		Phs.run(IDs=sys.argv[2])
	elif args.VecClass:
#		VectoriseClass_NR(filename=sys.argv[2], max_size=sys.argv[3])
		VectoriseClass_SD(filename=sys.argv[2], max_size=sys.argv[3])
	elif args.VecPhase:
		VectorisePhase(filename=sys.argv[2])

if __name__ == '__main__': main()
