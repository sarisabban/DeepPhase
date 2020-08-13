import os
import sys
import h5py
import tqdm
import random
import Bio.PDB
import numpy as np
import urllib.request
from iotbx.reflection_file_reader import any_reflection_file
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class ClassData():
	'''
	Build a dataset for protein classification (Alpha/Not_Alpha)
	from x-ray diffraction and get top 10,000 F-Obs reflections only
	'''
	def download(self, ID):
		''' Downloads a structure's .mtz and .pdb files '''
		Murl = 'http://edmaps.rcsb.org/coefficients/{}.mtz'.format(ID)
		Purl = 'https://files.rcsb.org/download/{}.pdb'.format(ID)
		urllib.request.urlretrieve(Murl, '{}.mtz'.format(ID))
		urllib.request.urlretrieve(Purl, '{}.pdb'.format(ID))
	def features(self, filename, size=10000):
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
				rr = list(C.d(P1))
				# Space Group
				ms_base = a.customized_copy()
				ms_all = ms_base.complete_set()
				ms=ms_all.customized_copy(space_group_info=a.space_group_info())
				S = str(ms).split()[-1].split(')')[0]
				# F-obs
				ff = list(a.expand_to_p1().f_sq_as_f().data())
				# Convert miller hkl to polar
				polar_coordinates = list(C.reciprocal_space_vector(P1))
				xx = []
				yy = []
				zz = []
				for x, y, z in polar_coordinates:
					xx.append(x)
					yy.append(y)
					zz.append(z)
				C = str(C)[1:-1]
				C = tuple(map(str, C.split(', ')))
				NC = []
				for x, y, z, r, f in zip(xx, yy, zz, rr, ff):
					if r >= 10.0 or r <= 2.5: continue
					else: NC.append((x, y, z, r, f))
				# Sort and choose top max_size F-Obs
				SORTED = sorted(NC, reverse=True, key=lambda c:c[4])
				SORTED = SORTED[:size]
				X, Y, Z, R, F = [], [], [], [], []
				for i in SORTED:
					X.append(i[0])
					Y.append(i[1])
					Z.append(i[2])
					R.append(i[3])
					F.append(i[4])
				return(S, C, X, Y, Z, R, F)
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
	def run(self, IDs='IDs.txt', max_size=10000):
		with open('DeepClass.csv', 'a') as TheFile:
			header = ['PDB_ID,Label,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max_size+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},F-obs_{}'\
				.format(i, i, i, i, i))
			header = ''.join(header)
			TheFile.write(header + '\n')
			size = []
			with open(IDs) as f:
				line = f.read().strip().lower().split(',')
				for item in tqdm.tqdm(line):
					try:
						self.download(item)
					except:
						print('\u001b[31m[-] {} failed to download\u001b[0m'\
						.format(item.upper()))
						continue
					try:
						Mfilename = item + '.mtz'
						Pfilename = item + '.pdb'
						S,C,X,Y,Z,R,F = self.features(Mfilename, size=max_size)
						H_frac, S_frac, L_frac = self.labels(Pfilename)
						if H_frac >= 0.5 or S_frac == 0.0:
							label = 'Alpha'
						else:
							label = 'Not_Alpha'
						assert len(X) == len(Y) == len(Z) == len(R) == len(F),\
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
						for x, y, z, r, f in zip(X, Y, Z, R, F):
							x = str(round(x, 5))
							y = str(round(y, 5))
							z = str(round(z, 5))
							r = str(round(r, 5))
							f = str(round(f, 5))
							exp.append(x+','+y+','+z+','+r+','+f)
						example = ','.join(exp)
						TheFile.write(item.upper()+'.mtz'+','+label+',')
						TheFile.write(example + '\n')
						size.append(len(X))
						os.remove(Mfilename)
						os.remove(Pfilename)
					except Exception as e:
						print('\u001b[31m[-] {} failed compiling\u001b[0m'\
						.format(item.upper()))
						continue

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
				# F-obs
				F = list(a.expand_to_p1().f_sq_as_f().data())
		nX, nY, nZ, nR, nF, nP = [], [], [], [], [], []
		for x, y, z, r, f, p in zip(X, Y, Z, R, F, P):
			if r >= 10.0 or r <= 2.5:
				continue
			else:
				nX.append(x)
				nY.append(y)
				nZ.append(z)
				nR.append(r)
				nF.append(f)
				nP.append(p)
		X = nX
		Y = nY
		Z = nZ
		R = nR
		F = nF
		P = nP
		return(S, C, X, Y, Z, R, F, P)
	def run(self, IDs='IDs.txt'):
		with open('temp', 'w') as temp:
			size = []
			with open(IDs) as f:
				line = f.read().strip().lower().split(',')
				for item in tqdm.tqdm(line):
					try:
						self.download(item)
					except:
						print('\u001b[31m[-] {} failed to download\u001b[0m'\
						.format(item.upper()))
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
						print('\u001b[31m[-] {} failed compiling\u001b[0m'\
						.format(item.upper()))
						continue
			header = ['PDB_ID,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max(size)+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},F-obs_{},Phase{}'\
				.format(i, i, i, i, i, i))
			header = ''.join(header)
		with open('DeepPhase.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(header + '\n')
				for line in t: f.write(line)
		os.remove('temp')

def Vectorise_Class(filename='DeepClass.csv', fp=np.float16, ip=np.int16):
	'''
	Since the .csv file cannot be loaded into RAM even that of a supercomputer,
	this function vectorises the dataset normalises it as well as construct the
	final tensors and export the result as a serial.
	'''
	# 1. Find number of rows
	rows = len(open(filename).readlines()) - 1
	# 2. Generate a list of random number of rows
	lines = list(range(1, rows + 1))
	random.shuffle(lines)
	# 3. Open CSV file
	File = open(filename)
	# 4. Import a single row
	all_lines_variable = File.readlines()
	L, S, UCe, UCa, X, Y, Z, R, F = [], [], [], [], [], [], [], [], []
	for i in lines:
		# 5. Isolate labels and crystal data columns
		line = all_lines_variable[i]
		line = line.strip().split(',')
		L.append(np.array(str(line[1]), dtype=str))
		S.append(np.array(int(line[2]), dtype=ip))
		UCe.append(np.array([float(i) for i in line[3:6]], dtype=fp))
		UCa.append(np.array([float(i) for i in line[6:9]], dtype=fp))
		# 6. Isolate points data columns
		Pts = line[9:]
		Pts = [float(i) for i in Pts]
		# 7. Fill gaps with zeros
		dif = 50000 - len(Pts)
		Pts.extend([0.0]*dif)
		# 8. Isolate different points data
		X.append(np.array(Pts[0::5], dtype=fp))
		Y.append(np.array(Pts[1::5], dtype=fp))
		Z.append(np.array(Pts[2::5], dtype=fp))
		R.append(np.array(Pts[3::5], dtype=fp))
		F.append(np.array(Pts[4::5], dtype=fp))
	# 9. Construct matrices
	L  = np.array(L)
	S  = np.array(S)
	UCe= np.array(UCe)
	UCa= np.array(UCa)
	X  = np.array(X)
	Y  = np.array(Y)
	Z  = np.array(Z)
	R  = np.array(R)
	F  = np.array(F)
	# 10. One-Hot encoding and normalisation
	''' Y labels '''
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(L)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	y = onehot_encoder.fit_transform(integer_encoded)
	y = np.float16(y)
	''' X features '''
	categories = [sorted([x for x in range(1, 230+1)])]
	S = S.reshape(-1, 1)
	onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
	S = onehot_encoder.fit_transform(S) # One-hot encode S   [Space Groups]
	mini = np.amin(UCe)
	maxi = np.amax(UCe)
	UCe = (UCe-mini)/(maxi-mini)     # Normalise min/max UCe [Unit Cell Edges]
	mini = 90.0
	maxi = 180.0
	UCa = (UCa-mini)/(maxi-mini)     # Normalise min/max UCa [Unit Cell Angles]
	mini = -1
	maxi = 1
	X = (X-mini)/(maxi-mini)         # Normalise min/max X   [X Coordinates]
	mini = -1
	maxi = 1
	Y = (Y-mini)/(maxi-mini)         # Normalise min/max Y   [Y Coordinates]
	mini = -1
	maxi = 1
	Z = (Z-mini)/(maxi-mini)         # Normalise min/max Z   [Z Coordinates]
	mini = 2.5
	maxi = 10
	R = (R-mini)/(maxi-mini)         # Normalise min/max R   [Resolution]
	#mini = np.amin(F)
	#maxi = np.amax(F)
	#F = (F-mini)/(maxi-mini)        # Normalise min/max F   [F-Obs](Already Normalised)
	# 11. Construct tensors - final features
	Space = S
	UnitC = np.concatenate([UCe, UCa], axis=1)
	Coord = np.array([X, Y, Z, R, F])
	Coord = np.swapaxes(Coord, 0, 2)
	Coord = np.swapaxes(Coord, 0, 1)
	S, UCe, UCa, X, Y, Z, R, F = [], [], [], [], [], [], [], []
	# 12. Serialise tensors
	with h5py.File('Y.hdf5', 'w') as Yh:
		dset = Yh.create_dataset('default', data=y)
	with h5py.File('Space.hdf5', 'w') as Sh:
		dset = Sh.create_dataset('default', data=Space)
	with h5py.File('UnitC.hdf5', 'w') as Uh:
		dset = Uh.create_dataset('default', data=UnitC)
	with h5py.File('Coord.hdf5', 'w') as Ch:
		dset = Ch.create_dataset('default', data=Coord)
	
def main():
	if sys.argv[1] == 'class':
		Cls = ClassData()
		Cls.run(IDs='Class.txt', max_size=10000)
	elif sys.argv[1] == 'phase':
		Phs = PhaseData()
		Phs.run(IDs='Phase.txt')
	elif sys.argv[1] == 'vectorise_class':
		Vectorise_Class()
	else:
		print('\u001b[31m[-] Wrong argument\u001b[0m')

if __name__ == '__main__': main()
