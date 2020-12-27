import os
import sys
import h5py
import tqdm
import random
import urllib
import Bio.PDB
import argparse
import numpy as np
from sklearn.utils import shuffle
from iotbx.reflection_file_reader import any_reflection_file
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

parser = argparse.ArgumentParser(description='Compiling X-ray diffraction datasets')
parser.add_argument('-D', '--Dataset'  , nargs='+', help='Compile a datset of protein reflections points, include a text file of PDB IDs')
parser.add_argument('-V', '--Vectorise', nargs='+', help='Vectorise and serialise the datset')
args = parser.parse_args()

class Dataset():
	'''
	Build a .csv dataset for protein X-ray crystal diffraction points
	from .mtz and .pdb files and compile all
	'''
	def download(self, ID):
		''' Downloads a structure's .mtz and .pdb files. '''
		Murl = 'http://edmaps.rcsb.org/coefficients/{}.mtz'.format(ID)
		Purl = 'https://files.rcsb.org/download/{}.pdb'.format(ID)
		urllib.request.urlretrieve(Murl, '{}.mtz'.format(ID))
		urllib.request.urlretrieve(Purl, '{}.pdb'.format(ID))
	def features(self, filename):
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
		with open('temp', 'a') as temp:
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
					try:
						Mfilename = item + '.mtz'
						Pfilename = item + '.pdb'
						Mfilename = './X/' + item + '.mtz'
						Pfilename = './X/' + item + '.pdb'
						S, C, X, Y, Z, R, E, P = self.features(Mfilename)
						H_frac, S_frac, L_frac = self.labels(Pfilename)
						if   H_frac>=0.50 and S_frac==0.00 and L_frac<=0.50:
							label = 'Helix'
						elif H_frac==0.00 and S_frac>=0.50 and L_frac<=0.50:
							label = 'Sheet'
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
						temp.write(item.upper()+','+label+',')
						temp.write(example + '\n')
						size.append(len(X))
						os.remove(Mfilename)
						os.remove(Pfilename)
					except:
						red = '\u001b[31m'
						ret = '\u001b[0m'
						print('{}[-] {} Failed: problem compiling{}'\
						.format(item.upper()))
						continue
			header = ['PDB_ID,Class,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max(size)+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},E-value_{},Phase_{}'\
				.format(i, i, i, i, i, i))
			header = ''.join(header)
		with open('CrystalDataset.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(header + '\n')
				for line in t: f.write(line)
		os.remove('temp')

def Vectorise(filename='CrystalDataset.csv', max_size='15000',
	Pids=False, export=True, phase=False, Type='class',
	fp=np.float64, ip=np.int64):
	'''
	For DeepClass:
	Since the .csv file for this dataset would require larger RAM than what is
	currently available, yet we still need to train a network on as much
	reflection points as possible this function randomly samples all points of
	each example, vectorises the sampled points, standerdises them, constructs
	the final tensors then either outputs the result or export it as a serial.
	It also allows construction of the dataset with different point sizes.
	----------------------------------------------------------------------------
	For DeepPhase:
	???
	???
	???
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
			# 3. Random sampling of points
			NC = [(x,y,z,r,e,p) for x,y,z,r,e,p in zip(
													T[0::6],T[1::6],T[2::6],
													T[3::6],T[4::6],T[5::6])]
			T = [random.choice(NC) for x in range(max_size)]
			assert len(T) == max_size, 'Max number of points incorrect'
			T = [i for sub in T for i in sub]
			# 4. Export points
			X.append(np.array(T[0::6], dtype=fp))
			Y.append(np.array(T[1::6], dtype=fp))
			Z.append(np.array(T[2::6], dtype=fp))
			R.append(np.array(T[3::6], dtype=fp))
			E.append(np.array(T[4::6], dtype=fp))
			P.append(np.array(T[5::6], dtype=fp))
	# 5. Build arrays
	I   = np.array(I)
	S   = np.array(S)
	UCe = np.array(UCe)
	UCa = np.array(UCa)
	X   = np.array(X)
	Y   = np.array(Y)
	Z   = np.array(Z)
	R   = np.array(R)
	E   = np.array(E)
	E   = np.array(P)
	if Type == 'class' or Type == 'Class':
		# 6. One-Hot encoding and normalisation
		''' Y labels '''
		L[L=='Helix'] = 0
		L[L=='Sheet'] = 1
		Class = L.astype(np.int)
		''' X features '''
		categories = [sorted([x for x in range(1, 230+1)])]
		S = S.reshape(-1, 1)
		onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
		S = onehot_encoder.fit_transform(S)       #One-hot encode [Space Groups]
		UCe = (UCe-np.mean(UCe))/np.std(UCe)      #Standardise [Unit Cell Edges]
		UCa = (UCa-np.mean(UCa))/np.std(UCa)      #Standardise [Unit Cell Angle]
		X = (X-np.mean(X))/np.std(X)              #Standardise [X Coordinates]
		Y = (Y-np.mean(Y))/np.std(Y)              #Standardise [Y Coordinates]
		Z = (Z-np.amin(Z))/(np.amax(Z)-np.amin(Z))#Normalise   [Z Coordinates]
		R = (R-np.amin(R))/(np.amax(R)-np.amin(R))#Normalise   [Resolution]
		E = (E-np.amin(E))/(np.amax(E)-np.amin(E))#Normalise   [E-value]
		P = (P-np.amin(P))/(np.amax(P)-np.amin(P))#Normalise   [Phases]
		# 7. Construct tensors
		Space = S
		UnitC = np.concatenate([UCe, UCa], axis=1)
		if phase: Coord = np.array([X, Y, Z, R, E, P])
		else: Coord = np.array([X, Y, Z, R, E])
		Coord = np.swapaxes(Coord, 0, 2)
		Coord = np.swapaxes(Coord, 0, 1)
		S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
		# 8. Shuffle examples
		Coord, UnitC, Space, Class, I = shuffle(Coord, UnitC, Space, Class, I)
		if export:
			# 9. Serialise tensors
			with h5py.File('Class.hdf5', 'w') as Yh:
				dset = Yh.create_dataset('default', data=Class)
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
			print('IDs     =', I.shape)
			print('Space   =', Space.shape)
			print('UnitC   =', UnitC.shape)
			print('X Coord =', Coord.shape)
			print('Y Class =', Class.shape)
		else: return(Coord, UnitC, Space, y, I)
	elif Type == 'phase' or Type == 'Phase':
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
		Phase = (P-np.amin(P))/(np.amax(P)-np.amin(P))# Normalise [Phase]
		''' X features '''
		categories = [sorted([x for x in range(1, 230+1)])]
		S = S.reshape(-1, 1)
		onehot_encoder = OneHotEncoder(sparse=False, categories=categories)
		S = onehot_encoder.fit_transform(S)       #One-hot encode [Space Groups]
		UCe = (UCe-np.mean(UCe))/np.std(UCe)      #Standardise [Unit Cell Edges]
		UCa = (UCa-np.mean(UCa))/np.std(UCa)      #Standardise [Unit Cell Angle]
		X = (X-np.mean(X))/np.std(X)              #Standardise [X Coordinates]
		Y = (Y-np.mean(Y))/np.std(Y)              #Standardise [Y Coordinates]
		Z = (Z-np.amin(Z))/(np.amax(Z)-np.amin(Z))#Normalise   [Z Coordinates]
		R = (R-np.amin(R))/(np.amax(R)-np.amin(R))#Normalise   [Resolution]
		E = (E-np.amin(E))/(np.amax(E)-np.amin(E))#Normalise   [E-value]
		# 12. Construct tensors - final features
		Space = S
		UnitC = np.concatenate([UCe, UCa], axis=1)
		Coord = np.array([X, Y, Z, R, E])
		Coord = np.swapaxes(Coord, 0, 2)
		Coord = np.swapaxes(Coord, 0, 1)
		S, UCe, UCa, X, Y, Z, R, E = [], [], [], [], [], [], [], []
		# 13. Serialise tensors
		if export:
			with h5py.File('Phase.hdf5', 'w') as Ph:
				dset = Ph.create_dataset('default', data=Phase)
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
			print('IDs     =', I.shape)
			print('Space   =', Space.shape)
			print('UnitC   =', UnitC.shape)
			print('X Coord =', Coord.shape)
			print('Y Phase =', Phase.shape)
		else: return(Coord, UnitC, Space, Phase, I)

def main():
	if  args.Dataset:
		D = Dataset()
		D.run(IDs=sys.argv[2])
	elif args.Vectorise:
		Clas = sys.argv[2]
		File = sys.argv[3]
		size = int(sys.argv[4])
		numb = int(sys.argv[5]) + 1
		for samples in range(1, numb):
			Vectorise(filename=File, max_size=size, Type=Clas)
			if numb != 1:
				print('-------Sample {}/{}-------\n'.format(samples+1, numb))
				try:
					os.rename('Class.hdf5', 'Class_{}.hdf5'.format(samples+1))
				except:
					os.rename('Phase.hdf5', 'Phase_{}.hdf5'.format(samples+1))
				os.system('mv Space.hdf5 Space_{}.hdf5'.format(samples+1))
				os.system('mv UnitC.hdf5 UnitC_{}.hdf5'.format(samples+1))
				os.system('mv Coord.hdf5 Coord_{}.hdf5'.format(samples+1))
				if os.path.exists('IDs.hdf5'):
					os.system('mv IDs.hdf5 IDs_{}.hdf5'.format(samples+1))

if __name__ == '__main__': main()
