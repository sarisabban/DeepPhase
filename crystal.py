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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

parser = argparse.ArgumentParser(description='Compiling and vectorising X-ray crystal diffraction datasets')
parser.add_argument('-D', '--Dataset'  , nargs='+', help='Compile a datset of protein reflections points, include a text file of PDB IDs')
parser.add_argument('-V', '--Vectorise', nargs='+', help='Vectorise the datset only')
parser.add_argument('-S', '--Serialise', nargs='+', help='Vectorise and serialise the datset')
parser.add_argument('-A', '--Augment',   nargs='+', help='Augment a .pdb file to different orientations and generate reflection data')
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
			h1 = 'PDB_ID,Class,Space_Group,'
			h2 = 'Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,'
			h3 = 'Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma'
			head = [h1 + h2 + h3]
			for i in range(1, max(size)+1):
				head.append(',X_{},Y_{},Z_{},Resolution_{},E-value_{},Phase_{}'\
				.format(i, i, i, i, i, i))
			head = ''.join(head)
		with open('CrystalDataset.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(head + '\n')
				for line in t: f.write(line)
		os.remove('temp')

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
			# 3. Random sampling of points
			NC = [(x,y,z,r,e,p) for x,y,z,r,e,p in zip(
			T[0::6], T[1::6], T[2::6], T[3::6], T[4::6], T[5::6])]
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
	P   = np.array(P)
	if Type == 'deepclass' or Type == 'DeepClass':
		# 6. One-Hot encoding and normalisation
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
		# 7. Construct tensors
		Space = S
		UnitC = np.concatenate([UCe, UCa], axis=1)
		Coord = np.array([X, Y, Z, R, E])
		Coord = np.swapaxes(Coord, 0, 2)
		Coord = np.swapaxes(Coord, 0, 1)
		S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
		# 8. Shuffle examples
		Coord, Class, UnitC, Space, I = shuffle(Coord, Class, UnitC, Space, I)
		print('X Coord =', Coord.shape)
		print('Y Class =', Class.shape)
		print('Space   =', Space.shape)
		print('UnitC   =', UnitC.shape)
		print('IDs     =', I.shape)
		return(Coord, Class, Space, UnitC, I)
	elif Type == 'deepphase' or Type == 'DeepPhase':
		# 6. One-Hot encoding and normalisation
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
		# 7. Construct tensors
		Space = S
		UnitC = np.concatenate([UCe, UCa], axis=1)
		Coord = np.array([X, Y, Z, R, E])
		Coord = np.swapaxes(Coord, 0, 2)
		Coord = np.swapaxes(Coord, 0, 1)
		S, UCe, UCa, X, Y, Z, R, E, P = [], [], [], [], [], [], [], [], []
		Coord, Phase, UnitC, Space, I = shuffle(Coord, Phase, UnitC, Space, I)
		print('X Coord =', Coord.shape)
		print('Y Phase =', Phase.shape)
		print('Space   =', Space.shape)
		print('UnitC   =', UnitC.shape)
		print('IDs     =', I.shape)
		return(Coord, Phase, Space, UnitC, I)

class Synthetic():
	def __init__(self, filename='alpha.pdb', Label='Helix', d=2.5, n=3):
		self.filename = filename
		self.Label = Label
		self.n = n
		self.d = d
	def augment(self, in_name):
		''' Augment a .pdb file '''
		import pymol
		pymol.cmd.load(in_name)
		name = in_name[:-4]
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
		pymol.cmd.rotate([0, 0, 1], -zr, name)
		pymol.cmd.rotate([0, 1, 0], -yr, name)
		pymol.cmd.rotate([1, 0, 0], -xr, name)
		pymol.cmd.translate([-x, -y, -z], name)
		with open(in_name, 'r') as f:
			line1 = f.readline()
			line2 = f.readline()
		with open('temp.pdb', 'r') as t:
			aug = t.readlines()
		augmented = line1+line2
		for i in aug: augmented += i
		os.remove('temp.pdb')
		print('X: {:3} Y: {:3} Z: {:3} Xr: {:3} Yr: {:3} Zr: {:3}'\
		.format(x, y, z, xr, yr, zr))
		return(augmented)
	def reflections(self, pdbstr='x', export_mtz=False):
		''' Generate reflection data from a .pdb file '''
		import iotbx.pdb
		xrs = iotbx.pdb.input(source_info=None, lines=pdbstr)\
		.xray_structure_simple()
		a = xrs.structure_factors(d_min=self.d).f_calc()
		UC = a.unit_cell()
		C = str(UC)[1:-1]
		C = tuple(map(str, C.split(', ')))
		S = str(a.customized_copy()).split()[-1].split(')')[0]
		P1 = a.expand_to_p1().indices()
		R = [round(x, 5) for x in list(UC.d(P1))]
		P = [round(x, 5) for x in list(a.phases().data())]
		amp = a.amplitudes()
		amp.setup_binner(auto_binning=True)
		amp.binner()
		e_val = amp.quasi_normalize_structure_factors()
		E = list(e_val.expand_to_p1().f_sq_as_f().data())
		E = [round(x, 5) for x in E]
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
			mtz_object.write(file_name='{}.mtz'.format(filename[:-4]))
		print(len(X))
		return(S, C, X, Y, Z, R, E, P)
	def generate(self):
		''' Generate synthetic reflection data for n orientations of a .pdb '''
		size = []
		with open('temp', 'w') as f:
			for i in range(1, self.n+1):
				pdb_str = self.augment(self.filename)
				S, C, X, Y, Z, R, E, P = self.reflections(pdb_str)
				size.append(len(X))
				exp = [str(i)]
				exp.append(self.Label)
				exp.append(S)
				exp.append(C[0])
				exp.append(C[1])
				exp.append(C[2])
				exp.append(C[3])
				exp.append(C[4])
				exp.append(C[5])
				for x, y, z, r, e, p in zip(X, Y, Z, R, E, P):
					exp.append(str(x))
					exp.append(str(y))
					exp.append(str(z))
					exp.append(str(r))
					exp.append(str(e))
					exp.append(str(p))
				line = ','.join(exp)
				f.write(line+'\n')
		h1 = 'Augment,Class,Space_Group,'
		h2 = 'Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,'
		h3 = 'Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma'
		head = [h1 + h2 + h3]
		for i in range(1, max(size)+1):
			head.append(',X_{},Y_{},Z_{},Resolution_{},E-value_{},Phase_{}'\
			.format(i, i, i, i, i, i))
		head = ''.join(head)
		with open('{}.csv'.format(self.Label), 'w') as F:
			with open('temp', 'r') as f:
				F.write(head+'\n')
				for line in f:
					F.write(line)
		os.remove('temp')

def main():
	if  args.Dataset:
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
		#with h5py.File('S.h5','w') as s:dset=s.create_dataset('default',data=S)
		#with h5py.File('U.h5','w') as u:dset=u.create_dataset('default',data=U)
		#with h5py.File('I.h5','w') as i:dset=i.create_dataset('default',data=I)
	elif args.Augment:
		S = Synthetic(filename=sys.argv[2], Label=sys.argv[3], n=sys.argv[4])
		S.generate()

if __name__ == '__main__': main()
