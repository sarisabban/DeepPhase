import os
import sys
import urllib.request

def setup():
	'''
	Installs required dependencies for this script to work
	https://github.com/cctbx/cctbx_project/
	'''
	os.system('sudo ln -s /usr/bin/python3 /usr/bin/python')
	os.system('sudo apt install libglu1-mesa-dev freeglut3-dev mesa-common-dev scons build-essential')
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
	os.sysrem('conda activate Cenv')
	os.sysrem('pip3 install tqdm biopython')

class ClassData():
	''' Build a dataset for protein classification from x-ray diffraction '''
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
		from iotbx.reflection_file_reader import any_reflection_file
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
				ms = ms_all.customized_copy(space_group_info=a.space_group_info())
				S = str(ms).split()[-1].split(')')[0]
				# F-obs
				F = list(a.expand_to_p1().f_sq_as_f().data())
				# Convert miller hkl to polar
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
				nX = []
				nY = []
				nZ = []
				nR = []
				nF = []
				for x, y, z, r, f in zip(X, Y, Z, R, F):
					if r >= 10.0 or r <= 2.5:
						continue
					else:
						nX.append(x)
						nY.append(y)
						nZ.append(z)
						nR.append(r)
						nF.append(f)
				X = nX
				Y = nY
				Z = nZ
				R = nR
				F = nF
				return(S, C, X, Y, Z, R, F)
	def labels(self, filename):
		import Bio.PDB
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
		import tqdm
		with open('temp', 'w') as temp:
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
						Mfilename = item + '.mtz'
						Pfilename = item + '.pdb'
						S, C, X, Y, Z, R, F = self.features(Mfilename)
						H_frac, S_frac, L_frac = self.labels(Pfilename)
						if H_frac >= 0.5 or S_frac == 0.0:
							label = 'Alpha'
						else:
							label = 'Not_Alpha'
						assert len(X) == len(Y) == len(Z) == len(R) == len(F),\
						'\u001b[31m[-] {} Failed: values not equal\u001b[0m'.format(item.upper())
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
						temp.write(item.upper()+'.mtz'+','+label+',')
						temp.write(example + '\n')
						size.append(len(X))
						os.remove(Mfilename)
						os.remove(Pfilename)
						#print('\u001b[32m[+] {} Done\u001b[0m'.format(item.upper()))
					except:
						print('\u001b[31m[-] {} Failed: problem compiling\u001b[0m'.format(item.upper()))
						continue
			header = ['PDB_ID,Label,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max(size)+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},F-obs_{}'\
				.format(i, i, i, i, i))
			header = ''.join(header)
		with open('temp2.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(header + '\n')
				for line in t: f.write(line)
		os.remove('temp')
		with open('temp2') as f:
			with open('DeepClass.csv'), 'a') as F:
				first_line = f.readline()
				F.write(first_line)
				size = len(first_line.strip().split(','))
				for line in tqdm.tqdm(f):
					line = line.strip().split(',')
					gap = size - len(line)
					for zero in range(gap):
						line.append('0')
					new_line = ','.join(line)
					F.write(new_line + '\n')
				os.remove('temp2')

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
				# F-obs
				F = list(a.expand_to_p1().f_sq_as_f().data())
		nX = []
		nY = []
		nZ = []
		nR = []
		nF = []
		nP = []
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
		import tqdm
		with open('temp', 'w') as temp:
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
						#print('\u001b[32m[+] {} Done\u001b[0m'.format(item.upper()))
					except:
						print('\u001b[31m[-] {} Failed: problem compiling\u001b[0m'.format(item.upper()))
						continue
			header = ['PDB_ID,Space_Group,Unit-Cell_a,Unit-Cell_b,Unit-Cell_c,Unit-Cell_Alpha,Unit-Cell_Beta,Unit-Cell_Gamma']
			for i in range(1, max(size)+1):
				header.append(',X_{},Y_{},Z_{},Resolution_{},F-obs_{},Phase{}'\
				.format(i, i, i, i, i, i))
			header = ''.join(header)
		with open('temp2.csv', 'w') as f:
			with open('temp', 'r') as t:
				f.write(header + '\n')
				for line in t: f.write(line)
		os.remove('temp')
		with open('temp2') as f:
			with open('DeepPhase.csv'), 'a') as F:
				first_line = f.readline()
				F.write(first_line)
				size = len(first_line.strip().split(','))
				for line in tqdm.tqdm(f):
					line = line.strip().split(',')
					gap = size - len(line)
					for zero in range(gap):
						line.append('0')
					new_line = ','.join(line)
					F.write(new_line + '\n')
				os.remove('temp2')

def main():
	if sys.argv[1] == 'class':
		Cls = ClassData()
		Cls.run(IDs='Class.txt')
	elif sys.argv[1] == 'phase':
		Phs = PhaseData()
		Phs.run(IDs='Phase.txt')
	elif sys.argv[1] == 'setup':
		setup()
	else:
		print('\u001b[31m[-] Wrong argument\u001b[0m')

if __name__ == '__main__': main()
