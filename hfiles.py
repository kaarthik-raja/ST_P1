import os
import re
import shutil
import numpy as np
BASE_DIR = "/home/raja/Study/SEMVIII_Y4/Speech_tech/"
save_dir = BASE_DIR+"MyP1Dt/"
BASE_DIR = BASE_DIR + "TimitDataset/"
FeatureExtDir = "/home/raja/Study/SEMVIII_Y4/Speech_tech/P1/FeatureExtractionSample/"

ExeFile = FeatureExtDir + "ComputeFeatures"
ConfigF = FeatureExtDir + "mfcc.config"
xtractComm = ExeFile +" " + ConfigF + "  {} frameCepstrum+frameDeltaCepstrum {} 0.06 A"

timit_dir  = BASE_DIR +  "timit/timit_train/"
ntimit_dir = BASE_DIR + "ntimit/ntimit_train/"
team_map   = BASE_DIR + "speaker_mapping/team_16.txt"


TS_dir  = save_dir + 'timitS/'
TT_dir  = save_dir + 'timitT/'
TF_dir  = save_dir + 'timitF/'


NtS_dir  = save_dir + 'ntimitS/'
NtT_dir  = save_dir + 'ntimitT/'
NtF_dir  = save_dir + 'ntimitF/'

def main():
	global save_dir

	# Get_Teams()
	Consolidate()


def Make_Dirs(fn=''):
	""" S - Sound '.wav' file 
		T - Text for the sound
		F - MFCC features of that sound
	"""
	global save_dir,team_map
	
	os.mkdir( TS_dir+fn)
	os.mkdir( TT_dir+fn)
	os.mkdir( TF_dir+fn)

	os.mkdir( NtS_dir+fn)
	os.mkdir( NtT_dir+fn)
	os.mkdir( NtF_dir+fn)

def Get_Teams():
	global timit_dir,ntimit_dir,team_map,save_dir
	if not os.path.exists(save_dir):
		os.mkdir( save_dir)
	Make_Dirs()
	folders = open(team_map).readlines()
	folders  = [re.sub(r"^\W+",'',f).strip() for f in folders]
	allspeakers = os.listdir(timit_dir)
	for f in allspeakers:
		# if f not in folders:
			# continue
		Make_Dirs(f)
		#Timit
		src = timit_dir+f+'/'

		files = os.listdir(src)
		for fl in files:
			if fl.endswith(".wav"):
				shutil.copy2(src+fl,TS_dir+f+'/'+fl)  
				os.system(xtractComm.format(src+fl,TF_dir+f+'/'+fl[:-3]+'mfcc'))

			if fl.endswith(".txt"):
				shutil.copy2(src+fl,TT_dir+f+'/'+fl)  

		src = ntimit_dir+f+'/'

		files = os.listdir(src)
		for fl in files:
			if fl.endswith(".wav"):
				shutil.copy2(src+fl,NtS_dir+f+'/'+fl)  
				os.system(xtractComm.format(src+fl,NtF_dir+f+'/'+fl[:-3]+'mfcc'))
			if fl.endswith(".txt"):
				shutil.copy2(src+fl,NtT_dir+f+'/'+fl)  
		print("Finished for ",f)
def Consolidate(from_dir=TF_dir,to_dir='speakers/'):
	from io import StringIO as SIO
	if not os.path.exists(save_dir+to_dir):
		os.mkdir(save_dir+to_dir)
	users = os.listdir(from_dir)
	files={}
	for user in users:
		files[user]={}
		mfccs = os.listdir(from_dir+user)
		user_fvec = np.zeros((0,38))
		for mfcc in mfccs:
			with open(from_dir+user+'/'+mfcc,"r") as mf:
				if mf.readline().strip().split(' ')[0] == '38':
					feat_str = files[user][mfcc] = mf.readlines()
					fvec = np.loadtxt(SIO(" ".join(feat_str)))
					user_fvec = np.concatenate((user_fvec,fvec),axis=0) 
				else:
					print('error')
					input("continue ")
		np.save(save_dir+to_dir+user,user_fvec)
		print("done for ",user)

if __name__ == '__main__':
	main()