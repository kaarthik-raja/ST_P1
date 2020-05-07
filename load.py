import os
import re
import shutil



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


files = {}
def alldata():
	users = os.listdir(TF_dir)
	for user in users:
		files[user]={}
		mfccs = os.listdir(TF_dir+user)
		for mfcc in mfccs:
			with open(TF_dir+user+'/'+mfcc,"r") as mf:
				if mf.readline().strip().split(' ')[0] == '38':
					files[user][mfcc] = mf.readlines()
				else:
					print('error')
					input("continue ")
		print("done for ",user)
	return files


import sys
from numbers import Number
from collections import Set, Mapping, deque

try: # Python 
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'

def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)