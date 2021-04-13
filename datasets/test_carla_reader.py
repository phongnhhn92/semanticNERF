import csv
import os

path = '/media/phong/Data2TB/dataset/carlaGVSNet/'

test_frames = []
with open(os.path.join(path, 'test_samples.txt'), 'r') as fid:
    reader = csv.reader(fid)
    for line in reader:
        src = os.path.join(path, line[0])
        trg = os.path.join(path, line[1])
        test_frames.append([src, trg])
print('a')