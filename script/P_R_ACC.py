import os

positive_dir = 'FHDMi/positive'
negative_dir = 'FHDMi/negative'

TP = 0
FP = 0
TN = 0
FN = 0

for filename in os.listdir(positive_dir):
	if filename.startswith("src") or filename.endswith("_m.png"):
		TP += 1
	else:
		FP += 1

for filename in os.listdir(negative_dir):
	if filename.endswith("tar.jpg") or filename.endswith("_warp.png"):
		TN += 1
	else:
		FN += 1

print("P=",TP/(TP+FP))
print("R=",TP/(TP+FN))
print("ACC=",(TP+TN)/(TP+FP+TN+FN))
