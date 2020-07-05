from pathlib import Path
import random

import numpy as np
import cv2

# set param
root = Path.cwd()
data_dir = root / "data"
target_dir = data_dir / "train"
shape_tuple = (224, 224)
color_channel = 3
SUFFIX = [".png"]

# init
class_name = [x.name for x in target_dir.iterdir() if x.is_dir()]
class_indices = dict(zip(class_name, range(len(class_name))))
n_data = sum(1 for cl in class_name 
            for x in (target_dir / cl).iterdir() if x.suffix in SUFFIX)
x = np.zeros((n_data,)+shape_tuple+(color_channel,))
y = np.zeros((n_data,)+(1,))

# get path and shuffle
pathes = [(p, class_indices[cl]) for cl in class_name 
            for p in (target_dir / cl).iterdir() if p.suffix in SUFFIX]
random.shuffle(pathes)

for idx, p in enumerate(pathes):
    img = cv2.imread(str(p[0]))
    x[idx,] = cv2.resize(img, shape_tuple)
    y[idx] = p[1]

# save
np.save(str(root / "x"), x)
np.save(str(root / "y"), y)
