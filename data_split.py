import os
import pickle
import pandas as pd

df = pd.read_csv("/home/pwt/PosMIL/datasets/camelyon16_resnet50_10fold.csv")
train_ids = set()
for i, row in df.iterrows():
    if row[2] not in [0, 1]:
        train_ids.add(row[0])
print(len(train_ids))
train = None
with open("/home/data/pwt/wsi/camelyon16/dtfdmil_data/mDATA_train.pkl", "rb") as f:
    train = pickle.load(f)

train_ = {}
val_ = {}
for k in train.keys():
    if k in train_ids:
        train_[k] = train[k]
    else:
        val_[k] = train[k]

with open("/home/data/pwt/wsi/camelyon16/dtfdmil_data/mDATA_train_train1.pkl", "wb") as f:
    pickle.dump(train_, f)

with open("/home/data/pwt/wsi/camelyon16/dtfdmil_data/mDATA_train_val1.pkl", "wb") as f:
    pickle.dump(val_, f)


# CUDA_VISIBLE_DEVICES=3 python Main_DTFD_MIL.py --mDATA0_dir_train0=/home/pwt/PosMIL/data/camelyon16/dtfdmil_data/mDATA_train_train1.pkl --mDATA0_dir_val0=/home/pwt/PosMIL/data/camelyon16/dtfdmil_data/mDATA_train_val1.pkl --mDATA_dir_test0=/home/pwt/PosMIL/data/camelyon16/dtfdmil_data/mDATA_test.pkl --log_dir=/home/pwt/PosMIL/data/posmil_logs/dtfdmil_original_1