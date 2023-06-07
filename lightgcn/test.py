import torch

from lightgcn.datasets import load_data, separate_data
from lightgcn.datasets import indexing_data, process_data


# load_data: train / test concat & userid, assid만 가져옴
data = load_data("/opt/ml/input/data")
print(data[:5])

print("=" * 90)

# separate_data: train / test 분리
train, test = separate_data(data)
print(train[:5])
print(test[:5])

print("=" * 90)

id2index = indexing_data(data) # dict
print(len(id2index))
# 총 16896개의 node가 존재(user + assessment)

use_cuda: bool = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
train_proc = process_data(train, id2index, device)
test_proc = process_data(test, id2index, device)
# train: 총 2475962개의 edge(link)가 존재
# test: 총 744개의 edge(link)가 존재
print(test_proc)