import os
import argparse
import sys

import numpy as np
import torch
import wandb
import copy

from sh_addFE_dkt.trainer import run
from sh_addFE_dkt.trainer import Trainer, Stacking
from sh_addFE_dkt.args import parse_args
from sh_addFE_dkt.dataloader import Preprocess
from sh_addFE_dkt.utils import get_logger, set_seeds, logging_conf, time_auc
from sh_addFE_dkt.metric import get_metric

import os
import sys
import gc
import re

import random
import easydict
import tarfile

from tqdm import notebook
from collections import OrderedDict

import time
import datetime
from datetime import datetime

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

import scipy.stats

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

import warnings
import pickle

warnings.filterwarnings("ignore")

logger = get_logger(logging_conf)


# 로깅되는 부분도 추가하면 좋을 것 같다.
def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Preparing train data....")
    preprocess = Preprocess(args)
    # TODO preprocess.load_train_data 속도 단축
    preprocess.load_train_data(file_name=args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(
        train_data, ratio=args.ratio, shuffle=True, seed=args.seed
    )
    logger.info("Preparing train data for ensemble....")
    # 기타 모델 성능 개선 전용으로 사용할 데이터
    size = int(len(train_data) * args.train_data_size)
    data = train_data
    temp_train_data = train_data[:size]
    temp_valid_data = train_data[size:]
    logger.info("Wandb Init")
    wandb.init(project="dkt", config=vars(args))
    # TODO: Ensemble을 시킬 때, args_list를 빠르게 만들어 오는 법은? : hyperparam tuning에서 최고의 args를 저장하는 방식으로 해야겠다.

    """# n_heads가 다른 5개의 args
    args_list = []
    n_batch_size = [32, 64, 128]

    for batch_size in n_batch_size:
        copy_args = copy.deepcopy(args)
        copy_args.__dict__['batch_size'] = batch_size
        args_list.append(copy_args)"""

    """# 경로 설정
    folder_path = input("type the directory: ")  #

    # args_list 불러오기
    with open(os.path.join(folder_path, "args_list.pickle"), "rb") as handle:
        temp_args_list = pickle.load(handle)

    # 첫번째 요소의 키를 이용해 결과 딕셔너리의 키를 초기화
    temp_args_dict = {key: [] for key in temp_args_list[0].keys()}

    # 각 딕셔너리를 순회하며 값을 추가
    for dict_item in temp_args_list:
        for key, value in dict_item.items():
            temp_args_dict[key].append(value)
    args_list = []

    N = len(temp_args_list)
    for i in range(N):
        for k in temp_args_dict.keys():
            copy_args = copy.deepcopy(args)
            copy_args.__dict__[k] = temp_args_dict[k][i]
        args_list.append(copy_args)"""

    # 일시적으로 사용하는 Ensemble 파일
    def load_args(file_path):
        # Open the file in read mode
        with open(file_path, "rb") as handle:
            # Load the data from the file
            args_list = pickle.load(handle)

        return args_list

    # Use the function
    file_path = "/opt/ml/input/code/sh_dkt/lv2_Sniper_DKT/args_list/2023_05_24_MyModel-183758/args_list.pickle"
    temp_args_list = load_args(file_path)
    args_list = []

    for temp_args in temp_args_list:
        copy_args = copy.deepcopy(args)
        for k, v in vars(
            temp_args
        ).items():  # Enumerate the attributes of the temp_args Namespace object
            copy_args.__dict__[k] = v
        args_list.append(copy_args)
    # 여기까지

    # TODO : Stacking을 새로 만드는 것
    # TODO : meta_model HyperParameter 조절에 관련된 부분.
    # oof stacking ensemble

    # TRAIN
    logger.info("Training Stacked Model")
    stacking = Stacking(args, Trainer())
    if args.meta_model == "GradientBoost":
        meta_model = GradientBoostingRegressor(
            random_state=args.seed
        )  # 기존에는 LinearRegression 이었다.
    elif args.meta_model == "LinearRegression":
        meta_model = LinearRegression()

    meta_model, models_list, S_train, target = stacking.train(
        meta_model, args_list, data
    )

    # TEST
    logger.info("Preparing TestDataset....")
    preprocess = Preprocess(args)
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data = preprocess.get_test_data()
    # 실제로 test data에는 target값이 들어가지만 임의의 target값이 배정된다

    # Test
    logger.info("Predicting using trained meta_model and models_list")
    stacking = Stacking(args, Trainer())
    test_predict, S_test = stacking.test(meta_model, models_list, test_data)

    logger.info("DONE!")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
