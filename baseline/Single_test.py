import os
import argparse
import sys

import numpy as np
import torch
import wandb
import copy

from sh_dkt.trainer import run
from sh_dkt.trainer import Trainer, Stacking
from sh_dkt.args import parse_args
from sh_dkt.dataloader import Preprocess
from sh_dkt.utils import get_logger, set_seeds, logging_conf, time_auc
from sh_dkt.metric import get_metric

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

import warnings

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
    train_data, valid_data = preprocess.split_data(train_data)

    # Wandb init
    wandb.init(project="dkt", config=vars(args), reinit=True)

    # Training
    logger.info("Training....")
    trainer = Trainer()
    model = trainer.train(args, train_data, valid_data)
    valid_target = trainer.get_target(valid_data)
    valid_predict = trainer.evaluate(args, model, valid_data)
    logger.info("Training Done!")

    # Validation
    valid_auc, valid_acc = get_metric(valid_target, valid_predict)
    logger.info("Valid_auc : {}, Valid_acc: {}".format(valid_auc, valid_acc))

    # TODO 날짜별로 구분해서 submission 파일 만들까?
    # submission.csv 만들기

    # Test
    logger.info("Preparing test data....")
    preprocess = Preprocess(args)
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data = preprocess.get_test_data()
    test_predict = trainer.test(args, model, test_data)
    logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
