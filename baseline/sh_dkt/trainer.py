import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb
import copy
from datetime import datetime

from collections import OrderedDict
from .criterion import get_criterion
from .dataloader import get_loaders, collate, DKTDataset
from .metric import get_metric
from .model import LSTM, Bert, LastQuery, Saint, FixupEncoder, Electra, XLNet, Roberta
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf, data_augmentation
import gc
import time
from tqdm import notebook

logger = get_logger(logger_conf=logging_conf)


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """Saves checkpoint to a given directory."""
    save_path = os.path.join(model_dir, model_filename)
    # logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == "last_query":
        model = LastQuery(args)
    if args.model == "saint":
        model = Saint(args)
    if args.model == "tfixup":
        model = FixupEncoder(args)
    if args.model == "electra":
        model = Electra(args)
    if args.model == "xlnet":
        model = XLNet(args)
    if args.model == "roberta":
        model = Roberta(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    test, question, tag, correct, mask = batch

    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0  # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)

    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    return (test, question, tag, correct, mask, interaction, gather_index)


# loss계산하고 parameter update!
def compute_loss(preds, targets, index):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
        index    : (batch_size, max_seq_len)

        만약 전체 sequence 길이가 max_seq_len보다 작다면 해당 길이로 진행
    """
    loss = get_criterion(preds, targets)
    loss = torch.gather(loss, 1, index)
    loss = torch.mean(loss)

    return loss


def get_gradient(model):
    gradient = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad != None:
            gradient.append(grad.cpu().numpy().astype(np.float16))
            # gradient.append(grad.clone().detach())
        else:
            gradient.append(None)

    return gradient


def train(train_loader, model, optimizer, scheduler, args, gradient=False):
    model.train()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)

        preds = model(input)

        targets = input[3]  # correct

        index = input[-1]  # gather index

        # print(targets, targets.size, index, preds)
        loss = compute_loss(preds, targets, index)
        loss.backward()

        # save gradient distribution
        if gradient:
            args.n_iteration += 1
            args.gradient[f"iteration_{args.n_iteration}"] = get_gradient(model)

        # grad clip
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        optimizer.zero_grad()

        # warmup scheduler
        if args.scheduler == "linear_warmup":
            scheduler.step()

        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)

        if args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    # print(total_targets.shape, type(total_targets))
    auc, acc = get_metric(total_targets, total_preds)

    return auc, acc


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3]  # correct
        index = input[-1]  # gather index

        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)

        if args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    if not len(np.unique(total_targets)) == 1:
        # Train AUC / ACC
        auc, acc = get_metric(total_targets, total_preds)
        return auc, acc, total_preds, total_targets
    else:
        return None, None, total_preds, None


def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model


def run(args, train_data, valid_data, gradient=False):
    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    # augmentation
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(
            f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n"
        )

    train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    logger.info("Building Model ...")
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # 🌟 분석에 사용할 값 저장 🌟
    report = OrderedDict()

    # gradient step 분석에 사용할 변수
    if gradient:
        args.n_iteration = 0
        args.gradient = OrderedDict()

        # 모델의 gradient값을 가리키는 모델 명 저장
        args.gradient["name"] = [name for name, _ in model.named_parameters()]

    best_auc = -1
    best_auc_epoch = -1
    best_acc = -1
    best_acc_epoch = -1
    best_model = -1
    for epoch in notebook.tqdm(range(args.n_epochs)):
        epoch_report = {}

        ### TRAIN
        train_start_time = time.time()
        train_auc, train_acc = train(
            train_loader, model, optimizer, scheduler, args, gradient
        )
        train_time = time.time() - train_start_time

        epoch_report["train_auc"] = train_auc
        epoch_report["train_acc"] = train_acc
        epoch_report["train_time"] = train_time

        ### VALID
        valid_start_time = time.time()
        valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)
        valid_time = time.time() - valid_start_time

        epoch_report["valid_auc"] = valid_auc
        epoch_report["valid_acc"] = valid_acc
        epoch_report["valid_time"] = valid_time

        # save lr
        epoch_report["lr"] = optimizer.param_groups[0]["lr"]

        # 🌟 save it to report 🌟
        report[f"{epoch + 1}"] = epoch_report

        wandb.log(
            dict(
                epoch=epoch,
                train_auc_epoch=train_auc,
                train_acc_epoch=train_acc,
                valid_auc_epoch=valid_auc,
                valid_acc_epoch=valid_acc,
                train_time=train_time,
                valid_time=valid_time,
            )
        )

        ### TODO: model save or early stopping
        if valid_auc > best_auc:
            best_auc = valid_auc
            logger.info("best auc {}".format(best_auc))
            early_stopping_counter = 0  # 추가
            best_auc_epoch = epoch + 1

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter,
                    args.patience,
                )
                break

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_acc_epoch = epoch + 1

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)

    # save best records
    report["best_auc"] = best_auc
    report["best_auc_epoch"] = best_auc_epoch
    report["best_acc"] = best_acc
    report["best_acc_epoch"] = best_acc_epoch

    # save gradient informations
    if gradient:
        report["gradient"] = args.gradient
        del args.gradient
        del args["gradient"]

    return report


class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_data, valid_data, gradient=False):
        """훈련을 마친 모델을 반환한다"""

        # args update
        self.args = args

        # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
        torch.cuda.empty_cache()
        gc.collect()

        # augmentation
        augmented_train_data = data_augmentation(train_data, args)
        if len(augmented_train_data) != len(train_data):
            print(
                f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n"
            )
        train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)
        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (
            args.n_epochs
        )
        args.warmup_steps = args.total_steps // 10

        model = get_model(args)

        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        """# 🌟 분석에 사용할 값 저장 🌟
        report = OrderedDict()

        # gradient step 분석에 사용할 변수
        if gradient:
            args.n_iteration = 0
            args.gradient = OrderedDict()

            # 모델의 gradient값을 가리키는 모델 명 저장
            args.gradient["name"] = [name for name, _ in model.named_parameters()]"""

        best_auc = -1
        best_auc_epoch = -1
        best_acc = -1
        best_acc_epoch = -1
        best_model = -1
        for epoch in notebook.tqdm(range(args.n_epochs)):
            # epoch_report = {}

            ### TRAIN
            train_start_time = time.time()
            train_auc, train_acc = train(
                train_loader, model, optimizer, scheduler, args, gradient
            )
            train_time = time.time() - train_start_time

            """epoch_report["train_auc"] = train_auc
            epoch_report["train_acc"] = train_acc
            epoch_report["train_time"] = train_time"""

            ### VALID
            valid_start_time = time.time()
            valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)
            valid_time = time.time() - valid_start_time

            """epoch_report["valid_auc"] = valid_auc
            epoch_report["valid_acc"] = valid_acc
            epoch_report["valid_time"] = valid_time

            # save lr
            epoch_report["lr"] = optimizer.param_groups[0]["lr"]

            # 🌟 save it to report 🌟
            report[f"{epoch + 1}"] = epoch_report"""

            wandb.log(
                dict(
                    epoch=epoch,
                    train_auc_epoch=train_auc,
                    train_acc_epoch=train_acc,
                    valid_auc_epoch=valid_auc,
                    valid_acc_epoch=valid_acc,
                    train_time=train_time,
                    valid_time=valid_time,
                )
            )

            ### TODO: model save or early stopping
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_model = copy.deepcopy(model)
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                    },
                    model_dir=args.model_dir,
                    model_filename="best_model.pt",
                )
                early_stopping_counter = 0
                best_auc_epoch = epoch + 1

            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                        "EarlyStopping counter: %s out of %s",
                        early_stopping_counter,
                        args.patience,
                    )
                    break

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)
            else:
                scheduler.step()

        """# save best records
        report["best_auc"] = best_auc
        report["best_auc_epoch"] = best_auc_epoch
        report["best_acc"] = best_acc
        report["best_acc_epoch"] = best_acc_epoch

        # save gradient informations
        if gradient:
            report["gradient"] = args.gradient
            del args.gradient
            del args["gradient"]"""

        return best_model

    def evaluate(self, args, model, valid_data):
        """훈련된 모델과 validation 데이터셋을 제공하면 predict 반환"""
        pin_memory = False

        valset = DKTDataset(valid_data, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

        auc, acc, preds, _ = validate(valid_loader, model, args)
        print(f"AUC : {auc}, ACC : {acc}")

        return preds

    def test(self, args, model, test_data):
        total_preds = self.evaluate(args, model, test_data)

        # 현재 시간을 문자열로 변환
        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        write_path = os.path.join(args.output_dir, f"submission_{cur_time}.csv")
        os.makedirs(name=args.output_dir, exist_ok=True)
        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write("{},{}\n".format(id, p))

        logger.info("Successfully saved submission as %s", write_path)

        return total_preds

    def get_target(self, datas):
        targets = []
        for data in datas:
            targets.append(data[-1][-1])

        return np.array(targets)


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class Stacking:
    def __init__(self, args, trainer):
        self.trainer = trainer
        self.args = args

    def get_train_oof(self, args, data, fold_n=5, stratify=True):
        oof = np.zeros(data.shape[0])

        fold_models = []

        if stratify:
            kfold = StratifiedKFold(n_splits=fold_n)
        else:
            kfold = KFold(n_splits=fold_n)

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        target = self.trainer.get_target(data)
        for i, (train_index, valid_index) in enumerate(kfold.split(data, target)):
            train_data, valid_data = data[train_index], data[valid_index]

            # 모델 생성 및 훈련
            print(f"Calculating train oof {i + 1}")
            # 테스트를 위한 확인 코드
            trained_model = self.trainer.train(args, train_data, valid_data)

            # 모델 검증
            predict = self.trainer.evaluate(args, trained_model, valid_data)

            # fold별 oof 값 모으기
            oof[valid_index] = predict
            fold_models.append(trained_model)

        return oof, fold_models

    def get_test_avg(self, args, models, test_data):
        predicts = np.zeros(test_data.shape[0])

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        for i, model in enumerate(models):
            print(f"Calculating test avg {i + 1}")
            predict = self.trainer.test(args, model, test_data)

            # fold별 prediction 값 모으기
            predicts += predict

        # prediction들의 average 계산
        predict_avg = predicts / len(models)

        return predict_avg

    def train_oof_stacking(self, args_list, data, fold_n=5, stratify=True):
        S_train = None
        models_list = []

        for i, args in enumerate(args_list):
            print(f"training oof stacking model [ {i + 1} ]")
            train_oof, models = self.get_train_oof(
                args, data, fold_n=fold_n, stratify=stratify
            )
            train_oof = train_oof.reshape(-1, 1)

            # oof stack!
            if not isinstance(S_train, np.ndarray):
                S_train = train_oof
            else:
                S_train = np.concatenate([S_train, train_oof], axis=1)

            # store fold models
            models_list.append(models)

        return models_list, S_train

    def test_avg_stacking(self, args, models_list, test_data):
        S_test = None
        for i, models in enumerate(models_list):
            print(f"test average stacking model [ {i + 1} ]")
            test_avg = self.get_test_avg(args, models, test_data)
            test_avg = test_avg.reshape(-1, 1)

            # avg stack!
            if not isinstance(S_test, np.ndarray):
                S_test = test_avg
            else:
                S_test = np.concatenate([S_test, test_avg], axis=1)

        return S_test

    def train(self, meta_model, args_list, data):
        models_list, S_train = self.train_oof_stacking(args_list, data)
        target = self.trainer.get_target(data)
        meta_model.fit(S_train, target)

        return meta_model, models_list, S_train, target

    def test(self, meta_model, models_list, test_data):
        S_test = self.test_avg_stacking(self.args, models_list, test_data)
        predict = meta_model.predict(S_test)

        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        write_path = os.path.join(
            self.args.output_dir, "submission_ensembled_{}.csv".format(cur_time)
        )
        os.makedirs(name=self.args.output_dir, exist_ok=True)
        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(predict):
                w.write("{},{}\n".format(id, p))

        return predict, S_test
