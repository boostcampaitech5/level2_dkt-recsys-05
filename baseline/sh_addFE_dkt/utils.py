import os
import random

import numpy as np
import torch
import pandas as pd
import hyperopt


def set_seeds(seed: int = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1

            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(
                        col[window_i * stride : window_i * stride + window_size]
                    )

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))

    return augmented_datas


def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data


# report에서 auc및 실행 시간 정보 얻기
def time_auc(report, n_epoch=10):
    total_time = 0
    for epoch in range(1, n_epoch + 1):
        result = report[str(epoch)]
        total_time += result["train_time"]
        total_time += result["valid_time"]

    return total_time, report["best_auc"], report["best_acc"]


def trials_to_df(trials, space, best):
    # 전체 결과
    rows = []
    keys = list(trials.trials[0]["misc"]["vals"].keys())

    # 전체 실험결과 저장
    for trial in trials:
        row = {}

        # tid
        tid = trial["tid"]
        row["experiment"] = str(tid)

        # hyperparameter 값 저장
        vals = trial["misc"]["vals"]
        hparam = {key: value[0] for key, value in vals.items()}

        # space가 1개 - 값을 바로 반환
        # space가 다수 - dict에 값을 반환
        hparam = hyperopt.space_eval(space, hparam)

        if len(keys) == 1:
            row[keys[0]] = hparam
        else:
            for key in keys:
                row[key] = hparam[key]

        # metric
        row["metric"] = abs(trial["result"]["loss"])

        # 소요 시간
        row["time"] = (trial["refresh_time"] - trial["book_time"]).total_seconds()

        rows.append(row)

    experiment_df = pd.DataFrame(rows)

    # best 실험
    row = {}
    best_hparam = hyperopt.space_eval(space, best)

    if len(keys) == 1:
        row[keys[0]] = best_hparam
    else:
        for key in keys:
            row[key] = best_hparam[key]
    row["experiment"] = "best"

    best_df = pd.DataFrame([row])

    # best 결과의 auc / time searching 하여 찾기
    search_df = pd.merge(best_df, experiment_df, on=keys)

    # column명 변경
    search_df = search_df.drop(columns=["experiment_y"])
    search_df = search_df.rename(columns={"experiment_x": "experiment"})

    # 가장 좋은 metric 결과 중 가장 짧은 시간을 가진 결과를 가져옴
    best_time = search_df.time.min()
    search_df = search_df.query("time == @best_time")

    df = pd.concat([experiment_df, search_df], axis=0)

    return df
