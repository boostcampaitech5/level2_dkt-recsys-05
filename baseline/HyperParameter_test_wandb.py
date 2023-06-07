import os
import torch
import wandb

from sh_dkt.trainer import run
from sh_dkt.args import parse_args
from sh_dkt.dataloader import Preprocess
from sh_dkt.utils import get_logger, set_seeds, logging_conf

from datetime import datetime
import pickle

import warnings

warnings.filterwarnings("ignore")

logger = get_logger(logging_conf)


def main(args):
    prorject_name = "{}".format(args.model)  # Model Name에 따른 실험관리

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "auc"},
        "parameters": {
            "lr": {"distribution": "uniform", "min": 0.0001, "max": 0.03},
            "batch_size": {"values": [32, 64, 128, 256]},
            "n_layers": {"values": [1, 2, 3]},
            "drop_out": {"distribution": "uniform", "min": 0.07, "max": 0.3},
            "hidden_dim": {"values": [64, 128, 256, 512]},
            "max_seq_len": {"values": [5, 10, 30, 50, 100]},
        },
    }

    # 최적화 할 함수
    def objective_function():
        wandb.init()

        # 하이퍼파라메타 값 변경
        for k, v in wandb.config.items():
            vars(args)[k] = v
            if k == "hidden_dim" and args.model == "electra":
                vars(args)[
                    "embedding_size"
                ] = v  # Electra는 embedding_size라는 추가적인 요소가 필요하다.

        # seed 설정
        set_seeds(args.seed)

        report = run(args, train_data, valid_data)

        best_auc = report["best_auc"]

        wandb.log({"auc": best_auc})

    sweep_id = wandb.sweep(sweep_configuration, project=prorject_name)

    wandb.agent(sweep_id, function=objective_function, count=40)

    sweep_results = wandb.Api().sweep(
        wandb.api.viewer()["entity"] + "/" + prorject_name + "/" + sweep_id
    )

    runs = sweep_results.runs

    sorted_runs = sorted(runs, key=lambda run: run.summary["auc"], reverse=True)

    args_list = []
    for i in range(3):
        param_dict = dict(sorted_runs[i].config)
        param_dict.update({"auc": sorted_runs[i].summary["auc"]})
        args_list.append(param_dict)

    model_name = args.model
    best_auc = sorted_runs[0].summary["auc"]
    cur_time = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )  # 현재 시간을 YYYYMMDD_HHMMSS 형식으로 표시
    folder_path = f"/opt/ml/input/code/sh_dkt/lv2_Sniper_DKT/args_list/{model_name}_{best_auc}_{cur_time}_args_list"
    # 해당 경로에 폴더가 없다면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # args_list 저장
    with open(os.path.join(folder_path, "args_list.pickle"), "wb") as handle:
        pickle.dump(args_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Preparing train_data, valid_data....")
    preprocess = Preprocess(args)
    # TODO preprocess.load_train_data 속도 단축
    preprocess.load_train_data(file_name=args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(
        train_data, ratio=args.ratio, shuffle=True, seed=args.seed
    )
    logger.info("data prepared")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
