import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="/opt/ml/input/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="outputs/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=20, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=128, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=1, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=4, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=64, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=0, type=int, help="clip grad")
    parser.add_argument("--patience", default=10, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )
    ### 중요 ###
    parser.add_argument("--model", default="bert", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )
    parser.add_argument("--window", default=False, type=bool, help="number of window")
    parser.add_argument("--stride", default=None, type=int, help="number of stride")
    parser.add_argument("--shuffle", default=False, type=bool, help="for shuffle")
    parser.add_argument("--shuffle_n", default=2, type=int, help="number of shuffle")
    parser.add_argument("--Tfixup", default=False, type=bool, help="Tfixup")
    parser.add_argument("--layer_norm", default=True, type=bool, help="layernorm")
    parser.add_argument("--ratio", default=0.7, type=float, help="split ratio")
    parser.add_argument(
        "--train_data_size", default=0.8, type=float, help="size for ensemble ratio"
    )
    parser.add_argument(
        "--embedding_size", default=None, type=int, help="embedding_size for electra"
    )
    parser.add_argument(
        "--meta_model",
        default="LinearRegression",
        type=str,
        help="type of ensemble meta_model",
    )

    # TODO : Jupyter 환경에서 실험을 할 때는 아래 주석 친 코드를 써야 합니다.
    # 실제로 터미널 창에서만 작업을 돌린다면 args = parser.parse_args() 만 남겨둬야합니다. """

    # Jupyter 환경에서 실행되는 경우에는 기본 값만 사용합니다.
    if "ipykernel" in sys.modules:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    # max_seq_len 값으로 stride의 기본값을 설정합니다.
    if args.stride is None:
        args.stride = args.max_seq_len

    return args
