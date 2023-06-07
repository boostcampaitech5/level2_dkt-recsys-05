import pandas as pd
import argparse

import argparse
import pandas as pd
import os


def update_prediction(file_directory, file_name):
    # csv 파일을 읽어옵니다.
    file_path = os.path.join(file_directory, file_name + ".csv")
    df = pd.read_csv(file_path)

    # prediction 값을 조건에 따라 변환합니다.
    df["prediction"] = df["prediction"].apply(
        lambda x: 1 if x >= 0.7 else (0 if x < 0.15 else x)
    )

    # 변환된 데이터를 다시 csv 파일로 저장합니다.
    updated_file_path = os.path.join(file_directory, "updated_" + file_name + ".csv")
    df.to_csv(updated_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="The name of the file to be processed",
    )
    parser.add_argument(
        "--file_directory",
        default="/opt/ml/input/code/sh_dkt/lv2_Sniper_DKT/outputs",
        type=str,
        required=False,
        help="The directory of the file to be processed",
    )

    args = parser.parse_args()

    update_prediction(args.file_directory, args.file_name)
