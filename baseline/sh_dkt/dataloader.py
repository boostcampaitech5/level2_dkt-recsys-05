import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        for col in cate_cols:
            # For UNKNOWN class
            a = df[col].unique().tolist() + [np.nan]

            le = LabelEncoder()
            le.fit(a)
            df[col] = le.transform(df[col])
            self.__save_labels(le, col)

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __preprocessing_test(self, df):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        for col in cate_cols:
            # For UNKNOWN class
            a = df[col].unique().tolist() + [np.nan]
            le = LabelEncoder()
            label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
            le.classes_ = np.load(label_path)

            le_classes_dict = {label: index for index, label in enumerate(le.classes_)}
            df[col] = df[col].apply(lambda x: le_classes_dict.get(str(x), np.nan))

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def load_data_from_file(self, file_name):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__preprocessing(df)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = df["assessmentItemID"].nunique()
        self.args.n_test = df["testId"].nunique()
        self.args.n_tag = df["KnowledgeTag"].nunique()

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_test_data_from_file(self, file_name):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__preprocessing_test(df)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = df["assessmentItemID"].nunique()
        self.args.n_test = df["testId"].nunique()
        self.args.n_tag = df["KnowledgeTag"].nunique()

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        return df

        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_test_data_from_file(file_name)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct = row[0], row[1], row[2], row[3]

        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[:seq_len] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            col_list[i].append(col)

    # 각 column의 값들을 대상으로 padding 진행
    # pad_sequence([[1, 2, 3], [3, 4]]) -> [[1, 2, 3],
    #                                       [3, 4, 0]]
    for i, col_batch in enumerate(col_list):
        col_list[i] = pad_sequence(col_batch, batch_first=True)

    # mask의 경우 max_seq_len을 기준으로 길이가 설정되어있다.
    # 만약 다른 column들의 seq_len이 max_seq_len보다 작다면
    # 이 길이에 맞추어 mask의 길이도 조절해준다
    col_seq_len = col_list[0].size(1)
    mask_seq_len = col_list[-1].size(1)
    if col_seq_len < mask_seq_len:
        col_list[-1] = col_list[-1][:, :col_seq_len]

    return tuple(col_list)


def get_loaders(args, train, valid):
    print(
        f"Type of args.batch_size: {type(args.batch_size)}"
    )  # args.batch_size의 데이터 타입 출력
    pin_memory = False
    trainset = DKTDataset(train, args)
    valset = DKTDataset(valid, args)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    valid_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    return train_loader, valid_loader
