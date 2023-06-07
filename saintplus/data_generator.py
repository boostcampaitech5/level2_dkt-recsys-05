import gc
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

"""
Reference:
https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch
"""

class Riiid_Sequence(Dataset):
    def __init__(self, groups, seq_len):
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []

        for user_id in groups.index:
            category, category_test, category_test_problem, problem_tag, problem_time, break_time, answer = groups[user_id]
            if len(category) < 2:
                continue

            if len(category) > self.seq_len:
                initial = len(category) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (
                        category[:initial], category_test[:initial], category_test_problem[:initial], problem_tag[:initial], 
                        problem_time[:initial], break_time[:initial], answer[:initial]
                    )
                chunks = len(category)//self.seq_len
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.samples[f"{user_id}_{c+1}"] = (
                        category[start:end], category_test[start:end], category_test_problem[start:end], problem_tag[start:end], 
                        problem_time[start:end], break_time[start:end], answer[start:end]
                    )
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (category, category_test, category_test_problem, problem_tag, problem_time, break_time, answer)

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        category, category_test, category_test_problem, problem_tag, problem_time, break_time, answer = self.samples[user_id]
        seq_len = len(category)

        category_sample = np.zeros(self.seq_len, dtype=int)
        category_test_sample = np.zeros(self.seq_len, dtype=int)
        category_test_problem_sample = np.zeros(self.seq_len, dtype=int)
        problem_tag_sample = np.zeros(self.seq_len, dtype=int)
        problem_time_sample = np.zeros(self.seq_len, dtype=float)
        break_time_sample = np.zeros(self.seq_len, dtype=float)
        
        answer_sample = np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)
  
        if seq_len == self.seq_len:
            category_sample[:] = category
            category_test_sample[:] = category_test
            category_test_problem_sample[:] = category_test_problem
            problem_tag_sample[:] = problem_tag
            problem_time_sample[:] = problem_time
            break_time_sample[:] = break_time
            answer_sample[:] = answer
        else:
            category_sample[-seq_len:] = category
            category_test_sample[-seq_len:] = category_test
            category_test_problem_sample[-seq_len:] = category_test_problem
            problem_tag_sample[-seq_len:] = problem_tag
            problem_time_sample[-seq_len:] = problem_time
            break_time_sample[-seq_len:] = break_time
            answer_sample[-seq_len:] = answer
           
        category_sample = category_sample[1:]
        category_test_sample = category_test_sample[1:]
        category_test_problem_sample = category_test_problem_sample[1:]
        problem_tag_sample = problem_tag_sample[1:]
        problem_time_sample = problem_time_sample[1:]
        break_time_sample = break_time_sample[1:]
        label = answer_sample[1:]
        answer_sample = answer_sample[:-1]
        

        return category_sample, category_test_sample, category_test_problem_sample, problem_tag_sample, problem_time_sample, break_time_sample, answer_sample, label
    
class Riiid_Sequence_Test(Dataset):
    def __init__(self, groups, seq_len):
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []

        for user_id in groups.index:
            category, category_test, category_test_problem, problem_tag, problem_time, break_time, answer = groups[user_id]
            if len(category) < 2:
                continue

            if len(category) > self.seq_len:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (
                    category[-seq_len:], category_test[-seq_len:], category_test_problem[-seq_len:], problem_tag[-seq_len:], 
                    problem_time[-seq_len:], break_time[-seq_len:], answer[-seq_len:]
                )
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (category, category_test, category_test_problem, problem_tag, problem_time, break_time, answer)

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        category, category_test, category_test_problem, problem_tag, problem_time, break_time, answer = self.samples[user_id]
        seq_len = len(category)
        
        category_sample = np.zeros(self.seq_len, dtype=int)
        category_test_sample = np.zeros(self.seq_len, dtype=int)
        category_test_problem_sample = np.zeros(self.seq_len, dtype=int)
        problem_tag_sample = np.zeros(self.seq_len, dtype=int)
        problem_time_sample = np.zeros(self.seq_len, dtype=float)
        break_time_sample = np.zeros(self.seq_len, dtype=float)
        
        answer_sample = np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)
  
        if seq_len == self.seq_len:
            category_sample[:] = category
            category_test_sample[:] = category_test
            category_test_problem_sample[:] = category_test_problem
            problem_tag_sample[:] = problem_tag
            problem_time_sample[:] = problem_time
            break_time_sample[:] = break_time
            answer_sample[:] = answer
        else:
            category_sample[-seq_len:] = category
            category_test_sample[-seq_len:] = category_test
            category_test_problem_sample[-seq_len:] = category_test_problem
            problem_tag_sample[-seq_len:] = problem_tag
            problem_time_sample[-seq_len:] = problem_time
            break_time_sample[-seq_len:] = break_time
            answer_sample[-seq_len:] = answer
           
        category_sample = category_sample[1:]
        category_test_sample = category_test_sample[1:]
        category_test_problem_sample = category_test_problem_sample[1:]
        problem_tag_sample = problem_tag_sample[1:]
        problem_time_sample = problem_time_sample[1:]
        break_time_sample = break_time_sample[1:]
        label = answer_sample[1:]
        answer_sample = answer_sample[:-1]
        

        return category_sample, category_test_sample, category_test_problem_sample, problem_tag_sample, problem_time_sample, break_time_sample, answer_sample, label