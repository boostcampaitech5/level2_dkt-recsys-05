import os
import random

import numpy as np
import torch

from torch import Tensor
from typing import Optional, Tuple


try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch import Tensor

from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor
from torch_geometric.utils.degree import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sort_edge_index import sort_edge_index
from torch_geometric.utils.subgraph import subgraph

class process:
    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.logger.info(f"{self.name} - Started")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(f"{self.name} - Complete")


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


def dropout_edge(
    edge_index: Tensor,
    p: float = 0.5,
    force_undirected: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:

    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def dropout_path(edge_index: Tensor, p: float = 0.2, walks_per_node: int = 1,
                 walk_length: int = 3, num_nodes: Optional[int] = None,
                 is_sorted: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    

    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_orders = None
    ori_edge_index = edge_index
    if not is_sorted:
        edge_orders = torch.arange(num_edges, device=edge_index.device)
        edge_index, edge_orders = sort_edge_index(edge_index, edge_orders,
                                                  num_nodes=num_nodes)

    row, col = edge_index
    sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
    start = row[sample_mask].repeat(walks_per_node)

    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges

    if edge_orders is not None:
        # permute edge ids
        e_id = edge_orders[e_id]
    edge_mask[e_id] = False
    edge_index = ori_edge_index[:, edge_mask]

    return edge_index, edge_mask

# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html
