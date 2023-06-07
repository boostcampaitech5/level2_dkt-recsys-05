import random, os
from tqdm.notebook import tqdm
import pandas as pd

from sklearn import  preprocessing

import torch
from torch import nn, Tensor


from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing


def load_data(data_dir: str) -> pd.DataFrame: 
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    return data
    

def load_edge_csv(df, 
                  src_index_col, 
                  dst_index_col, 
                  link_index_col):
    """Loads csv containing edges between users and items

    Args:
        src_index_col (str): column name of users
        dst_index_col (str): column name of items
        link_index_col (str): column name of user item interaction

    Returns:
        list of list: edge_index -- 2 by N matrix containing the node ids of N user-item edges
        N here is the number of interactions
    """
    
    edge_index = None
    
    # Constructing COO format edge_index
    src = [user_id for user_id in  df[src_index_col]]    
    dst = [(movie_id) for movie_id in df[dst_index_col]]

    # set interaction when answerCode equals to 1
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) == 1

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
    return edge_index

# convert interaction matrix into adjacency matrix
def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index,
                                                   num_users,
                                                   num_items):
    R = torch.zeros((num_users, num_items))
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = 1

    R_transpose = torch.transpose(R, 0, 1)
    adj_mat = torch.zeros((num_users + num_items , num_users + num_items))
    adj_mat[: num_users, num_users :] = R.clone()
    adj_mat[num_users :, : num_users] = R_transpose.clone()
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()
    return adj_mat_coo

class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, 
                 num_items, 
                 embedding_dim=128, # define the embding vector length for each node
                 K=3, 
                 add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        # define user and item embedding for direct look up. 
        # embedding dimension: num_user/num_item x embedding_dim
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: Tensor):
        
        edge_index_norm = gcn_norm(edge_index=edge_index, 
                                   add_self_loops=self.add_self_loops)

        # concat the user_emb and item_emb as the layer0 embing matrix
        # size will be (n_users + n_items) x emb_vector_len.   e.g: 10334 x 64
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        
        embs = [emb_0] # save the layer0 emb to the embs list
        
        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0 

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for _ in range(self.K):
            # propagate: method provided from messagepassing module
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)
            

        # this is doing the formula8 in LightGCN paper  
            
        # the stacked embs is a list of embedding matrix at each layer
        #    it's of shape n_nodes x (n_layers + 1) x emb_vector_len. 
        #        e.g: torch.Size([10334, 4, 64])
        embs = torch.stack(embs, dim=1)
        
        # From LightGCn paper: "In our experiments, we find that setting α_k uniformly as 1/(K + 1)
        #    leads to good performance in general."
        emb_final = torch.mean(embs, dim=1) # E^K


        # splits into e_u^K and e_i^K
        # users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items]) 

        return emb_final


    # called when call propagate
    def message(self, x_j, norm):
        # x_j is of shape:  edge_index_len x emb_vector_len
        #    e.g: torch.Size([77728, 64]
        #
        # x_j is basically the embedding of all the neighbors based on the src_list in coo edge index
        # 
        # elementwise multiply by the symmetrically norm. So it's essentiall what formula 7 in LightGCN
        # paper does but here we are using edge_index rather than Adj Matrix
        return norm.view(-1, 1) * x_j


def get_embedding():
    data_dir = "/opt/ml/input/data"

    data = load_data(data_dir=data_dir)

    lbl_user = preprocessing.LabelEncoder()
    lbl_item = preprocessing.LabelEncoder()

    data.userID = lbl_user.fit_transform(data.userID.values)
    data.assessmentItemID = lbl_item.fit_transform(data.assessmentItemID.values)
    
    edge_index = load_edge_csv(data, src_index_col="userID", dst_index_col="assessmentItemID", link_index_col="answerCode")
    edge_index = torch.LongTensor(edge_index) 

    num_users = len(data['userID'].unique())
    num_items = len(data['assessmentItemID'].unique())
    
    edge_index = convert_r_mat_edge_index_to_adj_mat_edge_index(edge_index,
                                                                num_users,
                                                                num_items)
    
    layers = 3    
    model = LightGCN(num_users=num_users, 
                    num_items=num_items, 
                    K=layers)
    
    # users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(edge_index)
    embed_final = model.forward(edge_index)
    # print(f"user_emb_size: {users_emb_final.size()}")
    # print(f"user_emb_size: {items_emb_final.size()}")
    embed_final = embed_final.to("cuda")
    return embed_final, num_users

    
# if __name__=="__main__":

#     emb_matrix, num_users = get_embedding()
#     emb_matrix = emb_matrix.to("cuda")
#     # print(emb_matrix)
#     print(f"emb_matrix_size: {emb_matrix.size()}") # [64, 20, 128]