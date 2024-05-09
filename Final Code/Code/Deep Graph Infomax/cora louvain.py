import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import networkx as nx
import community
import matplotlib.pyplot as plt
from models import DGI
from utils import process

# Step 1: Load Data and Preprocess
dataset = 'cora'
adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
features, _ = process.preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

# Convert data to PyTorch tensors
features = torch.FloatTensor(features[np.newaxis])
adj = process.sparse_mx_to_torch_sparse_tensor(adj)
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)

# Step 2: Initialize and Train DGI Model
model = DGI(n_in=ft_size, n_h=512, activation='prelu', hid_units=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()

b_xent = nn.BCEWithLogitsLoss()

best_loss = float('inf')
patience = 20
cnt_wait = 0

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(1, nb_nodes)
    lbl_2 = torch.zeros(1, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, adj, True, None, None, None) 

    loss = b_xent(logits, lbl)

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))

    if loss < best_loss:
        best_loss = loss
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimizer.step()

# Step 3: Load the Best Model
model.load_state_dict(torch.load('best_dgi.pkl'))

# Step 4: Extract Node Embeddings
embeds, _ = model.embed(features, adj, True, None)
embeds = embeds.squeeze(0)

# Step 5: Convert Embeddings to Numpy Array
embeds_np = embeds.cpu().detach().numpy()

# Step 6: Convert Embeddings to NetworkX Graph
G = nx.Graph(adj.to_dense().cpu().detach().numpy())
for node, emb in enumerate(embeds_np):
    G.nodes[node]['embedding'] = emb

# Step 7: Community Detection using Louvain Algorithm
partition = community.best_partition(G)

# Step 8: Print Communities
for com in set(partition.values()):
    members = [node for node, value in partition.items() if value == com]
    print("Community", com, ":", members)

# Step 9: Assign colors to nodes based on community
colors = [partition[node] for node in G.nodes()]

# Step 10: Visualize the Graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
nx.draw(G, pos, node_color=colors, cmap=plt.cm.tab10, with_labels=False)
plt.title("Community Detection using Louvain Algorithm")
plt.show()

# Step 11: Calculate Modularity
modularity = community.modularity(partition, G)
print("Modularity of the Network:", modularity)

# Step 12: Calculate Overall Density
overall_density = nx.density(G)
print("Overall Density of the Network:", overall_density)

# Step 13: Calculate Overall Conductance
overall_conductance = 0.0
for node in G.nodes():
    S = set(partition.keys()) - {node}
    T = {node}
    num_cut_edges = nx.cut_size(G, S, T)
    volume_S = nx.volume(G, S)
    volume_T = nx.volume(G, T)
    conductance = num_cut_edges / min(volume_S, volume_T) if min(volume_S, volume_T) != 0 else 0
    overall_conductance += conductance

overall_conductance /= len(G.nodes())
print("Overall Conductance of the Network:", overall_conductance)
