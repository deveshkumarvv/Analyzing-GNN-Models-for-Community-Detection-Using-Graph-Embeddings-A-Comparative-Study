import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import networkx as nx
from models import DGI
from utils import process
from scipy.stats import entropy
import community

# Load dataset
dataset = 'cora'

# Training parameters
batch_size = 1
nb_epochs = 100
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu'  # Special name to separate parameters

# Load data and preprocess
adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# Initialize DGI model
model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

# Train DGI model
for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))

# Extract embeddings
embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
embeds_np = embeds.squeeze().cpu().detach().numpy()

# Apply spectral clustering
n_clusters = 6  # Adjust the number of clusters as needed
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
cluster_labels = spectral_clustering.fit_predict(embeds_np)

# Visualize communities
G = nx.Graph()
for i, j, _ in zip(adj.row, adj.col, adj.data):
    G.add_edge(i, j)

node_colors = [cluster_labels[i] for i in range(len(cluster_labels))]

plt.figure(figsize=(8, 6))
nx.draw(G, node_color=node_colors, with_labels=False, node_size=50, cmap=plt.cm.tab10)
plt.title('Detected Communities (Spectral Clustering)')
plt.show()

# Calculate density
density = nx.density(G)
print("Density of the graph:", density)

# Calculate entropy
entropy_value = entropy(np.bincount(cluster_labels), base=2)
print("Entropy of the clusters:", entropy_value)

# Convert cluster labels to a dictionary for Louvain community detection
node_cluster_dict = {i: cluster_labels[i] for i in range(len(cluster_labels))}

# Perform community detection using Louvain algorithm
partition = community.best_partition(G, random_state=42)

# Convert the partition dictionary to a list of cluster labels
louvain_labels = np.array([partition[i] for i in range(len(partition))])

# Calculate modularity
modularity = community.modularity(partition, G)
print("Modularity of the network:", modularity)

# Calculate overall conductance
overall_conductance = nx.algorithms.cuts.conductance(G, cluster_labels)
print("Overall Conductance of the network:", overall_conductance)
