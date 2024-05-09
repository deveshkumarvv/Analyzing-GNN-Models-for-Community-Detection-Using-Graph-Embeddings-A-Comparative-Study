import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
import community
from scipy.stats import entropy

from models import DGI
from utils import process

def kmeans_clustering(embeddings, n_clusters):
    # Apply K-means clustering to embeddings.
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_

def plot_elbow(embeddings, max_clusters=10):
    # Plot the elbow curve for K-means clustering.
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
    
    # Plotting the elbow curve
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def overall_conductance(cluster_labels, adj):
    # Compute the overall conductance of clusters.
    unique_labels = np.unique(cluster_labels)
    total_external_edges = 0
    total_internal_edges = 0
    
    if isinstance(adj, sp.coo_matrix):
        adj = adj.tocsr()  # Convert to CSR format for efficient row slicing
    
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        external_edges = 0
        internal_edges = 0
        
        for i in cluster_indices:
            neighbors = adj.getrow(i).nonzero()[1]  # Get the indices of non-zero elements in the ith row
            for j in neighbors:
                if cluster_labels[j] == label:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        total_external_edges += external_edges
        total_internal_edges += internal_edges
    
    total_edges = total_external_edges + total_internal_edges
    overall_conductance = total_external_edges / total_edges if total_edges > 0 else 0
    return overall_conductance

try:
    dataset = 'citeseer'

    # Training parameters
    batch_size = 1
    nb_epochs = 100
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 512
    sparse = True
    nonlinearity = 'prelu'  # special name to separate parameters

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).tocoo()  # Convert to COO format

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj.toarray()[np.newaxis])  # Convert to dense array if not sparse
    labels = torch.FloatTensor(labels[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

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

    loss_values = []  # Store loss values for plotting

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

        loss_values.append(loss.item())  # Append loss value to list

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

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeds = embeds.squeeze().cpu().detach().numpy()  # Convert to numpy array and remove batch dimension

    # Apply K-means clustering
    n_clusters = 6  # Adjust the number of clusters to 6
    cluster_labels = kmeans_clustering(embeds, n_clusters)

    print("Cluster Labels:", cluster_labels)

    # Calculate density of the graph
    total_edges = adj.nnz  # Total number of edges in the graph
    total_possible_edges = adj.shape[0] * (adj.shape[0] - 1) / 2  # Total possible number of edges in an undirected graph
    density = total_edges / total_possible_edges
    print("Density of the graph:", density)

    # Create a graph from the adjacency matrix using edges
    G = nx.Graph()
    for i, j, _ in zip(adj.row, adj.col, adj.data):
        G.add_edge(i, j)

    # Set node colors based on cluster labels
    node_colors = [cluster_labels[i] for i in range(len(cluster_labels))]

    # Draw the graph with node colors
    plt.figure(figsize=(8, 6))
    nx.draw(G, node_color=node_colors, with_labels=False, node_size=50, cmap=plt.cm.tab10)
    plt.title('Detected Communities (k=6)')
    plt.show()

    # Apply the elbow method to determine the optimal value of k
    plot_elbow(embeds)

    # Compute entropy
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
    overall_cond = overall_conductance(cluster_labels, adj)
    print("Overall Conductance:", overall_cond)

    # Plot loss vs epochs
    plt.plot(range(len(loss_values)), loss_values, marker='o')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

except Exception as e:
    print("An error occurred:", e)
