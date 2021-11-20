import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import pykeops
from pykeops.torch import LazyTensor


# 구현사항1. Initializer
# 구현사항2. Update step
# 구현사항3. tolerance, patience


pykeops.clean_pykeops()
# pykeops.test_numpy_bindings()
# pykeops.test_torch_bindings()


def kmeans_initializer(data, n_clusters, n_inits):
    minval = data.min()
    maxval = data.max()
    n_features = data.shape[-1]

    # initialize = lambda: np.linspace(minval, maxval, num=n_features, endpoint=True)
    linspace = np.linspace(minval, maxval, num=n_features, endpoint=True)
    initialize = lambda: np.concatenate([
        np.array([minval]),
        np.random.uniform(linspace[1], linspace[-2], size=n_features-2),
        np.array([maxval])
    ])

    list_centroids = []
    list_mean_dist = []
    for i in range(n_inits):
        centroids = []
        for _ in range(n_clusters):
            point = initialize()
            np.random.shuffle(point)
            centroids.append(point.reshape(1, -1))
        centroids = np.concatenate(centroids)
        mean_dist = np.mean(
            [np.sqrt(np.power(centroids - centroids[n], 2).sum()) / n_clusters for n in range(n_clusters)]
        )
        list_centroids.append(centroids)
        list_mean_dist.append(mean_dist)

    nth = np.argmax(list_mean_dist)
    result = list_centroids[nth]
    print(f"Centroids are initialized (std_dev: {round(list_mean_dist[nth], 2)})")
    return result


def kmeans_cluster(data, n_clusters=10, n_inits=10, max_step=10, min_delta=0.0001, patience=5, verbose=True):
    """
    data: (numpy.ndarray) data for clustering
    n_clusters: (int) the number of clusters
    max_step: (int) maximum step to update centroids
    n_inits: (int) the number of samples to initialize centroids
    min_delta: (float) minumum improvement rate of inertia
    patience: (int) the number of ignoring inertia improvement rate less than min_delta
    verbose: (bool) whether to print detailed information about ongoing execution
    """
    ts = time.time()
    
    with torch.no_grad():

        # cuda_availability
        use_cuda = True if torch.cuda.is_available() else False
        dtype = torch.float32

        # Lazy tensor availability
        MB = 190
        data_size = (data.nbytes / 1024 ** 2)
        if data_size > MB:
            use_lazy = True
        else:
            use_lazy = False
        print("Input data is {:.2f} MB (use_lazy={})".format(data_size, use_lazy))

        # input to tensor
        if use_cuda:
            x = torch.Tensor(data).cuda().type(dtype)
        else:
            x = torch.Tensor(data).type(dtype)

        # initialize
        centroids = kmeans_initializer(data, n_clusters, n_inits)
        centroids = torch.Tensor(centroids).type(dtype)
        if use_cuda:
            centroids = centroids.cuda()

        n_samples, n_features = x.shape

        if use_lazy:
            x_broadcast = LazyTensor(x, axis=0)
        else:
            x_broadcast = x[:, None, :].clone()  # (n_samples, 1, n_features)

        # K-means execution
        pinned = None
        patience_count = 0
        last_inertia = None
        for step in range(1, max_step + 1):
            if use_lazy:
                centroids_broadcast = LazyTensor(centroids, axis=1)
            else:
                centroids_broadcast = centroids[None, :, :].clone()  # (1, n_clusters, n_features)

            distances_all = ((x_broadcast - centroids_broadcast) ** 2).sum(dim=-1)  # (n_samples, n_clusters) -> symbolic matrix of squared distances
            labels = distances_all.argmin(dim=1).long().view(-1)
            distances = distances_all.min(axis=1)
            # distances = torch.gather(distances_all, dim=1, index=labels.reshape(-1, 1))  # distances per cluster centre

            if use_lazy:
                inertia = distances.sum().item()
            else:
                inertia = distances.values.sum().item()

            if verbose:
                print("\nstep {:<3d} -> inertia is {:.2f}".format(step, inertia), end="")

            # pinned
            if (pinned is not None) and (pinned[3] <= inertia):
                pass
            else:
                pinned = (centroids, labels, distances, inertia)
                if verbose:
                    print(" (pinned)", end="")

            # Patience & Tolerance
            if last_inertia is None:
                pass
            elif (last_inertia * min_delta) > (last_inertia - inertia):
                patience_count += 1
            else:
                patience_count = 0
            if patience_count == patience:
                if verbose:
                    print(f"\n* Early stopped: patience_count is {patience_count}", end="")
                break

            # Update centroids (Compute the cluster centroids with torch.bincount)
            labels_count = torch.bincount(labels).type(dtype)
            for dimension in range(n_features):
                centroids[:, dimension] = torch.bincount(labels, weights=x[:, dimension]) / labels_count

            # last_inertia
            last_inertia = inertia

    te = time.time()

    if verbose:
        print("\n\nK-means execution with {:,} points in dimension {:,}, n_clusters = {:,}:".format(n_samples, n_features, n_clusters))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                step, te - ts, step, (te - ts) / step))

    result = (elem.numpy() if type(elem) == torch.Tensor else elem for elem in pinned)
    return result