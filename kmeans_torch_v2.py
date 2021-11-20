import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum

import torch
import numpy as np
from tqdm import trange
import math


CUDA = torch.cuda.is_available()


class KMeans:

    NORMALIZE_SCALE_CONST = 1e+2

    def __init__(self, data, initial_k=10, batch_size=10000, epochs=200, patience=5, min_delta=0.0001, all_cuda=True, **kwargs):
        """
        data: (numpy.ndarray) numpy array of data
        initial_k: (int) number of the starting centroids. it can be reduced by training
        batch_size: (int) batch size. make sure that too small batch size makes centroids reduced as the train begins
        epochs: (int) max epoch iterations. if the centeroids not shifting any more, the calculation will cease before this max number
        patience: (int) the number of ignoring inertia improvement(rate) less than min_delta
        min_delta: (float) minumum improvement rate of inertia
        all_cuda: (bool) whether to move all data array to the cuda, or per batch chunks
        """
        dtype = kwargs.get("dtype", np.float32)

        # cuda
        self.all_cuda = all_cuda

        # data
        data = data.astype(dtype)
        self.dim = data.shape[-1]
        
        # tensor
        self.tensor = torch.Tensor(data).detach()
        if all_cuda and CUDA:
            self.tensor = self.tensor.cuda()

        # centroids
        intial_centroids = self.kmeans_plus_plus(data, initial_k, None)
        self.centroids = torch.Tensor(intial_centroids).detach()
        if CUDA:
            self.centroids = self.centroids.cuda()

        # metrics
        self.min_delta = min_delta
        self.patience = patience
        self.patience_count = 0

        # training params
        self.k = initial_k
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.iters = math.ceil(data.shape[0]/self.batch_size)
        self.index = 0

    def train(self):
        # Train & Update
        for epoch in range(self.epochs):
            list_batch_inertia = []
            
            t = trange(self.iters)
            for i in t:
                data = self.get_data(self.index)
                if CUDA and (self.all_cuda == False):
                    data = data.cuda()
                self.step(data)
                self.index += self.batch_size
                t.set_description(
                    "[Epoch: {epoch}\tIter: {iter}]\tk: {k}\tInertia: {inertia:.3f}".format(
                        epoch=epoch + 1,
                        iter=i + 1,
                        k=self.k,
                        inertia=self.batch_inertia
                    )
                )
                list_batch_inertia.append(self.batch_inertia)
            self.index = 0

            # Evaluation
            inertia = np.mean(list_batch_inertia)
            if epoch == 0:
                pass
            elif (self.inertia * self.min_delta) > (self.inertia - inertia):
                self.patience_count += 1
            else:
                self.patience_count = 0
            self.inertia = inertia

            if self.patience_count == self.patience:
                print(f"* Early stopped: patience is reached the limits")
                break

        print("KMeans train finished")
    
    def predict(self):
        t = trange(self.iters)
        for i in t:
            data = self.get_data(self.index)
            self.index += self.batch_size
            if CUDA and (self.all_cuda == False):
                data = data.cuda()

            labels, distances = self.compute_labels(data)
            if i == 0:
                self.labels = labels
                self.distances = distances
            else:
                self.labels = torch.cat([self.labels, labels], dim=-1)
                self.distances = torch.cat([self.distances, distances], dim=-1)

        self.index = 0
        to_numpy = lambda x: x.cpu().numpy()
        return to_numpy(self.labels), to_numpy(self.distances), to_numpy(self.centroids)
    
    def get_data(self, index):
        starts = index
        ends = index + self.batch_size
        data = self.tensor[starts:ends, :]
        return data

    def step(self, data):
        labels, _ = self.compute_labels(data)
        self.update_centroids(labels, data)

    def compute_all_distances(self, data):
        bs = data.size()[0]
        broadcast_substraction = self.centroids.unsqueeze(0).repeat(bs, 1, 1) - data.unsqueeze(1).repeat(1, self.k, 1)
        all_distances = torch.pow(broadcast_substraction, 2).sum(dim=-1)
        return all_distances

    def compute_labels(self, data):
        all_distances = self.compute_all_distances(data)
        distances, labels = torch.min(all_distances, dim=-1)
        self.batch_inertia = distances.mean().item()  # self.distance = all_distances.mean().item()
        return labels, distances

    def update_centroids(self, labels, data):
        if CUDA:
            z = torch.cuda.FloatTensor(self.k, self.dim).fill_(0)
            o = torch.cuda.FloatTensor(self.k).fill_(0)
            ones = torch.cuda.FloatTensor(data.size()[0]).fill_(1)
        else:
            z = torch.zeros(self.k, self.dim)
            o = torch.zeros(self.k)
            ones = torch.ones(data.size()[0])
        centroids = o.index_add(0, labels, ones)
        
        # slice to remove empety sum (no more use such centroid)
        cent_slice = (centroids > 0)
        cent_sum = z.index_add(0, labels, data)[cent_slice.view(-1, 1).repeat(1, self.dim)].view(-1, self.dim)
        centroids = centroids[cent_slice].view(-1, 1)
        self.centroids = cent_sum / centroids
        self.k = self.centroids.size()[0]

    def kmeans_plus_plus(self, X, n_clusters, random_state, n_local_trials=None):
        """
        X: (numpy.ndarray) data
            > shape: (n_samples, n_features)
        n_clusters: (int) the number of seeds to choose
        x_squared_norms: (numpy.ndarray) Squared Euclidean norm of each data point
            > shape: (n_samples,)
        random_state: (int) an integer to make the randomness deterministic
        n_local_trials : (int) the number of seeding trials for each center (except the first)
        """
        n_samples, n_features = X.shape

        centers = np.empty((n_clusters, n_features), dtype=X.dtype)
        x_squared_norms = row_norms(X, squared=True)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))

        # Pick first center randomly
        random_state = check_random_state(random_state)
        center_id = random_state.randint(n_samples)
        if sp.issparse(X):
            centers[0] = X[center_id].toarray()
        else:
            centers[0] = X[center_id]

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = euclidean_distances(
            centers[0, np.newaxis],
            X,
            Y_norm_squared=x_squared_norms,
            squared=True
        )
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_dist_sq),
                rand_vals
            )
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = euclidean_distances(
                X[candidate_ids],
                X,
                Y_norm_squared=x_squared_norms,
                squared=True
            )

            # update closest distances squared and potential for each candidate
            np.minimum(
                closest_dist_sq,
                distance_to_candidates,
                out=distance_to_candidates
            )
            candidates_pot = distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if sp.issparse(X):
                centers[c] = X[best_candidate].toarray()
            else:
                centers[c] = X[best_candidate]
        return centers
