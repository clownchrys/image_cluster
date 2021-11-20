import logging
import os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import numpy as np
import torch
from kmeans_torch_v2 import KMeans


CUDA = torch.cuda.is_available()
logger = logging.getLogger(__name__)


def query_to_php(query):
    import requests
    import json
    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14'
    headers = {'User-Agent': user_agent,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    data={}
    data['query'] = query
    api_req = requests.post('http://api.crunchprice.com/goods/sql_query.php',data=data,headers=headers)
    try:
        data = json.loads(api_req.text)
        return data['response']
    except Exception as e:
        return None


class ImageCluster:

    def __init__(self, model):
        self.model = model
        if CUDA:
            model = model.cuda()

        resource_dir = "./"
        self.pca_components_dir = resource_dir + "_pca_components/"
        self.centroids_dir = resource_dir + "_centroids/"
        
        for directory in [self.pca_components_dir, self.centroids_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_samples(self, cateCd, is_all=True):
        # if is_all:
        #     filter = {"cateCd": cateCd}
        # else:
        #     filter = {"cateCd": cateCd, "clusterNo": None}

        # documents = self._collection.find(
        #     filter=filter,
        #     projection={"_id": 0, "goodsNo": 1, "imageName": 1}
        # )
        # list_goodsNo = []
        # list_url = []
        # for doc in documents:
        #     list_goodsNo.append(doc['goodsNo'])
        #     list_url.append(doc['imageName'])

        query = \
            f"""
            SELECT g.goodsNo, i.imageName
            FROM es_goods g
            JOIN es_goodsImage i ON g.goodsNo = i.goodsNo
            WHERE g.cateCd = {cateCd}
                AND g.delFl = 'n'
                AND i.imageKind = 'main'
                AND i.imageSize = 0
                AND i.imageHeightSize = 0
            """
        rows = query_to_php(query)
        list_goodsNo = []
        list_url = []
        for row in rows:
            list_goodsNo.append(row['goodsNo'])
            list_url.append(row['imageName'])

        return list_goodsNo, list_url

    def featurize(self, dataset, batch_size, num_workers):
        # setup dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
        )

        # prediction
        with torch.no_grad():
            list_features = []
            for batch_no, data in enumerate(dataloader, 1):
                if CUDA:
                    data = data.cuda()
                features = self.model(data)
                features = features.view(len(data), -1).cpu().numpy()
                list_features.append(features)
                print("\r{} / {} Batches Featurized".format(batch_no, len(dataloader)), end="")

        print(f"\nLoad failed: {len(dataset.failed)} qty")

        # samplewise re-distribution
        features = np.concatenate(list_features)
        features = normalize(features, axis=1, copy=False)
        return features

    def clusterize(self, cateCd, features, batch_size=10000, epochs=200, patience=5, min_delta=0.0001, all_cuda=False):
        # hyper-parameters
        E = 100
        max_initial_k = 10
        min_initial_k = 1
        reduce_dimension = 5

        # calculate initial_k
        n_samples = features.shape[0]
        initial_k = int(min(max(n_samples / E, min_initial_k), max_initial_k))

        # pca
        pca = PCA(reduce_dimension)
        pca.fit(features)
        pca.components_ *= KMeans.NORMALIZE_SCALE_CONST
        features = pca.transform(features)

        file_path = os.path.join(self.pca_components_dir, "pca_component_{}.npy".format(cateCd))
        np.save(file_path, pca.components_)
        logger.info("pca_component saved ({})".format(file_path))

        # kmeans
        kmeans = KMeans(features, initial_k=initial_k, batch_size=batch_size, epochs=epochs, patience=patience, min_delta=min_delta, all_cuda=all_cuda)
        kmeans.train()
        labels, l2_norms, centroids = kmeans.predict()

        file_path = os.path.join(self.centroids_dir, "centroid_{}.npy".format(cateCd))
        np.save(file_path, centroids)
        logger.info("centroid saved ({})".format(file_path))

        return features, labels, l2_norms, centroids

    def compute_clusters(self, cateCd, features):
        pca_components_path = os.path.join(self.pca_components_dir, "pca_component_{}.npy".format(cateCd))
        centroids_path = os.path.join(self.centroids_dir, "centroid_{}.npy".format(cateCd))

        pca_components = np.load(pca_components_path)
        centroids = np.load(centroids_path)
        features = np.matmul(features, pca_components.T)
        
        kmeans = KMeans(data, initial_k=None, batch_size=10000, epochs=None, patience=None, min_delta=None, all_cuda=True)
        kmeans.centroids = centroids
        labels, distances = kmeans.predict()
        
        return features, labels, distances, centroids
    
    def write_to_db(self, list_goodsNo, list_url, cateCd, cateNm, labels, distances, centroids):
        for goodsNo, url, label, distance in zip(list_goodsNo, list_url, labels, distances):
            print("goodsNo: {goodsNo},\nimageName: {url},\ncateCd: {cateCd},\ncateNm: {cateNm},\nclusterNo: {label},\nl2_norm: {distance}\n".format(
                goodsNo = goodsNo, url=url, cateCd=cateCd, cateNm=cateNm, label=label, distance=distance
            ))
    
    def update_to_db(self, list_goodsNo, list_url, cateCd, cateNm, labels, distances, centroids):
        for goodsNo, url, label, distance in zip(list_goodsNo, list_url, labels, distances):
            print("goodsNo: {goodsNo},\nimageName: {url},\ncateCd: {cateCd},\ncateNm: {cateNm},\nclusterNo: {label},\nl2_norm: {distance}\n".format(
                goodsNo = goodsNo, url=url, cateCd=cateCd, cateNm=cateNm, label=label, distance=distance
            ))
