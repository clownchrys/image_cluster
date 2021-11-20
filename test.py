from image_cluster import ImageCluster
from data import ImageData
from models import get_model


MODEL = get_model("vgg16_sparse", kwargs=None, state_file="default")
INPUT_SHAPE = MODEL.input_shape


def run_v1():
    cluster = ImageCluster(MODEL)

    cateCd = "009006007"
    cateNm = "Category"

    list_goodsNo, list_url = cluster.get_samples(cateCd, is_all=True)
    dataset = ImageData(list_url, list_goodsNo, shape=INPUT_SHAPE, timeout=30, max_retry=3)
    features = cluster.featurize(
        dataset,
        batch_size=100,
        num_workers=10
    )
    features, labels, distances, centroids = cluster.clusterize(
        cateCd,
        features,
        batch_size=10000,
        epochs=200,
        patience=5,
        min_delta=0.0001,
        all_cuda=False
    )
    
    cluster.update_to_db(list_goodsNo, list_url, cateCd, cateNm, labels, distances, centroids)
    
def run_v2(cateCd):
    cluster = ImageCluster(MODEL)

    list_goodsNo, list_url = cluster.get_samples(cateCd, is_all=False)
    dataset = ImageData(list_url, list_goodsNo, shape=INPUT_SHAPE, timeout=30, max_retry=3)
    features = cluster.featurize(
        dataset,
        batch_size=100,
        num_workers=10
    )
    try:
        features, labels, distances, centroids = cluster.compute_clusters(cateCd, features)
    except FileNotFoundError:
        features, labels, distances, centroids = cluster.clusterize(
            cateCd,
            features,
            batch_size=10000,
            epochs=200,
            patience=5,
            min_delta=0.0001,
            all_cuda=False
        )
        
    cluster.update_to_db(list_goodsNo, list_url, cateCd, cateNm, labels, distances, centroids)


if __name__ == "__main__":
    run_v1()