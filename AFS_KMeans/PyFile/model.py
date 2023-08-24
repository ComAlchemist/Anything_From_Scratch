import numpy as np 
import matplotlib.pyplot as plt 

# Caculate euclidean distance 
def euclidean_distance(x1, x2): 
    return np.sqrt(np.sum(x1-x2)**2)

# Build model 
class Kmeans: 
    # Item 
    def __init__(self, K=5, max_iters=100, plot_steps=False): 
        self.K = K 
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        #list of sample indices for each cluster 
        self.clusters = [[] for _ in range(self.K)]

        # List of the centers of each cluster 
        self.centrods = []

    
    # Predict 
    def predict(self, X): 
        self.X = X 
        self.n_samples, self.n_features = X.shape 

        # Initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace = False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # Optimizes Clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Plot points 
            if self.plot_steps:
                self.plot()

            # Calculate New centeroids from the clusters 
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check the changing òf the clustering 
            if self._is_converged(centroids_old, self.centroids):
                break
            
            # plot again 
            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    # Get labels cluster 
    def _get_cluster_labels(self, clusters): 
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    # Set up clusters
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids 
        clusters = [[] for _ in range(self.K)]
        # Get pair index and value in self.X
        for idx, sample in enumerate(self.X): 

            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # Finding closest center 
    def _closest_centroid(self, sample, centroids): 
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # Geting center 
    def _get_centroids(self, clusters): 
        # Setting up the center 
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    # def checking the changing of the center 
    def _is_converged(self, centeroids_old, centeroids): 
        # This function check that the center have changing or not 
        # Return the True or False about the changing of all center 
        distances = [euclidean_distance(centeroids_old[i], centeroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    # Plot data 
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()