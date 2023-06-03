#segment images of different fruits based on their color and texture features
import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    centroids_idx = np.random.choice(range(X.shape[0]), size=k, replace=False)
    centroids = X[centroids_idx]
    return centroids

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[clusters == i]
        centroids[i] = cluster_points.mean(axis=0)
    return centroids

def kmeans(X, k, num_iterations=10):
    centroids = initialize_centroids(X, k)
    for _ in range(num_iterations):
        clusters = assign_clusters(X, centroids)
        centroids = update_centroids(X, clusters, k)
    return clusters, centroids

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def segment_fruits(image_path, k):
    image = load_image(image_path)
    height, width, _ = image.shape
    reshaped_image = image.reshape(height * width, 3)

    clusters, centroids = kmeans(reshaped_image, k)
    segmented_image = centroids[clusters].astype(np.uint8).reshape(height, width, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title("Original Image")
    ax1.imshow(image)
    ax1.axis('off')
    ax2.set_title(f"Segmented Image (k={k})")
    ax2.imshow(segmented_image)
    ax2.axis('off')
    plt.show()

# Example usage
image_path = 'fruits.jpg'
k = 5
segment_fruits(image_path, k)
