"""
K-Means Clustering

This module contains functions to perform K-Means clustering on a set of points.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np


def read_points_from_file() -> np.ndarray:
    """
    Read points from the input file and return them as a NumPy array.

    Returns:
        np.ndarray: Array containing points.
    """
    points = []
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            x_axis, y_axis = map(float, line.strip().split())
            points.append([x_axis, y_axis])
    return np.array(points)


def centroids_initialization(points: np.ndarray) -> np.ndarray:
    """
    Initialize centroids randomly from the given points.

    Args:
        points (np.ndarray): Array containing points.

    Returns:
        np.ndarray: Array containing initial centroids.
    """
    centroids_indices = np.random.choice(len(points), NUM_CLUSTERS, replace=False)
    centroids = points[centroids_indices]
    return centroids


def assign_points_to_clusters(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign points to clusters based on the nearest centroids.

    Args:
        points (np.ndarray): Array containing points.
        centroids (np.ndarray): Array containing centroids.

    Returns:
        np.ndarray: Array containing cluster assignments for each point.
    """
    num_centroids = len(centroids)
    num_points = len(points)

    squared_distances = np.zeros((num_centroids, num_points))

    for i in range(num_centroids):
        for j in range(num_points):
            point = points[j]

            squared_distance = np.sum((point - centroids[i]) ** 2)
            squared_distances[i, j] = squared_distance

    distances = np.sqrt(squared_distances)
    return np.argmin(distances, axis=0)


def update_centroids(points: np.ndarray, cluster_assignments: np.ndarray) -> np.ndarray:
    """
    Update centroids based on the assigned clusters.

    Args:
        points (np.ndarray): Array containing points.
        cluster_assignments (np.ndarray): Array containing cluster assignments for each point.

    Returns:
        np.ndarray: Updated centroids.
    """
    centroids = np.zeros((NUM_CLUSTERS, 2))
    for i in range(NUM_CLUSTERS):
        centroids[i] = np.average(points[cluster_assignments == i], axis=0)
    return centroids


def write_output_file(centroids: np.ndarray) -> None:
    """
    Write centroids to the output file.

    Args:
        centroids (np.ndarray): Array containing centroids.
    """
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as file:
        file.write(str(NUM_CLUSTERS) + '\n')
        for i, centroid in enumerate(centroids):
            file.write(f"{i + 1} {centroid[0]} {centroid[1]}\n")


def plot_clusters(points: np.ndarray, cluster_assignments: np.ndarray) -> None:
    """
    Plot clusters and save the plot to an image file.

    Args:
        points (np.ndarray): Array containing points.
        cluster_assignments (np.ndarray): Array containing cluster assignments for each point.
    """
    plt.figure(figsize=(6, 6))
    for i in range(NUM_CLUSTERS):
        cluster_points = points[cluster_assignments == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
    plt.title('K-Means')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(IMAGE_FILE_PATH)
    plt.show()


def k_means(max_iterations: int) -> None:
    """
    Perform K-means clustering.

    Args:
        max_iterations (int): Maximum number of iterations for K-means algorithm.
    """
    points = read_points_from_file()

    centroids = centroids_initialization(points)
    cluster_assignments = np.array([])
    for _ in range(max_iterations):
        cluster_assignments = assign_points_to_clusters(points, centroids)
        new_centroids = update_centroids(points, cluster_assignments)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    write_output_file(centroids)

    plot_clusters(points, cluster_assignments)


if len(sys.argv) != 5:
    print("Try again! Enter: python [script.py]"
          " [input_file_path] [output_file_path] [image_file_path] [number_of_clusters]")
    sys.exit(1)

INPUT_FILE_PATH: str = sys.argv[1]
OUTPUT_FILE_PATH: str = sys.argv[2]
IMAGE_FILE_PATH: str = sys.argv[3]
NUM_CLUSTERS: int = int(sys.argv[4])

k_means(NUM_CLUSTERS)
