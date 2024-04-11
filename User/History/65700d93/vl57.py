import numpy as np


# function that computes mean vectors for each class and returns them in a list, given the training images and labels
def compute_mean_vectors(images_train, labels_train):
    mean_vectors = []
    for i in range(10):
        mean_vectors.append(np.mean(images_train[labels_train == i], axis=0))
    return mean_vectors


# given the list of mean vectors, return a dictionary with class label as key and the list of images in that class as value
def get_class_images(images_train, labels_train):
    class_images = []
    for i in range(10):
        class_images.append(images_train[labels_train == i])
    return class_images


def euclidean_distance(test_image, train_image):
    # simple euclidean distance
    return np.sum((test_image - train_image) ** 2) ** 0.5


def knn(test_image, images_train, labels_train, k, zipped):
    # compute euclidean distance between test_image and each train_image

    sorted1 = sorted(zipped, key=lambda x: euclidean_distance(test_image, x[0]))

    distances_list = [(euclidean_distance(test_image, train_image), label) for train_image, label in zip(images_train, labels_train)]
    distances_list.sort(key=lambda x: x[0])
    # get k nearest neighbors
    k_nearest_neighbors = [label for _, label in distances_list[:k]]
    # return the most common label
    return max(k_nearest_neighbors, key=k_nearest_neighbors.count)


def test_kNN():
    images_train = np.load("datasets/images_train.npy")
    labels_train = np.load("datasets/labels_train.npy")
    images_test = np.load("datasets/images_test.npy")
    labels_test = np.load("datasets/labels_test.npy")

    means = compute_mean_vectors(images_train, labels_train)
    class_images = get_class_images(images_train, labels_train)
    zipped = zip(means, class_images)

    correct = 0
    for test_image, label in zip(images_test, labels_test):
        prediction = knn(test_image, images_train, labels_train, 5, zipped)
        if prediction == label:
            correct += 1
            print("correct", prediction, label)
        else:
            print("incorrect", prediction, label)
        if correct == 5:
            print("porn)")
            return
    print(f"Accuracy: {correct / len(labels_test)}")


if __name__ == "__main__":
    test_kNN()
