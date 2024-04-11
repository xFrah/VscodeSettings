import numpy as np


def euclidean_distance(test_image, train_image):
    # simple euclidean distance
    return np.sum((test_image - train_image) ** 2) ** 0.5


zipped = zip(images_train, labels_train)


@profile
def knn(test_image, images_train, labels_train, k):
    # compute euclidean distance between test_image and each train_image
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

    correct = 0
    for test_image, label in zip(images_test, labels_test):
        prediction = knn(test_image, images_train, labels_train, 5)
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
