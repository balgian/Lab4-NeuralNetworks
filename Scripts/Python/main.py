import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


def main() -> None:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Filter the data to get only the digits 1 and 8
    x_train_18 = x_train[np.isin(y_train, [1, 8])]
    x_test_18 = x_test[np.isin(y_test, [1, 8])]
    y_train_18 = y_train[np.isin(y_train, [1, 8])]
    y_test_18 = y_test[np.isin(y_test, [1, 8])]

    # Split the data into train and test
    x_train_18, _, y_train_18, _ = train_test_split(x_train_18, y_train_18, test_size=0.2, random_state=42)

    # Normalize the data
    x_train_18 = x_train_18 / 255.0
    x_test_18 = x_test_18 / 255.0

    # Reshape the images for the autoencoder
    x_train_18 = x_train_18.reshape((x_train_18.shape[0], 28, 28, 1))
    x_test_18 = x_test_18.reshape((x_test_18.shape[0], 28, 28, 1))
    # Train the autoencoder
    autoencoder = Autoencoder(input_shape=(28, 28, 1), encoding_dim=2, optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train_18, epochs=100, batch_size=256, validation_data=(x_test_18, x_test_18),
                    activation_encoding='relu', activation_decoding='sigmoid')
    # Predicting on test data
    decoded_imgs = autoencoder.predict(x_test_18)

    # Display original and reconstructed images
    n = 20
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_18[i].reshape(28, 28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

    # Plot the clusters
    plt.figure(figsize=(10, 10))
    # Obtain the labels for the test data
    encoded_imgs = autoencoder.encoder.predict(x_test_18)
    plotcl(encoded_imgs, y_test_18, coord=slice(0, 2))

    plotmodelhistory(autoencoder.autoencoder.history)

    # Compute Variance Accounted For (VAF)
    vaf_value = compute_vaf(x_test_18.flatten(), decoded_imgs.flatten())

    # Plot VAF as bar chart
    plt.figure()
    x = np.unique(y_test_18)
    vaf_values = [compute_vaf(x_test_18[y_test_18 == digit].flatten(), decoded_imgs[y_test_18 == digit].flatten()) for digit in x]
    plt.bar(x, vaf_values, color='skyblue')
    plt.xlabel('Digits')
    plt.ylabel('VAF (%)')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    for i, vaf in enumerate(vaf_values):
        plt.text(x[i], vaf, f'{vaf:.2f}%', ha='center', va='bottom')
    plt.xticks(x)
    plt.ylim(0, 100)
    plt.show()


def compute_vaf(true_data: np.ndarray, predicted_data: np.ndarray) -> float:
    """
    * Compute the Variance Accounted For (VAF) between true data and predicted data.

    * :param true_data: Actual data.
    * :param predicted_data: Predicted data.
    * :return: VAF value.
    """
    vaf = 1 - np.var(true_data - predicted_data) / np.var(true_data)
    return vaf * 100  # Convert to percentage


# * Autoencoder Model loss
def plotmodelhistory(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plotcl(x: np.ndarray, Xlbl: np.ndarray, coord: Optional[slice] = None, psize: int = 2) -> None:
    """
    * Plot clusters in 1, 2 or 3 dimensions.

    * :param x : Data to plot.
    * :param Xlbl : Labels of the data.
    * :param coord : Coordinates to plot (default is to plot the first 3 coordinates).
    * :param psize : Size of the points (default is 2).
    """
    unique_labels = np.unique(Xlbl)
    colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5],
              [1, .5, 0], [.5, 1, 0], [.5, 0, 1], [0, 1, .5], [0, .5, 1], [.5, .5, 0], [.5, 0, .5],
              [0, .5, .5]]
    pointstyles = 'o+*xsd^v><ph'
    if coord is None:
        coord = slice(0, min(3, x.shape[1]))
    x = x[:, coord]

    plt.figure()

    for label in unique_labels:
        indices = np.where(Xlbl == label)
        cc = colors[label % len(colors)]
        ps = pointstyles[label % len(pointstyles)]
        if x.shape[1] == 1:
            plt.scatter(x[indices, 0], np.zeros_like(x[indices, 0]), c=[cc], marker=ps, s=psize, linewidth=int(np.log(psize + 1) + 1), label=f"Label {label}")
            plt.xlabel('1st dimension of the encoder')
        elif x.shape[1] == 2:
            plt.scatter(x[indices, 0], x[indices, 1], c=[cc], marker=ps, s=psize, linewidth=int(np.log(psize + 1) + 1), label=f"Label {label}")
            plt.xlabel('1st dimension of the encoder')
            plt.ylabel('2nd dimension of the encoder')
        else:
            ax = plt.axes(projection='3d')
            ax.scatter3D(x[indices, 0], x[indices, 1], x[indices, 2], c=[cc], marker=ps, s=psize,
                         linewidth=int(np.log(psize + 1) + 1), label=f"Label {label}")
            ax.set_xlabel('1st dimension of the encoder')
            ax.set_ylabel('2nd dimension of the encoder')
            ax.set_zlabel('3rd dimension of the encoder')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()