import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from autoencoder import Autoencoder


def main() -> None:
    (x_train, y_train), _ = mnist.load_data()

    # Filter the data to get only the digits 1 and 8
    x_data_18 = x_train[np.isin(y_train, [1, 8])]
    y_data_18 = y_train[np.isin(y_train, [1, 8])]

    # Split the data into train and test
    x_train_18, x_test_18, _, _ = train_test_split(x_data_18, y_data_18, test_size=0.2, random_state=42)

    # Normalize the data
    x_train_18 = x_train_18 / 255.0
    x_test_18 = x_test_18 / 255.0

    # Reshape the images for the autoencoder
    x_train_18 = x_train_18.reshape((x_train_18.shape[0], 28, 28, 1))
    x_test_18 = x_test_18.reshape((x_test_18.shape[0], 28, 28, 1))

    # Train the autoencoder
    autoencoder = Autoencoder(input_shape=(28, 28, 1), encoding_dim=32, optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train_18, epochs=50, batch_size=256, validation_data=(x_test_18, x_test_18),
                    activation_encoding='relu', activation_decoding='sigmoid')

    # Predicting on test data
    decoded_imgs = autoencoder.predict(x_test_18)
    # TODO: Add the other graph
    # Display original and reconstructed images
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_18[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
