import numpy as np
from tensorflow.keras import layers, Model, optimizers
from typing import Tuple, Optional


class Autoencoder(object):
    """
    * An autoencoder model for compressing and reconstructing input data.
    * Made for MNIST input digits' database.
    """

    def __init__(self, input_shape: Tuple[int, int, int], encoding_dim: int, optimizer: str = 'adam',
                 loss: str = 'binary_crossentropy'):
        """
        * Initializes the Autoencoder with specified parameters.

        * :param input_shape: Shape of the input data (height, width, channels).
        * :param encoding_dim: Dimension of the encoded representation.
        * :param optimizer: Optimizer to use during training.
        * :param loss: Loss function to use during training.
        """
        self.input_shape: Tuple[int, int, int] = input_shape
        self.encoding_dim: int = encoding_dim
        self.optimizer: str = optimizer
        self.loss: str = loss
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None

    def _build_model(self, activation_encoding, activation_decoding) -> None:
        """
        * Builds the encoder, decoder, and autoencoder models.
        """
        input_img = layers.Input(shape=self.input_shape)
        x = layers.Flatten()(input_img)
        encoded = layers.Dense(self.encoding_dim, activation=activation_encoding)(x)

        x = layers.Dense(int(np.prod(self.input_shape)), activation=activation_decoding)(encoded)
        decoded = layers.Reshape(self.input_shape)(x)

        self.autoencoder = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)
        encoded_input = layers.Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoencoder.layers[-2]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x_train: np.ndarray, epochs: int = 50, batch_size: int = 256, shuffle: bool = True,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            activation_encoding: str = 'relu', activation_decoding: str = 'sigmoid') -> None:
        """
        * Trains the autoencoder on the training data.

        * :param x_train: Training data.
        * :param epochs: Number of training epochs.
        * :param batch_size: Size of each batch.
        * :param shuffle: Whether to shuffle the data.
        * :param validation_data: Tuple of validation data (x_val, y_val).
        * :param activation_encoding: Activation function for the encoding layer.
        * :param activation_decoding: Activation function for the decoding layer.
        """
        self._build_model(activation_encoding=activation_encoding, activation_decoding=activation_decoding)
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                             validation_data=validation_data)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        * Encodes and reconstructs the test data.

        * :param x_test: Test data to reconstruct.
        * :return: Reconstructed data.
        """
        return self.decoder.predict(self.encoder.predict(x_test))
