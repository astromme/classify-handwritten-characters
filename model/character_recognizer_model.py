import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU, Dropout
from tensorflow.keras import Model

class CharacterRecognizerModel(Model):
    def __init__(self, keep_prob, n_classes):
        super(CharacterRecognizerModel, self).__init__()
        self.keep_prob = keep_prob
        self.n_classes = n_classes

        mu = 0
        sigma = 0.1
        self.initializer = tf.keras.initializers.TruncatedNormal(mu, sigma)

        # layer 1
        self.conv1 = Conv2D(filters=3, kernel_size=5, padding='valid')
        self.maxpool1 = MaxPool2D(pool_size=(2,2), padding='same')
        self.relu1 = ReLU()
        self.dropout1 = Dropout(1 - keep_prob)

        # layer 2
        self.conv2 = Conv2D(filters=3, kernel_size=5, padding='valid')
        self.maxpool2 = MaxPool2D(pool_size=(2,2), padding='same')
        self.relu2 = ReLU()
        self.dropout2 = Dropout(1 - keep_prob)
        self.flatten2 = tf.keras.layers.Flatten()

        # layer 3
        self.conv3 = Conv2D(filters=3, kernel_size=3, padding='valid')
        self.maxpool3 = MaxPool2D(pool_size=(2,2), padding='same')
        self.relu3 = ReLU()
        self.dropout3 = Dropout(1 - keep_prob)
        self.flatten3 = tf.keras.layers.Flatten()

        self.dense4 = Dense(384)
        self.relu4 = ReLU()
        self.dense5 = Dense(128)
        self.relu5 = ReLU()

        self.dense6 = Dense(n_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        layer_2_flattened = self.flatten2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = tf.keras.layers.concatenate([self.flatten3(x), layer_2_flattened], 1)

        x = self.dense4(x)
        x = self.relu4(x)
        x = self.dense5(x)
        x = self.relu5(x)

        return self.dense6(x)

    def get_config(self):
        return {
            "keep_prob": self.keep_prob,
            "n_classes": self.n_classes
        }
