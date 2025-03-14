import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, 
                                     Dropout, Flatten, Dense, RandomFlip, RandomRotation)
from tensorflow.keras.models import Model

# Strategy Pattern: Base Model Strategy
class ModelStrategy:
    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model method")

# Concrete Strategy: CNN Model
class CNNModel(ModelStrategy):
    def build_model(self, input_shape=(32, 32, 3), num_classes=10):
        inputs = Input(shape=input_shape)

        # Data Augmentation
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)

        # Conv Block 1
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)

        # Conv Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)

        # Conv Block 3 (Last Conv Layer for Grad-CAM)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name="last_conv")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.4)(x)

        # Flatten and Fully Connected Layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # Output Layer
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

# Context Class to Select Strategy
class ModelContext:
    def __init__(self, strategy: ModelStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelStrategy):
        self.strategy = strategy

    def build_model(self, **kwargs):
        return self.strategy.build_model(**kwargs)
