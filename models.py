from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation, Dropout
from keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB5
import tensorflow as tf

import matplotlib.pyplot as plt


class tf_models():

    def __init__(self,image_shape,classes):
        self.image_shape = image_shape
        self.classes = classes

    def VGG_16(self):
        #VGG16 Model
        base_model=VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=self.image_shape+[3]
        )

        for layer in base_model.layers:
            layer.trainable = False

        VGG16_model=Sequential()
        VGG16_model.add(base_model)
        VGG16_model.add(Flatten())
        VGG16_model.add(Dense(800,activation=('relu')))
        VGG16_model.add(Dense(650,activation=('relu')))
        VGG16_model.add(Dropout(0.3))
        VGG16_model.add(Dense(500,activation=('relu')))
        VGG16_model.add(Dense(350,activation=('relu')))
        VGG16_model.add(Dense(200,activation=('relu')))
        VGG16_model.add(Dense(110,activation=('relu')))
        VGG16_model.add(Dropout(0.4))
        VGG16_model.add(Dense(98,activation=('relu')))
        VGG16_model.add(Dense(40,activation=('relu')))
        VGG16_model.add(Dense(len(self.classes),activation=('softmax')))

        VGG16_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


        return VGG16_model
    
    def ResNet15_V2(self):

        # Resnet15 V2
        base_model = ResNet152V2(input_shape=self.image_shape + [3],
                                    include_top=False,
                                    weights='imagenet',
                                    pooling='avg'
                                )

        for layer in base_model.layers:
            layer.trainable = False

        Resnet_model=Sequential()
        Resnet_model.add(base_model)
        Resnet_model.add(Dense(128,activation='relu'))
        Resnet_model.add(Dropout(0.2))
        Resnet_model.add(Dense(len(self.classes), activation='softmax'))

        Resnet_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        return Resnet_model
    

    def EfficientNet_B5(self):

        
        base_model=EfficientNetB5(include_top=False, 
                                    weights="imagenet",
                                    input_shape=self.image_shape+[3],
                                    pooling='max')
        for layer in base_model.layers:
            layer.trainable = False

        B5_model=Sequential()
        B5_model.add(base_model)
        B5_model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ))
        B5_model.add(Dense(256, kernel_regularizer = regularizers.l2(l = 0.016),activity_regularizer=regularizers.l1(0.006),
                        bias_regularizer=regularizers.l1(0.006) ,activation='relu'))
        B5_model.add(Dropout(rate=.4, seed=123))
        B5_model.add(Dense(len(self.classes), activation='softmax'))
        
        B5_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        return B5_model

    def customCNN(self):

        #Custom built CNN model
        custom_model=Sequential([layers.Input(shape=self.image_shape+[3]),
                        Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'),
                        MaxPool2D(pool_size=(2,2)),
                        Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'),
                        MaxPool2D(pool_size=(2,2)),
                        layers.Flatten(),
                        Dense(8,activation='relu'),
                        Dense(len(self.classes),activation='softmax')
                        ])
        
        custom_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


        return custom_model
    


def plot_metrics(metrics):
    """
    Plots training and validation loss and accuracy curves side by side using subplots.

    Parameters:
    metrics (pandas.DataFrame): A DataFrame containing training history with 
                                 columns 'loss', 'val_loss', 'accuracy', and 'val_accuracy'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    axes[0].plot(metrics['loss'], label='Training Loss')
    axes[0].plot(metrics['val_loss'], label='Validation Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(metrics['accuracy'], label='Training Accuracy')
    axes[1].plot(metrics['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
