from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Add, Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class VGGNet:
    def __init__(self, modelName, inputShape, classesNumber):
        self.width = inputShape[0]
        self.height = inputShape[1]
        self.channels = inputShape[2]
        self.classesNumber = classesNumber
        if modelName.lower() == "vgg16":
            self.model = self.VGG16()
        elif modelName.lower() == "vgg19":
            self.model = self.VGG19()
        else:
            self.model = None

        self.model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        return

    def VGG16(self):
        input = Input(shape = (self.height, self.width, self.channels), name = "input")
        endlayer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same")(input)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)
        
        endlayer = Flatten()(endlayer)
        endlayer = Dense(4096)(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Dense(4096)(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Dense(self.classesNumber)(endlayer) 
        endlayer = Activation("softmax")(endlayer)
        model = keras.Model(input, endlayer)
        return model

    def VGG19(self):
        input = Input(shape = (self.height, self.width, self.channels), name = "input")
        endlayer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same")(input)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)

        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same")(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)
        
        endlayer = Flatten()(endlayer)
        endlayer = Dense(4096)(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Dense(4096)(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Dense(self.classesNumber)(endlayer) 
        endlayer = Activation("softmax")(endlayer)
        model = keras.Model(input, endlayer)
        return model

    def GetModel(self):
        return self.model

    def Save(self, path):
        print("---model saved---")
        self.model.save("../models/" + path + '.h5')
        return

    def Summary(self):
        if self.model == None:
            print("Model is not define")
        else:
            self.model.summary()
        return

if __name__ == "__main__":
    classes_number = 2
    inputShape = (224, 224, 3)
    model = VGGNet("vgg16", inputShape, classes_number)
    model.Summary()
    model.Save("vgg16")