from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Add, Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class MobileNet:
    def __init__(self, modelName, inputShape, classesNumber):
        self.width = inputShape[0]
        self.height = inputShape[1]
        self.channels = inputShape[2]
        self.classesNumber = classesNumber
        if modelName.lower() == "mobilenet_v1":
            self.model = self.MobileNet_v1()
        elif modelName.lower() == "mobilenet_v3":
            self.model = self.MobileNet_v2()
        elif modelName.lower() == "mobilenet_v3":
            self.model = self.MobileNet_v3()
        else:
            self.model = None
        return

    def DepthwiseSeparableConv(self, input, filters, strides):
        endlayer = input
        endlayer = DepthwiseConv2D(kernel_size = (3, 3), strides = strides, padding = "same")(endlayer)
        endlayer = BatchNormalization()(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = Conv2D(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = "same")(endlayer)
        endlayer = BatchNormalization()(endlayer)
        endlayer = Activation("relu")(endlayer)
        return endlayer

    def MobileNet_v1(self):
        input = Input(shape = (self.height, self.width, self.channels), name = "input")
        endlayer = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = "same")(input)
        endlayer = BatchNormalization()(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = self.DepthwiseSeparableConv(endlayer, 64, (1, 1))
        endlayer = self.DepthwiseSeparableConv(endlayer, 128, (2, 2))
        endlayer = self.DepthwiseSeparableConv(endlayer, 128, (1, 1))
        endlayer = self.DepthwiseSeparableConv(endlayer, 256, (2, 2))
        endlayer = self.DepthwiseSeparableConv(endlayer, 256, (1, 1))
        endlayer = self.DepthwiseSeparableConv(endlayer, 512, (1, 1))
        for index in range(5):
            endlayer = self.DepthwiseSeparableConv(endlayer, 512, (1, 1))
        endlayer = self.DepthwiseSeparableConv(endlayer, 512, (2, 2))
        endlayer = self.DepthwiseSeparableConv(endlayer, 1024, (2, 2))
        endlayer = GlobalAveragePooling2D()(endlayer)
        endlayer = Dense(1024)(endlayer) 
        endlayer = Dense(self.classesNumber)(endlayer) 
        endlayer = Activation("softmax")(endlayer)
        model = keras.Model(input, endlayer)
        return model

    def MobileNet_v2(self):
        return None

    def MobileNet_v3(self):
        return None

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
    model = MobileNet("mobilenet_v1", inputShape, classes_number)
    model.Summary()
    model.Save("mobilenet")