from tensorflow import keras
from tensorflow.keras.layers import Add, Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D

class ResNet:
    def __init__(self, modelName, inputShape, classesNumber):
        self.width = inputShape[0]
        self.height = inputShape[1]
        self.channels = inputShape[2]
        self.classesNumber = classesNumber
        self.block1ChannelsArray = (64, 64, 256)
        self.block1KernelArray = [(1, 1), (3, 3), (1, 1)]
        self.block1ActivationArray = ["relu", "relu", "relu"]
        self.block2ChannelsArray = (128, 128, 512)
        self.block2KernelArray = [(1, 1), (3, 3), (1, 1)]
        self.block2ActivationArray = ["relu", "relu", "relu"]
        self.block3ChannelsArray = (256, 256, 1024)
        self.block3KernelArray = [(1, 1), (3, 3), (1, 1)]
        self.block3ActivationArray = ["relu", "relu", "relu"]
        self.block4ChannelsArray = (512, 512, 2048)
        self.block4KernelArray = [(1, 1), (3, 3), (1, 1)]
        self.block4ActivationArray = ["relu", "relu", "relu"]
        if modelName.lower() == "resnet50":
            self.ResNet50()
        elif modelName.lower() == "resnet101":
            self.ResNet101()
        elif modelName.lower() == "resnet152":
            self.ResNet101()
        else:
            self.model = None
        return
    
    def Bottleneck(self, input, channels, kernels, strides, activations):
        endlayer = input
        length = len(kernels)
        for index in range(0, length):
            if index == 0:
                endlayer = Conv2D(filters = channels[index], kernel_size = kernels[index], strides = strides, padding = "same")(endlayer)
                if strides[0] == 2:
                    input = Conv2D(filters = channels[length - 1], kernel_size = kernels[index], strides = strides, padding = "same")(input)
                    input = BatchNormalization()(input)
            else:
                endlayer = Conv2D(filters = channels[index], kernel_size = kernels[index], strides = (1, 1), padding = "same")(endlayer)
            endlayer = BatchNormalization()(endlayer)
            if index < length - 1:
                endlayer = Activation(activations[index])(endlayer)
        endlayer = Add()([input, endlayer])
        endlayer = Activation(activations[index])(endlayer)
        return endlayer
        
    def ConvBlock(self, input, channelsArray, kernelsArray, activationsArray, iteration):
        endlayer = input
        index = 0
        while index < iteration:
            if index != 0:
                endlayer = self.Bottleneck(endlayer, channelsArray, kernelsArray, (1, 1), activationsArray)
            else:
                endlayer = self.Bottleneck(endlayer, channelsArray, kernelsArray, (2, 2), activationsArray)
            index += 1
        return endlayer

    def ResNet50(self):
        input = Input(shape = (self.height, self.width, self.channels), name = "input")
        endlayer = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same")(input)
        endlayer = BatchNormalization()(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)
        endlayer = self.ConvBlock(endlayer, self.block1ChannelsArray, self.block1KernelArray, self.block1ActivationArray, 3)
        endlayer = self.ConvBlock(endlayer, self.block2ChannelsArray, self.block2KernelArray, self.block2ActivationArray, 4)
        endlayer = self.ConvBlock(endlayer, self.block3ChannelsArray, self.block3KernelArray, self.block3ActivationArray, 6)
        endlayer = self.ConvBlock(endlayer, self.block4ChannelsArray, self.block4KernelArray, self.block4ActivationArray, 3)
        endlayer = GlobalAveragePooling2D()(endlayer)
        endlayer = Dense(self.classesNumber)(endlayer) 
        endlayer = Activation("softmax")(endlayer)
        self.model = keras.Model(input, endlayer)
        return

    def ResNet101(self):
        input = Input(shape = (self.height, self.width, self.channels), name = "input")
        endlayer = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same")(input)
        endlayer = BatchNormalization()(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)
        endlayer = self.ConvBlock(endlayer, self.block1ChannelsArray, self.block1KernelArray, self.block1ActivationArray, 3)
        endlayer = self.ConvBlock(endlayer, self.block2ChannelsArray, self.block2KernelArray, self.block2ActivationArray, 4)
        endlayer = self.ConvBlock(endlayer, self.block3ChannelsArray, self.block3KernelArray, self.block3ActivationArray, 23)
        endlayer = self.ConvBlock(endlayer, self.block4ChannelsArray, self.block4KernelArray, self.block4ActivationArray, 3)
        endlayer = GlobalAveragePooling2D()(endlayer)
        endlayer = Dense(self.classesNumber)(endlayer) 
        endlayer = Activation("softmax")(endlayer)
        self.model = keras.Model(input, endlayer)
        return

    def ResNet152(self):
        input = Input(shape = (self.height, self.width, self.channels), name = "input")
        endlayer = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same")(input)
        endlayer = BatchNormalization()(endlayer)
        endlayer = Activation("relu")(endlayer)
        endlayer = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same", data_format = None)(endlayer)
        endlayer = self.ConvBlock(endlayer, self.block1ChannelsArray, self.block1KernelArray, self.block1ActivationArray, 3)
        endlayer = self.ConvBlock(endlayer, self.block2ChannelsArray, self.block2KernelArray, self.block2ActivationArray, 8)
        endlayer = self.ConvBlock(endlayer, self.block3ChannelsArray, self.block3KernelArray, self.block3ActivationArray, 36)
        endlayer = self.ConvBlock(endlayer, self.block4ChannelsArray, self.block4KernelArray, self.block4ActivationArray, 3)
        endlayer = GlobalAveragePooling2D()(endlayer)
        endlayer = Dense(self.classesNumber)(endlayer) 
        endlayer = Activation("softmax")(endlayer)
        self.model = keras.Model(input, endlayer)
        return

    def GetModel(self):
        return self.model

    def Summary(self):
        if self.model == None:
            print("Model is not define")
        else:
            self.model.summary()
        return

if __name__ == "__main__":
    classes = 2
    inputShape = (224, 224, 3)
    model = ResNet("resnet50", inputShape, classes)
    model.Summary()
    model.Save("resnet50")