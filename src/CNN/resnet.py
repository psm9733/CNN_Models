from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Add, Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D
import tensorflow as tf
import os
import matplotlib.pyplot as plt

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
batch_size = 200
epochs = 1000
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=.15, height_shift_range=.15, horizontal_flip=True, zoom_range=0.5) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')

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
            self.model = self.ResNet50()
        elif modelName.lower() == "resnet101":
            self.model = self.ResNet101()
        elif modelName.lower() == "resnet152":
            self.model = self.ResNet101()
        else:
            self.model = None

        self.model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit_generator(train_data_gen, steps_per_epoch = total_train // batch_size, epochs = epochs, validation_data = val_data_gen, validation_steps = total_val // batch_size)
        print("-- Evaluate --")
        scores = self.model.evaluate_generator(val_data_gen, steps = 10)
        print("%s: %.2f%%" %(self.model.metrics_names[1], scores[1] * 100))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
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
        model = keras.Model(input, endlayer)
        return model

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
        model = keras.Model(input, endlayer)
        return model

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
        model = keras.Model(input, endlayer)
        return model

    def GetModel(self):
        return self.model

    def Save(self, path):
        print("---model saved---")
        self.model.save(path + '.h5')
        return

    def Summary(self):
        if self.model == None:
            print("Model is not define")
        else:
            self.model.summary()
        return

if __name__ == "__main__":
    classes_number = 2
    inputShape = (150, 150, 3)
    model = ResNet("resnet50", inputShape, classes_number)
    # model.Summary()
    model.Save("resnet50")