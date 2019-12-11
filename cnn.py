# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

class BinaryCNN():
    def __init__(self):
        # Initialization
        self.classifier = Sequential()

        # First Convolutional Layer
        self.classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Second Convolutional Layer
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Third Convolutional Layer
        self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Flattening
        self.classifier.add(Flatten())

        # Dense
        self.classifier.add(Dense(units = 128, activation = 'relu'))
        self.classifier.add(Dense(units = 2, activation = 'softmax'))
        
        # Compiling
        self.classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    