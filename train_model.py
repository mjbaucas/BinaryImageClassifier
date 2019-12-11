from cnn import BinaryCNN
from datagen import ImageDataGen

test_set_dir = 'dataset/test_set'
training_set_dir = 'dataset/training_set'

# Initialize Classes
conv_nn = BinaryCNN()
img_datagen = ImageDataGen(training_set_dir, test_set_dir)

# Fitting
conv_nn.classifier.fit_generator(img_datagen.training_set,
                         steps_per_epoch = 250,
                         epochs = 30,
                         validation_data = img_datagen.test_set,
                         validation_steps = 128)

conv_nn.classifier.save("test_model1.h5")