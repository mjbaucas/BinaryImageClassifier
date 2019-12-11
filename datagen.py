# For image processing
from keras.preprocessing.image import ImageDataGenerator

class ImageDataGen():
    def __init__(self, training_set_dir, test_set_dir):
        # Generate Image Generator
        self.datagen = ImageDataGenerator(
                        rescale = 1./255,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        horizontal_flip = True)
        
        self.training_set = self.datagen.flow_from_directory(training_set_dir,
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'categorical')
        
        self.test_set = self.datagen.flow_from_directory(test_set_dir,
                                            target_size = (64, 64),
                                            batch_size = 32,
                                                class_mode = 'categorical')


    