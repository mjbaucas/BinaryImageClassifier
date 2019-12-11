import numpy as np
from keras.preprocessing import image
from keras.models import load_model

classifier = load_model('test_model1.h5')
test_img = image.load_img('cat_test.jpg', target_size = (64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
prediction = classifier.predict(test_img)

print(prediction)