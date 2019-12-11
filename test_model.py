import numpy as np
from arg_parser import get_arg_parser
from keras.preprocessing import image
from keras.models import load_model

args = get_arg_parser()

classifier = load_model(args.model)
test_img = image.load_img(args.input, target_size = (64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
prediction = classifier.predict(test_img)

print(prediction)