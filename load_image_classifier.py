import keras
import numpy as np
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical
from keras.api.models import load_model


labels = ["airplane", " automobile", "bird", " cat", "deer", "dog", "frog", "horse", "ship", "truck"]
(_, _), (test_images, test_labels) = cifar10.load_data()
test_labels = to_categorical(test_labels)

test_images = test_images.astype("float32") / 255.0

model = load_model("C:\Sam's Code\Machine Learning\image_classifier.keras")
# results = keras.models.Model.evaluate(self=model, x=test_images, y=test_labels)
# print("Test loss: ", results[0])
# print("Test accuracy: ", results[1])
test_image_data = np.asarray([test_images[0]])


prediction = keras.models.Model.predict(self=model, x=test_image_data)
max_index = np.argmax(prediction[0])
print("Prediction: ", labels[max_index])