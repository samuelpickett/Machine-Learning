from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.datasets import cifar10
from keras.api.optimizers import SGD
from keras.api.utils import to_categorical
from keras.api.constraints import max_norm
labels = ["airplane", " automobile", "bird", " cat", "deer", "dog", "frog", "horse", "ship", "truck"]
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Create the model as a sequential type so we can add layers in order
model = Sequential()
# Add the first convolution to output a feature map
# filters: output 32 kernels
# kernel_size: 3x3 kernel or filter matrix used to calculate output features
# input_shape: each image is 32x32x3
# activation: relu activation for each of the operations as it produces the best results
# padding: "same" adds padding to the input image to make sure that the output feature map is the same siza as the input
# kernel_constraing: maxnorm normalizes the values in the kernel to make sure that the max value is 3
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape = (32, 32, 3), activation="relu", padding="same", kernel_constraint = max_norm(3)))
# Add the max pool layer to decrease the image size from 32x32 to 16x16
# pool_size: finds the max value in each 2x2 section of the input
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten layer converts a matrix into a 1D array
model.add(Flatten())
# First dense layer to creat the actual prediction network
# units: 512 neurons at this layer, increase for greater accuracy, decrease for faster train speed
# activation: relu because it works so well
# kernel_contraint: see above
model.add(Dense(units=512, activation="relu", kernel_constraint=max_norm(3)))
# Dropout layer to ignore some neurons during training which improves model reliability
# rate: 0.5 means half neurons dropped
model.add(Dropout(rate=0.5))
# Final dense layer used to produce output for each of the 10 categories
# units: 10 categories so 10 output units
# activation: softmax because we are calculating probabilities for each of the 10 categories (not as clear as 0 or 1)
model.add(Dense(units=10, activation="softmax"))

model.compile(optimizer=SGD(learning_rate=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x=train_images, y=train_labels, epochs=50, batch_size=32)

model.save(filepath="image_classifier.keras")