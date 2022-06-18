import tensorflow as tf
import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

#chartab
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
		 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#loading test image filename ('test.png' for default)
if len(sys.argv) > 1:
	filename = sys.argv[1]
else:
	filename = 'test.png'

#loading cnn model
model = tf.keras.models.load_model('emnist_model.h5')
model.summary()

#loading test image
img = cv.bitwise_not(cv.imread(filename, cv.IMREAD_GRAYSCALE))
img = cv.resize(img, (28, 28))
img = np.expand_dims(img, axis=0)
img = tf.keras.utils.normalize(img, axis=1)

# plt.imshow(img, cmap = plt.cm.binary)
# plt.show()

#predicting character
prediction = model.predict(img)
for i in range(len(prediction[0])):
	print("Probability of", CHARS[i], ":", prediction[0][i])
print("Predicted value:", CHARS[np.argmax(prediction[0])])

#plotting prediction probability
plt.plot(prediction[0], 'o'), plt.grid()
plt.xticks(range(len(CHARS)), CHARS)
plt.gca().set_yticks(np.arange(0, 1.1, 0.1))
plt.show()