import tensorflow as tf
import emnist
import numpy as np
import matplotlib.pyplot as plt

#extracting samples from the EMNIST dataset
print('Extracting samples from the EMNIST dataset...')
trainImages, trainLabels = emnist.extract_training_samples('byclass')
testImages, testLabels = emnist.extract_test_samples('byclass')

#data normalization
print('Normalizing data...')
trainImages = tf.keras.utils.normalize(trainImages, axis=1)
testImages = tf.keras.utils.normalize(testImages, axis=1)
trainImages = np.expand_dims(trainImages, axis=3)
testImages = np.expand_dims(testImages, axis=3)

#data augmentation
rotationRange = 15
widthShiftRange = 0.1
heightShiftRange = 0.1
trainDataGen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = rotationRange,
															   width_shift_range = widthShiftRange,
															   height_shift_range = heightShiftRange)

#augmentation test
# num_row = 4
# num_col = 8
# num= num_row*num_col
# fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
# for i in range(num):
#      ax = axes1[i//num_col, i%num_col]
#      ax.imshow(trainImages[i], cmap='gray_r')
#      ax.set_title('Label: {}'.format(trainLabels[i]))
# plt.tight_layout()
# fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
# for X, Y in trainDataGen.flow(trainImages.reshape(trainImages.shape[0], 28, 28, 1),trainLabels.reshape(trainLabels.shape[0], 1),batch_size=num,shuffle=False):
#      for i in range(0, num):
#           ax = axes2[i//num_col, i%num_col]
#           ax.imshow(X[i].reshape(28,28), cmap='gray_r')
#           ax.set_title('Label: {}'.format(int(Y[i])))
#      break
# plt.tight_layout()
# plt.show()

valDataGen = tf.keras.preprocessing.image.ImageDataGenerator()
valDataGen.fit(testImages.reshape(testImages.shape[0], 28, 28, 1))

#convolutional neural network model definition
def cnn_model() -> tf.keras.Model:
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(2,2))
	model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(2,2))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(62, activation = 'softmax'))
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model

model = cnn_model()
model.summary()

history = model.fit(trainDataGen.flow(trainImages, trainLabels, batch_size=1024),
         validation_data= valDataGen.flow(testImages, testLabels,
         batch_size=32), epochs=20)

scores = model.evaluate(testImages, testLabels)

plt.figure(1), plt.grid(), plt.title('Model Accuracy'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='Training'), plt.plot(history.history['val_accuracy'], label='Test'), plt.legend(loc='lower right'), plt.savefig('model_accuracy.eps')
plt.figure(2), plt.grid(), plt.title('Model Loss'), plt.xlabel('Epoch'), plt.ylabel('Loss')
plt.plot(history.history['loss'], label='Training'), plt.plot(history.history['val_loss'], label='Test'), plt.legend(loc='upper right'), plt.savefig('model_loss.eps')

model.save('emnist_model.h5')