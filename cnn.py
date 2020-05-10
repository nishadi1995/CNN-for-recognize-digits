
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras



num_classes = 10
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))



#adding tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#load mnist data set 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path="mnist.npz")



# normalizing
train_images, test_images = train_images/255.0, test_images/255.0

# reshaping the array to 4-dims so that it can work with the Keras API
x_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
y_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

target_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
model = tf.keras.Sequential() 



# first convolutional layer = 6, 5 x 5 filters with padding 2 stride 1
model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape= (28,28,1)))
# first max pooling layer = 2 x 2 window at stride 2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# second convolutional layer = 16, 5 x 5 filters with padding 2 stride 1
model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu')) 
# second  max pooling layer = 8 x 8 window at stride 2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(2,2)))

# third convolutional layer = 128, 3 x 3 filters with padding 1 stride 1
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))



# flattern the layer
model.add(tf.keras.layers.Flatten())

# fully connected layer = 64 neurons relu activation
model.add(tf.keras.layers.Dense(64, activation='relu')) 

# output layer with softmax activation for the 10 classes
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.summary()



#compile the model using a loss function along with an optimization function (specifing the training config).
model.compile(optimizer= 'adam', loss=tf.keras.losses.sparse_categorical_crossentropy,  metrics=['accuracy'])

# train the model based on the training set through 4 epochs
# and validating(testing) data
history=model.fit(x_images, train_labels, batch_size=128, epochs=4, verbose=1, validation_data=(y_images, test_labels), callbacks=[tensorboard_callback])

print('\nhistory dict:', history.history)



#plt training accuracy
plt.plot(history.history['accuracy'], label=['Accuracy'])
plt.plot(history.history['loss'],label =['Loss'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy/ Loss')
plt.ylim([0.05, 1])
plt.legend(loc='upper left')
plt.show()



# evaluate the model on the test data
test_loss, test_acc = model.evaluate(y_images, test_labels)
print('Test Accuracy',test_acc, 'Test loss', test_loss)

# generate predictions (probabilities -- the output of the last layer) on test data
prediction  = model.predict (y_images)



# checking 10 th test image prediction 
i=10
print (prediction[i])
print (test_labels[i])
t=test_labels[i]
print ("tested", target_labels[t])
print (np.argmax(prediction[i]))
print ("predicted" , target_labels[np.argmax(prediction[i])])



# plot the results for 10 test images
for i in range(10):

    #plt.grid(False)
    plt.figure(figsize= (4,4))
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    t=test_labels[i]
  
    plt.xlabel("Actual : " + target_labels[t])
    plt.title("Predicted :" +  target_labels[np.argmax(prediction[i])])

plt.show()






