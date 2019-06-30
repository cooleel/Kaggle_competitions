from keras.datasets import mnist

#import dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#check dataset dimension
train_images.shape
len(train_labels)
train_labels[:10]

test_images.shape
len(test_labels)
test_labels[:10]


#preprocssing data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') /255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') /255

#preparing the labels
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#Network arthitecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
network.add(layers.Dense(10, activation = 'softmax'))

#the compilation step
'''
a loss function
an optimizer
metrics to monitor during training and testing
'''
network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])


#training demo
network.fit(train_images, train_labels, epochs=5, batch_size=128)

'''
Display loss and accuracy on training data
60000/60000 [==============================] - 3s 52us/step - loss: 0.2596 - acc: 0.9247
Epoch 2/5
60000/60000 [==============================] - 1s 19us/step - loss: 0.1050 - acc: 0.9688
Epoch 3/5
60000/60000 [==============================] - 1s 19us/step - loss: 0.0694 - acc: 0.9790
Epoch 4/5
60000/60000 [==============================] - 1s 18us/step - loss: 0.0499 - acc: 0.9853
Epoch 5/5
60000/60000 [==============================] - 1s 18us/step - loss: 0.0373 - acc: 0.9886
'''

#model perform on test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss',test_loss)
print('test_acc', test_acc)

#reached test accuracy at 0.977


'''
Reference:
1. Kaggle
2. Deep learning with python
'''


