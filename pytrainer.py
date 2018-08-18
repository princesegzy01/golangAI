import tensorflow as tf
from keras  import backend as K


#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers.core import K
K._LEARNING_PHASE = tf.constant(0)

sess = tf.Session()
K.set_session(sess)


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Activation
import time
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import zipfile


def trainCurrencyDetector():
    #Initializing the CNN
    clf = Sequential()

    #Step 1 : Add the first convolutional layer
    clf.add(Conv2D(256, (3,3), input_shape= (128,128,3), activation = 'relu', name="inputNode"))

    #Step 2 : Pooling using Max pooling
    clf.add(MaxPooling2D( pool_size = (2,2)))
    
    #Step 1 : Add the first convolutional layer
    clf.add(Conv2D(128, (3,3), input_shape= (128,128,3), activation = 'relu'))
    #Step 2 : Pooling using Max pooling
    clf.add(MaxPooling2D( pool_size = (2,2)))

    #Step 1 : Add the first convolutional layer
    clf.add(Conv2D(64, (3,3), input_shape= (128,128,3), activation = 'relu'))
    #Step 2 : Pooling using Max pooling
    clf.add(MaxPooling2D( pool_size = (2,2)))


    # Flatten our images to vector
    clf.add(Flatten())

    #Fully connected hidden layer
    clf.add(Dense(units = 256, activation = 'relu'))

    #Fully connected hidden layer
    clf.add(Dense(units = 128, activation = 'relu'))

    #Fully connected hidden layer
    clf.add(Dense(units = 64, activation = 'relu'))

    #Our ouput layer
    clf.add(Dense(units = 3, activation = 'softmax', name="inferNode"))
    #classifier.add(Dense(units = 5))
    #classifier.add(Activation("softmax"))

    #compile out cnn
    clf.compile(optimizer='adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    #fit our model to training set


    #using imageDataGenerator to preprocess our data
    from keras.preprocessing.image import ImageDataGenerator


    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('dataset/training_set',target_size=(128, 128), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('dataset/test_set', target_size=(128, 128), batch_size=32, class_mode='categorical')

    clf.fit_generator(train_generator, steps_per_epoch=5, epochs=1, validation_data=test_generator, validation_steps=100)
    #classifier.fit_generator(train_generator, steps_per_epoch=90, epochs=3)
    
    clf.save("currency_detector.h5")

    for n in sess.graph.as_graph_def().node:
        #print(n.name)
        if "input_node" in n.name:
            print(n.name)

    builder = tf.saved_model.builder.SavedModelBuilder("forGo2")
    builder.add_meta_graph_and_variables(sess, ["tags"])
    builder.save()
    sess.close()

    print("Successfully Trained Currency Detector")
    

trainCurrencyDetector()
#print("Done Training Currency")

# def predictCurrency(train_data, test_image):
    
#     #train_data = "train_data.ell"
#     currency_data_file_name, _ = utils.returnTrainDataFIleName(train_data)
#     clf = load_model(currency_data_file_name)

#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis = 0)
#     className = clf.predict(test_image)

#     return className
