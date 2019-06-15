import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras import backend as k
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential, Model
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import itertools
%matplotlib inline
import os

#Train and Validation data Paths
train_file = 'Skin_lesions/Train'
test_file = 'Skin_lesions/Validation'

#Create Train and Test batches
train_batch = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input).\
                flow_from_directory(train_file,
                                    target_size = (224, 224),
                                    batch_size = 20)
valid_batch = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input).\
                flow_from_directory(test_file,
                                    target_size = (224, 224),
                                    batch_size = 20)

#To do Transfer Learnig using MobileNet model, import the model from Keras appplications and Fine Tune the model 
#accordign to our need
mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()

#We donot need the MobileNet model as it is. For our application, we are removing the last 6 layers of MobileNet
#and inseting one Dropout layer (to prevent OverFitting) and one Dense layer with softmax activation
x = mobile.layers[-6].output
#Now add one Dopout layer nad one Dense layer
x = Dropout(0.25)(x)
op_layer = Dense(7, activation = 'softmax')(x)
model = Model(input = mobile.input, output = op_layer)
model.summary()
#We donot need to trai all the layers of hte model. Just change the weights of last few layers nad update their weights
for layer in model.layers[:-10]:
    layer.trainable = False
#To save the best weights and have dynamic learning rate, use the callbacks - ModelCheckpoint and ReduceLROnPlateau
filepath = "model.h5"
# Declare a checkpoint to save the best version of the model
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                             save_best_only=True, mode='max')
# Reduce the learning rate as the learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]

#Compile and train the model
model.compile(Adam(lr = 0.5), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_batch, steps_per_epoch = len(train_batch), validation_data = valid_batch,
                    validation_steps = len(valid_batch), epochs = 5, verbose = 2, callbacks=callbacks_list)
 
predictions = model.predict_generator(valid_batch, steps = len(valid_batch), verbose = 0)

#Plot Confusion Matrix to analyze the accuracy
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

test_img, test_labels = next(test_batch)

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

plot_confusion_matrix(cm, cm_plot_labels)
