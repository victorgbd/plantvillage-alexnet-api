import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import backend as K
from keras import layers
K.clear_session()



data_entrenamiento = './data/train'
data_validacion = './data/valid'

"""
Parameters
"""
epocas=25
longitud, altura = 224, 224
batch_size = 128
filtrosConv1 = 96
filtrosConv25 = 256
filtrosConv34 = 384
tamano_pool = (2, 2)
clases = 3
lr = 0.001


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
#directorios de las clases
#clase_direct = entrenamiento_generador.class_indices
#lista de las clase
#li = list(class_dict.keys())
# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(filtrosConv1, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu',trainable=False))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = tamano_pool, strides = (2, 2), padding = 'valid', trainable=False))
classifier.add(BatchNormalization( trainable=False))

# Convolution Step 2
classifier.add(Convolution2D(filtrosConv25, 11, strides = (1, 1), padding='valid', activation = 'relu', trainable=False))

# Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = tamano_pool, strides = (2, 2), padding='valid', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Convolution Step 3
classifier.add(Convolution2D(filtrosConv34, 3, strides = (1, 1), padding='valid', activation = 'relu', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Convolution Step 4
classifier.add(Convolution2D(filtrosConv34, 3, strides = (1, 1), padding='valid', activation = 'relu', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Convolution Step 5
classifier.add(Convolution2D(filtrosConv25, 3, strides=(1, 1), padding='valid', activation = 'relu', trainable=False))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = tamano_pool, strides = (2, 2), padding = 'valid', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Flattening Step
classifier.add(Flatten(trainable=False))

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu', trainable=False))
classifier.add(Dropout(0.4, trainable=False))
classifier.add(BatchNormalization(trainable=False))
classifier.add(Dense(units = 4096, activation = 'relu', trainable=False))
classifier.add(Dropout(0.4, trainable=False))
classifier.add(BatchNormalization(trainable=False))
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = clases, activation = 'softmax'))

for layer in enumerate(classifier.layers):
    print(layer)
print (classifier.summary())
classifier.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=lr,momentum=0.9, decay=0.005),
            metrics=['accuracy'])

train_num = entrenamiento_generador.samples
valid_num = validacion_generador.samples

print(train_num,valid_num)
classifier.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=train_num//batch_size,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=valid_num//batch_size)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./modelo/modelo.h5')
classifier.save_weights('./modelo/pesos.h5')