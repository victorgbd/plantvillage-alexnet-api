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
#limpiamos cualquier sesión que haya abierte en keras
K.clear_session()



#La direccion de nuestro dataset de entrenamiento y de validacion
data_entrenamiento = './data/train'
data_validacion = './data/valid'

"""
Parametros
"""
epocas=25
#tamaño de las imagenes
longitud, altura = 224, 224

#tamaño de la cantidad de ejemplos que se evaluaran por epoca
batch_size = 128

#cantidad de filtros para cada capa
filtrosConv1 = 96
filtrosConv25 = 256
filtrosConv34 = 384

#tamaño de la ventana de max pooling 
tamano_pool = (2, 2)

#el numero de clases que deseamos predecir o nuestros targets
clases = 3
#el learning rate o ratio de aprendizaje
lr = 0.001


#Preparamos nuestras imagenes

"""
rescalamos las imagenes a 255 pixeles
aplicamos ciertos movimientos y zooms en las imagenes para lograr que 
no solo aprenda que nuestro target este en un solo lugar de la imagen
"""
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)
#aqui especificamos donde esta nuestra data, el tamaño de esta, el batch size y como se clasificara nuestra data
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
#lo mismo para la data de validación o test
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
#directorios de las clases
#nos servirá para identificar las predicciones que nos da el modelo ya entrenado(utilizar en un notebook como recomendacion)
"""
clase_direct = entrenamiento_generador.class_indices
lista de las clase
li = list(class_dict.keys())
"""
# inicializando la CNN
classifier = Sequential()

# Capa de convolucion Paso 1
classifier.add(Convolution2D(filtrosConv1, 11, strides = (4, 4), padding = 'valid', input_shape=(longitud, altura, 3), activation = 'relu',trainable=False))

# Capa de  Max Pooling  Paso 1
classifier.add(MaxPooling2D(pool_size = tamano_pool, strides = (2, 2), padding = 'valid', trainable=False))
classifier.add(BatchNormalization( trainable=False))

# Capa de convolucion Paso 2
classifier.add(Convolution2D(filtrosConv25, 11, strides = (1, 1), padding='valid', activation = 'relu', trainable=False))

# Capa de  Max Pooling  Paso 2
classifier.add(MaxPooling2D(pool_size = tamano_pool, strides = (2, 2), padding='valid', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Capa de convolucion Paso 3
classifier.add(Convolution2D(filtrosConv34, 3, strides = (1, 1), padding='valid', activation = 'relu', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Capa de convolucion Paso 4
classifier.add(Convolution2D(filtrosConv34, 3, strides = (1, 1), padding='valid', activation = 'relu', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Capa de convolucion Paso 5
classifier.add(Convolution2D(filtrosConv25, 3, strides=(1, 1), padding='valid', activation = 'relu', trainable=False))

# Capa de  Max Pooling  Paso 3
classifier.add(MaxPooling2D(pool_size = tamano_pool, strides = (2, 2), padding = 'valid', trainable=False))
classifier.add(BatchNormalization(trainable=False))

# Paso de aplastamiento o flattening
classifier.add(Flatten(trainable=False))

# Paso de Full Connection 
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

#se muestran cada una de las capas y la cantidad de parametros que pasaran cada una de estas capas
for layer in enumerate(classifier.layers):
    print(layer)
print (classifier.summary())

"""
se compila la nuestro modelo especificando nuestra funcion de coste, 
el optimizador y sus hiperparametros y la metrica como se evaluará nuestro coste
"""
classifier.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=lr,momentum=0.9, decay=0.005),
            metrics=['accuracy'])

#esta es la cantidad de ejemplos que tenemos en nuestra data de entrenamiento y validación
train_num = entrenamiento_generador.samples
valid_num = validacion_generador.samples
print(train_num,valid_num)

#entrenamos el modelos
classifier.fit_generator(
    #data de entrenamiento
    entrenamiento_generador,
    #los pasos que dara por cada epoca
    #es la division floor de el numero de los ejemplos y el batch_size especificado
    steps_per_epoch=train_num//batch_size,
    #el numero de epocas o iteraciones
    epochs=epocas,
    #data de validacion o test
    validation_data=validacion_generador,
    #los pasos que dara por cada epoca
    #es la division floor de el numero de los ejemplos y el batch_size especificado
    validation_steps=valid_num//batch_size)

#directorio donde se guardara el modelo y los pesos
target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
classifier.save('./modelo/modelo.h5')
classifier.save_weights('./modelo/pesos.h5')
