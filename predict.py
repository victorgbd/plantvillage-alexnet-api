import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf
import json
#traemos la sesión del grafo computacional predeterminda y lo igualamos a graph
graph = tf.get_default_graph()

class RED:
    def __init__(self):
      #tamaño de las imagenes
      self.longitud, self.altura = 224, 224
        
      #cargamos nuestro modelo entrenado desde una dirección
      modelo = './modelo/modelo.hdf5'
      pesos_modelo = './modelo/pesos.hdf5'
      self.cnn = load_model(modelo)
      self.cnn.load_weights(pesos_modelo)
    
    #predict recibe por parametro una imagen
    def predict(self, file):
      #cargamos la imagen y la ajustamos al tamaño que recibe nuestro modelo
      x = load_img(file, target_size=(self.longitud, self.altura))
      #la convertimos a array
      x = img_to_array(x)
      #agregamos otra dimension
      x = np.expand_dims(x, axis=0)
      x = x/255
      #y dividimos en la cantidad de pixeles
        
      #especificamos que queremos que se ejecute como predeterminado nuestra sesión
      with graph.as_default():
        #pasamos nuestra imagen a la funcion predict de nuestro modelo e igualamos las predicciones en una lista
        array = self.cnn.predict(x)
        
      #la predicciones se encuentran en la posicion 0
      result = array[0]
      #elegimos un top 3 de los indices de mayor probabilidad
      indexs=result.argsort()[-3:][::-1]
      aux=[]
      index=[]
      for i,e in enumerate(indexs):
        for t,j in enumerate(result):
          if e == t:
            j = j*100 #la probabilidad
            if(j>0.009):
              j=format(j, '.3f')
              aux.append(str(j))#agregamos a aux las probabilidades
              index.append(e)#agregamos a index los indices de mayor probabilidad
            
      #las clases que tiene nuestro modelo                
      li=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'
      , 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy'
      , 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_'
      , 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot'
      , 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy'
      , 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy'
      , 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight'
      , 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy'
      , 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy'
      , 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold'
      , 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot'
      , 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
      
      lista=[]
      for i,e in enumerate(index):
        for t,j in enumerate(li):
          if e == t:
            lista.append(j)#agregamos la clase a la que pertenecen nuestros indices de mayor probabilidad
      #creamos una variable diccionario que contendra la clase y la probabildad que dio 
      j_son=dict(zip(lista,aux)) 
      #la retornamos como un json
      return json.dumps(j_son)
