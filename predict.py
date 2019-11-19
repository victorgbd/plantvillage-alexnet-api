import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import tensorflow as tf
import json
graph = tf.get_default_graph()
class RED:
    def __init__(self):
      self.longitud, self.altura = 224, 224
      modelo = './modelo/AlexNetModel.hdf5'
      pesos_modelo = './modelo/best_weights_9.hdf5'
      self.cnn = load_model(modelo)
      self.cnn.load_weights(pesos_modelo)

    def predict(self, file):
      x = load_img(file, target_size=(self.longitud, self.altura))
      x = img_to_array(x)
      x = np.expand_dims(x, axis=0)
      x = x/255
      with graph.as_default():
        array = self.cnn.predict(x)
      
      result = array[0]
      answer = np.argmax(result)
      print(result)
      print(answer)
      indexs=result.argsort()[-3:][::-1]
      aux=[]
      index=[]
      for i,e in enumerate(indexs):
        for t,j in enumerate(result):
          if e == t:
            j = j*100
            if(j>0.009):
              j=format(j, '.3f')
              aux.append(str(j))
              index.append(e)
            #else:
              #j=format(j, '.3e')  
            
      print(aux)
      print(indexs)            
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
            lista.append(j)
      print(lista)  
      print(li[answer])  
      j_son=dict(zip(lista,aux))      
      return json.dumps(j_son)
