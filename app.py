from flask import Flask, request
from predict import RED
app = Flask(__name__)
#instanciamos el objeto RED
redn = RED()

@app.route("/upload/<string:file>")
#se pasa por parametros la direccion de la imagen
def upload(file):
    prediccion = redn.predict(file)
    #prediccion contiene el json a retornar
    return prediccion    
if __name__ == "__main__":
    #corremos nuetro servidors
    app.run(debug=True,port=5000)
