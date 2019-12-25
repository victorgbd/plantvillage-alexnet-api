from flask import Flask, request
from predict import RED
#instanciamos el objeto Flask
app = Flask(__name__)
#instanciamos el objeto RED
redn = RED()

@app.route("/upload/<string:file>")
#se pasa por parametros la direccion de la imagen
def upload(file):
    prediccion = redn.predict(file)
    #prediccion contiene el json a retornar
    return prediccion

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        filename = f.filename
        print(f)
        print(filename)
        prediccion = redn.predict(filename)
        return prediccion

    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Upload File</title>
</head>
<body>
    <h1>Upload File</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
</body>
</html>"""
if __name__ == "__main__":
    #corremos nuetro servidor
    app.run(debug=True,port=5000)
