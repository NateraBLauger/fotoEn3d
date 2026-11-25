import io
import os
import tempfile
import cv2
from flask import Flask, request, send_file

from foto3d import PRESETS, Foto3D, Parametros3D, elegir_preset

app = Flask(__name__)


@app.route("/", methods=["GET"])
def inicio():
    opciones = "".join(f'<option value="{p}">{p}</option>' for p in PRESETS)
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Foto a 3D</title>
        <style>
            body {{ font-family: sans-serif; max-width: 500px; margin: 40px auto; padding: 20px; }}
            .campo {{ margin-bottom: 15px; }}
            button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h2>Convertidor de Foto a 3D</h2>
        <form action="/procesar" method="post" enctype="multipart/form-data">
            <div class="campo">
                <label>Sube tu imagen:</label><br>
                <input type="file" name="imagen" accept="image/*" required>
            </div>
            <div class="campo">
                <label>Estilo (Preset):</label><br>
                <select name="preset">
                    <option value="auto">auto (recomendado)</option>
                    {opciones}
                </select>
            </div>
            <div class="campo">
                <label><input type="checkbox" name="marco"> Agregar marco de madera</label><br>
                <label><input type="checkbox" name="objetos"> Separar por objetos</label>
            </div>
            <button type="submit">Generar imagen 3D</button>
        </form>
    </body>
    </html>
    """


@app.route("/procesar", methods=["POST"])
def procesar():
    if "imagen" not in request.files:
        return "No se envió ninguna imagen", 400

    archivo = request.files["imagen"]
    if archivo.filename == "":
        return "Archivo no válido", 400

    preset_elegido = request.form.get("preset", "auto")
    extra = {}
    if "marco" in request.form:
        extra["marco"] = True
    if "objetos" in request.form:
        extra["capas"] = "objetos"

    # Guardamos temporalmente la imagen en /tmp (única carpeta escribible en Vercel)
    ext = os.path.splitext(archivo.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir="/tmp") as temp_in:
        archivo.save(temp_in.name)
        ruta_temporal = temp_in.name

    try:
        nombre = elegir_preset(ruta_temporal) if preset_elegido == "auto" else preset_elegido
        resultado = Foto3D(Parametros3D.preset(nombre, **extra)).procesar(ruta_temporal)

        _, buffer = cv2.imencode(".png", resultado.imagen)
        return send_file(
            io.BytesIO(buffer),
            mimetype="image/png",
            as_attachment=False,
            download_name="imagen_3d.png",
        )
    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)


if __name__ == "__main__":
    app.run(debug=True)
