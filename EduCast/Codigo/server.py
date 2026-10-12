from flask import Flask, request, jsonify, send_from_directory
from prompting import tema_to_speach
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Carpeta donde se guardarán los podcasts (debe coincidir con prompting.py)
CARPETA_PODCASTS = r"C:\Documentos\EduCast\Frontend\mini-podcast"
os.makedirs(CARPETA_PODCASTS, exist_ok=True)

@app.route('/api/generar', methods=['POST'])
def generar():
    data = request.get_json()
    tema = data.get('tema')

    if not tema:
        return jsonify({"error": "Falta el tema"}), 400

    try:
        print(f"\n🎧 Solicitando generación para el tema: {tema}")
        path_audio = tema_to_speach(tema)
        filename = os.path.basename(path_audio)

        # Verificar que el archivo existe justo antes de responder
        if not os.path.isfile(path_audio):
            raise FileNotFoundError(f"Archivo no encontrado después de generar: {path_audio}")

        print(f"✅ Podcast generado: {filename} (en {CARPETA_PODCASTS})")
        return jsonify({
            "mensaje": "Audio generado correctamente",
            "ruta": f"mini-podcast/{filename}"
        })
    except Exception as e:
        print(f"❌ Error al generar el audio: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/biblioteca', methods=['GET'])
def biblioteca():
    try:
        archivos = [f for f in os.listdir(CARPETA_PODCASTS) if f.endswith(".mp3")]
        return jsonify({"podcasts": archivos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Servir los audios desde la carpeta absoluta
@app.route('/mini-podcast/<path:filename>')
def servir_audio(filename):
    print(f"🎵 Solicitando audio: {filename}")
    return send_from_directory(CARPETA_PODCASTS, filename, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
