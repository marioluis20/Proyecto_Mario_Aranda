from openai import OpenAI
import os
import uuid

# Ruta absoluta unificada (forzada)
BASE_PODCAST_PATH = r"C:\Documentos\EduCast\Frontend\mini-podcast"

def tema_to_speach(tema_podcast):
    # Asegurar que la carpeta existe
    os.makedirs(BASE_PODCAST_PATH, exist_ok=True)

    # Leer la API key
    with open(r"C:\Documentos\EduCast\Frontend\apikey.txt", "r") as api:
        apikey = api.read().strip()

    client = OpenAI(api_key=apikey)

    tema = tema_podcast.strip()
    tema_sanitizado = tema.replace(' ', '_')
    nombre_podcast = f"mini_podcast_{tema_sanitizado}.mp3"

    print(f"\n🎙️ Generando texto para el tema: {tema}")
    response = client.responses.create(
        prompt={
            "id": "pmpt_6900452e683c81968968095b63314537048297b09036aa5a",
            "version": "4",
            "variables": {"tema": f"{tema}"}
        }
    )

    texto_generado = response.output_text
    print("\n🧠 Texto generado por la IA:\n")
    print(texto_generado)

    print("\n🔊 Convirtiendo texto a voz...")
    speech_response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        instructions="Habla en un tono alegre y entusiasta como si hablaras con niños.",
        input=texto_generado
    )

    path_podcast = os.path.join(BASE_PODCAST_PATH, nombre_podcast)

    # Guardar el archivo de audio en la ruta absoluta especificada
    with open(path_podcast, "wb") as f:
        f.write(speech_response.read())

    print(f"\n✅ Audio guardado en: {path_podcast}\n")

    return path_podcast
