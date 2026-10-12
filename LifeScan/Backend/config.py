# config.py - Configuración del sistema
import os

def load_api_key():
    """Cargar API Key desde archivo"""
    key_path = r"C:\Documentos\LifeScan\Backend\apikey.txt"
    
    try:
        if os.path.exists(key_path):
            with open(key_path, 'r') as f:
                api_key = f.read().strip()
                
            if api_key:
                print("✅ API Key cargada correctamente")
                return api_key
            else:
                print("⚠️ Archivo API Key vacío")
                return None
        else:
            print(f"❌ Archivo API Key no encontrado en: {key_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error cargando API Key: {e}")
        return None

# Configuración global
API_KEY = load_api_key()
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Verificar API Key
if not API_KEY:
    print("⚠️ ADVERTENCIA: No se pudo cargar la API Key")
    print("   El chat funcionará en modo demo limitado")