# check_models.py
import os
import shutil

# Configuración
BACKEND_PATH = r"C:\Documentos\LifeScan\Backend"
MODEL_FILES = ['stroke_model.pkl', 'heart_model.pkl']

print("=" * 50)
print("VERIFICADOR DE MODELOS")
print("=" * 50)

print(f"Directorio objetivo: {BACKEND_PATH}")

# Verificar si el directorio existe
if not os.path.exists(BACKEND_PATH):
    print(f"❌ El directorio no existe. Creando...")
    os.makedirs(BACKEND_PATH, exist_ok=True)

# Listar archivos actuales
print(f"\n📁 Archivos en {BACKEND_PATH}:")
files = os.listdir(BACKEND_PATH)
for file in files:
    print(f"  - {file}")

# Verificar modelos
print(f"\n🔍 Buscando modelos:")
for model in MODEL_FILES:
    model_path = os.path.join(BACKEND_PATH, model)
    if os.path.exists(model_path):
        print(f"  ✅ {model} - ENCONTRADO")
    else:
        print(f"  ❌ {model} - NO ENCONTRADO")

# Preguntar al usuario
print(f"\n¿Qué quieres hacer?")
print("1. Copiar modelos desde otra ubicación")
print("2. Crear modelos dummy para pruebas")
print("3. Solo verificar")

option = input("\nSelecciona una opción (1-3): ")

if option == "1":
    # Copiar modelos
    for model in MODEL_FILES:
        source = input(f"\nRuta completa de {model}: ").strip()
        if os.path.exists(source):
            destination = os.path.join(BACKEND_PATH, model)
            shutil.copy2(source, destination)
            print(f"  ✅ {model} copiado a {destination}")
        else:
            print(f"  ❌ Archivo no encontrado: {source}")

elif option == "2":
    # Crear modelos dummy
    print("\nCreando modelos dummy...")
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np
    
    # Crear modelo dummy
    dummy_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Datos dummy
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    dummy_model.fit(X_dummy, y_dummy)
    
    # Guardar
    for model in MODEL_FILES:
        model_path = os.path.join(BACKEND_PATH, model)
        joblib.dump(dummy_model, model_path)
        print(f"  ✅ {model} creado como modelo dummy")

print("\n" + "=" * 50)
print("VERIFICACIÓN FINAL")
print("=" * 50)

for model in MODEL_FILES:
    model_path = os.path.join(BACKEND_PATH, model)
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024  # KB
        print(f"✅ {model} - {size:.1f} KB")
    else:
        print(f"❌ {model} - NO EXISTE")

print("\n🎯 Ahora puedes ejecutar: python app.py")
input("Presiona Enter para salir...")