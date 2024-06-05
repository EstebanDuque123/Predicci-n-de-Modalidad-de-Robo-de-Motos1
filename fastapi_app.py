from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd

# 1. Cargar el modelo entrenado
model = load("modelo_robos_motos1.joblib")

# 2. Cargar los encoders
le_MUNICIPIO = load("le_MUNICIPIO.joblib")
le_FECHA_HECHO = load("le_FECHA_HECHO.joblib")
le_ARMAS_MEDIOS = load("le_ARMAS_MEDIOS.joblib")

# 3. Definir el formato de los datos de entrada
class DatosEntrada(BaseModel):
    MUNICIPIO: str
    FECHA_HECHO: str

# 4. Crear la aplicación FastAPI
app = FastAPI()

# 5. Crear un endpoint para hacer predicciones
@app.post("/predecir_modalidad_robo1/")
async def predecir_modalidad(datos_entrada: DatosEntrada):
    # Transformamos la entrada a un DataFrame
    datos_nuevos = pd.DataFrame([datos_entrada.dict()])

    # Verificar y codificar MUNICIPIO
    if datos_nuevos['MUNICIPIO'][0] not in le_MUNICIPIO.classes_:
        raise HTTPException(status_code=400, detail=f"MUNICIPIO desconocido: {datos_nuevos['MUNICIPIO'][0]}")
    datos_nuevos['MUNICIPIO_Encoded'] = le_MUNICIPIO.transform(datos_nuevos['MUNICIPIO'])

    # Verificar y codificar FECHA_HECHO
    if datos_nuevos['FECHA_HECHO'][0] not in le_FECHA_HECHO.classes_:
        raise HTTPException(status_code=400, detail=f"FECHA_HECHO desconocido: {datos_nuevos['FECHA_HECHO'][0]}")
    datos_nuevos['FECHAHECHO_Encoded'] = le_FECHA_HECHO.transform(datos_nuevos['FECHA_HECHO'])

    # Realizamos la predicción
    prediccion = model.predict(datos_nuevos[['MUNICIPIO_Encoded', 'FECHAHECHO_Encoded']])

    # Decodificamos la predicción para obtener la modalidad en texto
    modalidad_predicha = le_ARMAS_MEDIOS.inverse_transform(prediccion)[0]

    # Retornamos la predicción
    return {"Modalidad_Predicha": modalidad_predicha}

print("Se abrio la api")

# Para ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())