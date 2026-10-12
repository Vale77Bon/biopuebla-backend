from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import requests
from fastapi.middleware.cors import CORSMiddleware
from prisma import Prisma
from google import genai
import json
import uuid
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Cargar variables de entorno (para pruebas locales)
load_dotenv()

app = FastAPI(title="BioPuebla API - Reconocimiento de Flora")

# ==========================================
# CONFIGURACIÓN DE CORS (PERMISOS PARA CELULAR)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Prisma()

# ==========================================
# CONFIGURACIÓN DE SUPABASE
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

PLANTNET_API_KEY = "2b10T8PRnlY5i6iUQbGrvfYe"
PLANTNET_URL = f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}&lang=es"

@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()

@app.post("/identificar/")
async def identificar_especie(
    imagen: UploadFile = File(...),
    latitud: float = Form(None),
    longitud: float = Form(None)
):
    contenido_imagen = await imagen.read()

    archivos = [
        ('images', (imagen.filename, contenido_imagen, imagen.content_type))
    ]
    
    datos_formulario = {
        'organs': ['auto']
    }

    try:
        # 1. Analizar con Pl@ntNet
        respuesta = requests.post(PLANTNET_URL, files=archivos, data=datos_formulario)
        datos = respuesta.json()

        if respuesta.status_code == 404 or "not found" in str(datos).lower():
            return {
                "exito": False,
                "mensaje": "No se detectó ninguna planta en la imagen. Asegúrate de enfocar bien las hojas o la flor."
            }

        if respuesta.status_code != 200:
            raise HTTPException(status_code=500, detail="Error en el servidor de reconocimiento.")

        mejor_resultado = datos.get('results', [])[0]
        nombre_cientifico = mejor_resultado['species']['scientificNameWithoutAuthor']
        nombres_comunes = mejor_resultado['species'].get('commonNames', [])
        nombre_comun = nombres_comunes[0] if nombres_comunes else nombre_cientifico
        probabilidad = float(mejor_resultado['score'] * 100)
        
        # 2. Extraer descripción con Inteligencia Artificial
        descripcion = None
        try:
            prompt_botanico = f"Actúa como un botánico experto. Escribe un resumen breve, de máximo 3 líneas y fácil de entender, sobre la planta conocida como {nombre_comun} (nombre científico: {nombre_cientifico}). Menciona algún dato curioso o su cuidado básico."
            
            # Nueva conexión limpia de Gemini
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            respuesta_ia = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt_botanico
            )
            descripcion = respuesta_ia.text.strip()
        except Exception as e:
            print(f"Error con la IA: {e}")
            descripcion = "Descripción biológica no disponible en este momento."

        # 3. Subir la foto a Supabase Storage
        nombre_archivo = f"{uuid.uuid4()}.jpg"
        
        res = supabase.storage.from_("plantas").upload(
            path=nombre_archivo,
            file=contenido_imagen,
            file_options={"content-type": "image/jpeg"}
        )
        
        url_publica = supabase.storage.from_("plantas").get_public_url(nombre_archivo)
        
        # 4. Guardar en Prisma (Base de datos)
        await db.consultaplanta.create(
            data={
                'nombre_cientifico': nombre_cientifico,
                'nombre_comun': nombre_comun,
                'probabilidad': probabilidad,
                'foto_url': url_publica,
                'latitud': latitud,
                'longitud': longitud,
                'descripcion': descripcion
            }
        )
        
        return {
            "exito": True,
            "mensaje": "Especie identificada y guardada",
            "datos": {
                "nombre_cientifico": nombre_cientifico,
                "nombre_comun": nombre_comun.capitalize(),
                "probabilidad": f"{probabilidad:.2f}%",
                "descripcion": descripcion
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/historial/")
async def obtener_historial():
    try:
        historial = await db.consultaplanta.find_many(order={'fecha_escaneo': 'desc'})
        
        lista_historial = []
        for item in historial:
            lista_historial.append({
                "id": item.id,
                "nombre_cientifico": item.nombre_cientifico,
                "nombre_comun": item.nombre_comun,
                "probabilidad": item.probabilidad,
                "fecha_escaneo": item.fecha_escaneo.isoformat() if item.fecha_escaneo else None,
                "foto_url": item.foto_url,
                "latitud": item.latitud,
                "longitud": item.longitud,
                "descripcion": item.descripcion
            })
        return lista_historial
    except Exception as e:
        print(f"Error al cargar el historial: {e}")
        return []