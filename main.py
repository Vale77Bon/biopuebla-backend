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
import time

# Cargar variables de entorno (para pruebas locales)
load_dotenv()

app = FastAPI(title="BioPuebla API - Reconocimiento de Flora y Fauna")

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

# ==========================================
# ENDPOINT 1: FLORA (Pl@ntNet + Caché Gemini)
# ==========================================
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
        
        # ========================================================
        # 2. SISTEMA DE CACHÉ Y EXTRACCIÓN CON IA
        # ========================================================
        descripcion = None
        
        plantas_previas = await db.consultaplanta.find_many(
            where={'nombre_cientifico': nombre_cientifico}
        )
        
        for planta in plantas_previas:
            if planta.descripcion and "no disponible" not in planta.descripcion:
                descripcion = planta.descripcion
                print(f"🌿 ¡Ahorro de API! Recuperado de CACHÉ: {nombre_cientifico}")
                break
                
        # B. Si no estaba en caché, le preguntamos a Gemini con REINTENTOS AUTOMÁTICOS
        if not descripcion:
            print(f"🔍 Nueva especie detectada. Consultando a Gemini: {nombre_cientifico}")
            
            max_reintentos = 3
            for intento in range(max_reintentos):
                try:
                    prompt_botanico = f"Actúa como un botánico experto. Escribe un resumen breve, de máximo 3 líneas y fácil de entender, sobre la planta conocida como {nombre_comun} (nombre científico: {nombre_cientifico}). Menciona algún dato curioso o su cuidado básico."
                    
                    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                    respuesta_ia = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt_botanico
                    )
                    descripcion = respuesta_ia.text.strip()
                    break  # Si tiene éxito, rompemos el ciclo y continuamos
                    
                except Exception as e:
                    print(f"⚠️ Error de red/demanda con IA (Intento {intento + 1}/{max_reintentos}): {e}")
                    if intento < max_reintentos - 1:
                        time.sleep(2)  # Esperar 2 segundos antes de volver a tocar la puerta de Google
                    else:
                        descripcion = "Descripción biológica no disponible en este momento."
        # ========================================================

        # 3. Subir la foto a Supabase Storage
        nombre_archivo = f"flora_{uuid.uuid4()}.jpg"
        
        supabase.storage.from_("plantas").upload(
            path=nombre_archivo,
            file=contenido_imagen,
            file_options={"content-type": "image/jpeg"}
        )
        url_publica = supabase.storage.from_("plantas").get_public_url(nombre_archivo)
        
        # 4. Guardar en Prisma
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

# ==========================================
# ENDPOINT 2: FAUNA (Gemini Multimodal)
# ==========================================
@app.post("/identificar-fauna/")
async def identificar_fauna(
    imagen: UploadFile = File(...),
    latitud: float = Form(None),
    longitud: float = Form(None)
):
    contenido_imagen = await imagen.read()
    
    try:
        # 1. Analizar imagen directamente con Gemini
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        prompt_fauna = """
        Eres un biólogo experto. Analiza esta imagen e identifica al animal (insecto, ave, mamífero, etc.).
        Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta, sin texto extra ni formato Markdown:
        {
            "nombre_comun": "Nombre común del animal",
            "nombre_cientifico": "Nombre científico",
            "probabilidad": 95.5,
            "descripcion": "Un resumen breve de 3 líneas con un dato curioso o su hábitat."
        }
        Si no hay un animal claro en la imagen, devuelve un JSON con "nombre_comun": "Ninguno".
        """
        
        respuesta_ia = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=[
                prompt_fauna, 
                {"mime_type": "image/jpeg", "data": contenido_imagen}
            ]
        )
        
        texto_json = respuesta_ia.text.replace('```json', '').replace('```', '').strip()
        datos_animal = json.loads(texto_json)
        
        if datos_animal.get("nombre_comun") == "Ninguno":
            return {"exito": False, "mensaje": "No se detectó ningún animal en la imagen."}

        # 2. Subir a Supabase Storage
        nombre_archivo = f"fauna_{uuid.uuid4()}.jpg"
        supabase.storage.from_("plantas").upload(
            path=nombre_archivo, file=contenido_imagen, file_options={"content-type": "image/jpeg"}
        )
        url_publica = supabase.storage.from_("plantas").get_public_url(nombre_archivo)
        
        # 3. Guardar en Prisma
        await db.consultaplanta.create(
            data={
                'nombre_cientifico': datos_animal["nombre_cientifico"],
                'nombre_comun': datos_animal["nombre_comun"],
                'probabilidad': datos_animal["probabilidad"],
                'foto_url': url_publica,
                'latitud': latitud,
                'longitud': longitud,
                'descripcion': datos_animal["descripcion"]
            }
        )
        
        return {
            "exito": True,
            "mensaje": "Animal identificado",
            "datos": {
                "nombre_cientifico": datos_animal["nombre_cientifico"],
                "nombre_comun": datos_animal["nombre_comun"],
                "probabilidad": f"{datos_animal['probabilidad']}%",
                "descripcion": datos_animal["descripcion"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis de fauna: {str(e)}")

# ==========================================
# ENDPOINT 3: HISTORIAL
# ==========================================
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