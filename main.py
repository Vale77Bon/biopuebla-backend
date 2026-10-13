from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import requests
from fastapi.middleware.cors import CORSMiddleware
from prisma import Prisma
from google import genai
from google.genai import types
import json
import uuid
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import time
import base64

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
# ENDPOINT 1: FLORA (Pl@ntNet + Caché + Gemini/Groq)
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
        
        # 2. SISTEMA DE CACHÉ Y EXTRACCIÓN CON IA MULTIPLE
        descripcion = None
        
        plantas_previas = await db.consultaplanta.find_many(
            where={'nombre_cientifico': nombre_cientifico}
        )
        
        for planta in plantas_previas:
            if planta.descripcion and "no disponible" not in planta.descripcion and "mantenimiento" not in planta.descripcion:
                descripcion = planta.descripcion
                print(f"🌿 ¡Ahorro de API! Recuperado de CACHÉ: {nombre_cientifico}")
                break
                
        if not descripcion:
            print(f"🔍 Nueva especie detectada. Consultando IA: {nombre_cientifico}")
            prompt_botanico = f"Actúa como un botánico experto. Escribe un resumen breve, de máximo 3 líneas y fácil de entender, sobre la planta conocida como {nombre_comun} (nombre científico: {nombre_cientifico}). Menciona algún dato curioso o su cuidado básico."
            
            max_reintentos = 2
            exito_gemini = False
            
            # Intento 1: Google Gemini
            for intento in range(max_reintentos):
                try:
                    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                    respuesta_ia = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt_botanico
                    )
                    descripcion = respuesta_ia.text.strip()
                    exito_gemini = True
                    break 
                except Exception as e:
                    print(f"⚠️ Error Gemini (Intento {intento + 1}/{max_reintentos}): {e}")
                    if intento < max_reintentos - 1:
                        time.sleep(1)
            
            # Intento 2: Groq (Llama 3)
            if not exito_gemini:
                print("🛟 Gemini saturado. Activando IA de repuesto (Groq / Llama 3)...")
                try:
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    headers = {
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": "Eres un biólogo experto. Responde siempre de forma breve y precisa."},
                            {"role": "user", "content": prompt_botanico}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 150
                    }
                    
                    res_groq = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                    if res_groq.status_code == 200:
                        descripcion = res_groq.json()['choices'][0]['message']['content'].strip()
                        print("✅ Resumen obtenido de Groq (Llama 3) exitosamente.")
                    else:
                        raise Exception("Fallo al conectar con Groq")
                except Exception as e:
                    print(f"❌ Falló también la IA de repuesto: {e}")
                    descripcion = "Planta identificada correctamente, pero los servidores de Inteligencia Artificial están en mantenimiento."

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
# ENDPOINT 2: FAUNA (Gemini + Groq Vision 90B)
# ==========================================
@app.post("/identificar-fauna/")
async def identificar_fauna(
    imagen: UploadFile = File(...),
    latitud: float = Form(None),
    longitud: float = Form(None)
):
    contenido_imagen = await imagen.read()
    
    try:
        from google.genai import types
        import base64
        import time
        import re

        prompt_fauna = """
        Eres un biólogo experto. Analiza esta imagen e identifica al animal (insecto, ave, mamífero, etc.).
        Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta, sin texto extra:
        {
            "nombre_comun": "Nombre común",
            "nombre_cientifico": "Nombre científico",
            "probabilidad": 95.5,
            "descripcion": "Un resumen breve de 3 líneas."
        }
        Si no hay un animal claro, devuelve {"nombre_comun": "Ninguno"}.
        """
        
        datos_animal = None
        exito_gemini = False
        max_reintentos = 2
        
        # INTENTO 1: GEMINI MULTIMODAL
        for intento in range(max_reintentos):
            try:
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                respuesta_ia = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=[
                        prompt_fauna, 
                        types.Part.from_bytes(data=contenido_imagen, mime_type='image/jpeg')
                    ]
                )
                
                texto = respuesta_ia.text
                inicio = texto.find('{')
                fin = texto.rfind('}') + 1
                if inicio != -1 and fin != 0:
                    datos_animal = json.loads(texto[inicio:fin])
                    exito_gemini = True
                    break
                else:
                    raise Exception("Formato incorrecto devuelto por Gemini")
                    
            except Exception as e:
                print(f"⚠️ Error Gemini Fauna (Intento {intento + 1}): {e}")
                if intento < max_reintentos - 1:
                    time.sleep(1)

        # INTENTO 2: GROQ VISION (LLAMA 3.2 - 90B) DE RESPALDO
        if not exito_gemini:
            print("🛟 Activando repuesto Groq Vision (Llama 3.2 - 90B) para Fauna...")
            groq_key = os.getenv("GROQ_API_KEY")
            imagen_b64 = base64.b64encode(contenido_imagen).decode('utf-8')
            
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.2-90b-vision-preview",  # <--- EL NUEVO MODELO DE 90 BILLONES
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_fauna},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imagen_b64}"}}
                        ]
                    }
                ],
                "temperature": 0.5
            }
            
            res_groq = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            if res_groq.status_code == 200:
                texto = res_groq.json()['choices'][0]['message']['content']
                inicio = texto.find('{')
                fin = texto.rfind('}') + 1
                if inicio != -1 and fin != 0:
                    datos_animal = json.loads(texto[inicio:fin])
                else:
                    raise Exception("Groq no devolvió un JSON válido")
            else:
                print(f"Fallo en Groq: {res_groq.text}")
                # ANTI-CRASH: Si Groq también falla, mandamos un error controlado, NO un error 500
                return {
                    "exito": False, 
                    "mensaje": "Los servidores de IA (Google y Meta) están saturados por alta demanda mundial. Intenta escanear de nuevo en un minuto."
                }

        # Validar si encontró algo
        if not datos_animal or datos_animal.get("nombre_comun") == "Ninguno":
            return {"exito": False, "mensaje": "No se detectó ningún animal claramente en la imagen."}

        # Limpiar probabilidad
        try:
            prob_float = float(str(datos_animal.get("probabilidad", 0)).replace('%', ''))
        except:
            prob_float = 90.0

        # Subir foto a Supabase
        nombre_archivo = f"fauna_{uuid.uuid4()}.jpg"
        supabase.storage.from_("plantas").upload(
            path=nombre_archivo, file=contenido_imagen, file_options={"content-type": "image/jpeg"}
        )
        url_publica = supabase.storage.from_("plantas").get_public_url(nombre_archivo)
        
        # Guardar en Prisma
        await db.consultaplanta.create(
            data={
                'nombre_cientifico': datos_animal.get("nombre_cientifico", "Desconocido"),
                'nombre_comun': datos_animal.get("nombre_comun", "Desconocido"),
                'probabilidad': prob_float,
                'foto_url': url_publica,
                'latitud': latitud,
                'longitud': longitud,
                'descripcion': datos_animal.get("descripcion", "Sin descripción.")
            }
        )
        
        return {
            "exito": True,
            "mensaje": "Animal identificado",
            "datos": {
                "nombre_cientifico": datos_animal.get("nombre_cientifico"),
                "nombre_comun": datos_animal.get("nombre_comun", "").capitalize(),
                "probabilidad": f"{prob_float}%",
                "descripcion": datos_animal.get("descripcion")
            }
        }

    except Exception as e:
        print(f"❌ Error CRÍTICO en Fauna: {e}")
        # RED DE SEGURIDAD FINAL: Nunca devolver error 500 a la aplicación
        return {
            "exito": False, 
            "mensaje": "Los servidores de Inteligencia Artificial están experimentando alta demanda. Intenta de nuevo en unos segundos."
        }   

        
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