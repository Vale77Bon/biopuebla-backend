import asyncio
import base64
import json
import os
import uuid

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from prisma import Prisma
from supabase import Client, create_client

# Carga .env en local. En Render configura las mismas variables en Environment.
load_dotenv()

app = FastAPI(title="BioPuebla API - Reconocimiento de Flora y Fauna")

# Para producción, reemplaza "*" por el origen real de tu frontend si es posible.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Prisma()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PLANTNET_URL = (
    f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}&lang=es"
    if PLANTNET_API_KEY
    else None
)

supabase: Client | None = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY
    else None
)

GEMINI_MODEL = "gemini-3.8-flash"
# Qwen 3.8 27B acepta texto e imágenes en Groq.
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


@app.on_event("startup")
async def startup():
    await db.connect()
    print("BIOPUEBLA_API_VERSION=2026-10-fauna-groq-qwen38")


@app.on_event("shutdown")
async def shutdown():
    await db.disconnect()


def get_supabase() -> Client:
    if supabase is None:
        raise RuntimeError("Faltan SUPABASE_URL o SUPABASE_KEY en las variables de entorno.")
    return supabase


def parse_json_from_text(texto: str) -> dict:
    """Extrae un objeto JSON de la respuesta de un modelo de IA."""
    if not texto:
        raise ValueError("La IA devolvió una respuesta vacía.")

    texto = texto.strip()
    if texto.startswith("```"):
        texto = texto.strip("`")
        if texto.lower().startswith("json"):
            texto = texto[4:].strip()

    inicio = texto.find("{")
    fin = texto.rfind("}") + 1
    if inicio < 0 or fin <= inicio:
        raise ValueError("La IA no devolvió un objeto JSON.")

    datos = json.loads(texto[inicio:fin])
    if not isinstance(datos, dict):
        raise ValueError("La respuesta JSON no es un objeto.")
    return datos


def groq_chat(payload: dict) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("Falta configurar GROQ_API_KEY.")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    respuesta = requests.post(
        GROQ_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    if respuesta.status_code != 200:
        # No registrar la clave ni el contenido de variables secretas.
        raise RuntimeError(
            f"Groq respondió HTTP {respuesta.status_code}: {respuesta.text[:500]}"
        )
    return respuesta.json()


async def generar_descripcion_planta(prompt: str) -> str:
    """Gemini primero; Groq como respaldo."""
    if GEMINI_API_KEY:
        for intento in range(2):
            try:
                cliente = genai.Client(api_key=GEMINI_API_KEY)
                respuesta = await asyncio.to_thread(
                    cliente.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                if respuesta.text and respuesta.text.strip():
                    return respuesta.text.strip()
                raise ValueError("Gemini devolvió una respuesta vacía.")
            except Exception as exc:
                print(f"Gemini flora intento {intento + 1}/2 falló: {exc}")
                if intento == 0:
                    await asyncio.sleep(2)

    try:
        resultado = await asyncio.to_thread(
            groq_chat,
            {
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un biólogo experto. Responde en español, "
                            "de forma breve, clara y precisa."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 180,
            },
        )
        descripcion = resultado["choices"][0]["message"]["content"].strip()
        if descripcion:
            return descripcion
        raise ValueError("Groq devolvió una descripción vacía.")
    except Exception as exc:
        print(f"Groq flora también falló: {exc}")
        return (
            "La planta se identificó correctamente, pero no fue posible "
            "generar su descripción en este momento."
        )


async def identificar_animal_con_gemini(contenido_imagen: bytes, mime_type: str, prompt: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta configurar GEMINI_API_KEY.")

    cliente = genai.Client(api_key=GEMINI_API_KEY)
    respuesta = await asyncio.to_thread(
        cliente.models.generate_content,
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=contenido_imagen, mime_type=mime_type),
        ],
    )
    return parse_json_from_text(respuesta.text or "")


async def identificar_animal_con_groq(contenido_imagen: bytes, mime_type: str, prompt: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("Falta configurar GROQ_API_KEY.")

    imagen_b64 = base64.b64encode(contenido_imagen).decode("utf-8")
    resultado = await asyncio.to_thread(
        groq_chat,
        {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un biólogo experto. Identifica animales con prudencia. "
                        "No inventes datos y responde solo con JSON válido."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{imagen_b64}"
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.2,
            "max_completion_tokens": 500,
            "response_format": {"type": "json_object"},
        },
    )
    texto = resultado["choices"][0]["message"]["content"]
    return parse_json_from_text(texto)


async def subir_foto(contenido: bytes, prefijo: str) -> str:
    cliente = get_supabase()
    nombre_archivo = f"{prefijo}_{uuid.uuid4()}.jpg"
    await asyncio.to_thread(
        cliente.storage.from_("plantas").upload,
        path=nombre_archivo,
        file=contenido,
        file_options={"content-type": "image/jpeg"},
    )
    return cliente.storage.from_("plantas").get_public_url(nombre_archivo)


@app.get("/")
async def inicio():
    return {"exito": True, "mensaje": "BioPuebla API activa"}


@app.get("/health")
async def health():
    return {
        "exito": True,
        "servicio": "BioPuebla API",
        "version": "2026-10-fauna-groq-qwen38",
    }


# ============================================================
# ENDPOINT 1: FLORA (PlantNet + caché + Gemini/Groq)
# ============================================================
@app.post("/identificar/")
async def identificar_especie(
    imagen: UploadFile = File(...),
    latitud: float = Form(None),
    longitud: float = Form(None),
):
    contenido_imagen = await imagen.read()
    if not contenido_imagen:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")
    if not PLANTNET_URL:
        raise HTTPException(status_code=500, detail="Falta configurar PLANTNET_API_KEY.")

    try:
        respuesta = await asyncio.to_thread(
            requests.post,
            PLANTNET_URL,
            files={
                "images": (
                    imagen.filename or "planta.jpg",
                    contenido_imagen,
                    imagen.content_type or "image/jpeg",
                )
            },
            data={"organs": ["auto"]},
            timeout=60,
        )

        try:
            datos = respuesta.json()
        except ValueError:
            datos = {}

        if respuesta.status_code == 404 or "not found" in str(datos).lower():
            return {
                "exito": False,
                "mensaje": "No se detectó ninguna planta. Enfoca bien las hojas, flores o el tallo.",
            }

        if respuesta.status_code != 200:
            print(f"PlantNet respondió HTTP {respuesta.status_code}: {str(datos)[:400]}")
            raise HTTPException(
                status_code=502,
                detail="El servicio de identificación de plantas no respondió correctamente.",
            )

        resultados = datos.get("results") or []
        if not resultados:
            return {
                "exito": False,
                "mensaje": "PlantNet no encontró una planta identificable en esta imagen.",
            }

        mejor_resultado = resultados[0]
        especie = mejor_resultado.get("species") or {}
        nombre_cientifico = especie.get("scientificNameWithoutAuthor") or "Desconocido"
        nombres_comunes = especie.get("commonNames") or []
        nombre_comun = nombres_comunes[0] if nombres_comunes else nombre_cientifico
        probabilidad = float(mejor_resultado.get("score", 0)) * 100

        descripcion = None
        plantas_previas = await db.consultaplanta.find_many(
            where={"nombre_cientifico": nombre_cientifico}
        )
        for planta in plantas_previas:
            descripcion_previa = getattr(planta, "descripcion", None)
            if (
                descripcion_previa
                and "no disponible" not in descripcion_previa.lower()
                and "mantenimiento" not in descripcion_previa.lower()
            ):
                descripcion = descripcion_previa
                print(f"Flora: descripción recuperada de caché para {nombre_cientifico}")
                break

        if not descripcion:
            prompt_botanico = (
                "Actúa como botánico experto. Escribe en español un resumen breve "
                "(máximo 3 líneas) y fácil de entender sobre la planta "
                f"{nombre_comun} (nombre científico: {nombre_cientifico}). "
                "Incluye un dato curioso o un consejo de cuidado básico."
            )
            descripcion = await generar_descripcion_planta(prompt_botanico)

        url_publica = await subir_foto(contenido_imagen, "flora")

        await db.consultaplanta.create(
            data={
                "nombre_cientifico": nombre_cientifico,
                "nombre_comun": nombre_comun,
                "probabilidad": probabilidad,
                "foto_url": url_publica,
                "latitud": latitud,
                "longitud": longitud,
                "descripcion": descripcion,
            }
        )

        return {
            "exito": True,
            "mensaje": "Especie identificada y guardada",
            "datos": {
                "nombre_cientifico": nombre_cientifico,
                "nombre_comun": nombre_comun.capitalize(),
                "probabilidad": f"{probabilidad:.2f}%",
                "descripcion": descripcion,
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error en identificación de flora: {exc}")
        raise HTTPException(status_code=500, detail="Error interno al identificar la planta.")


# ============================================================
# ENDPOINT 2: FAUNA (Gemini + Groq Qwen 3.8 27B Vision)
# ============================================================
@app.post("/identificar-fauna/")
async def identificar_fauna(
    imagen: UploadFile = File(...),
    latitud: float = Form(None),
    longitud: float = Form(None),
):
    contenido_imagen = await imagen.read()
    if not contenido_imagen:
        raise HTTPException(status_code=400, detail="El archivo de imagen está vacío.")

    mime_type = imagen.content_type or "image/jpeg"
    prompt_fauna = """
Eres un biólogo experto. Analiza la imagen e identifica el animal si se ve
claramente (insecto, ave, mamífero, reptil, anfibio, pez u otro animal).
No inventes una identificación si la imagen no permite distinguirlo.
Devuelve únicamente un objeto JSON válido con esta estructura:
{
  "nombre_comun": "Nombre común del animal",
  "nombre_cientifico": "Nombre científico o Desconocido",
  "probabilidad": 90.0,
  "descripcion": "Descripción breve en español, de máximo 3 líneas."
}
Si no se distingue claramente ningún animal, devuelve:
{"nombre_comun":"Ninguno","nombre_cientifico":"Desconocido","probabilidad":0,"descripcion":"No se distingue un animal claramente en la imagen."}
"""

    datos_animal = None

    # Gemini: dos intentos con espera asíncrona (no bloquea el servidor).
    if GEMINI_API_KEY:
        for intento in range(2):
            try:
                datos_animal = await identificar_animal_con_gemini(
                    contenido_imagen, mime_type, prompt_fauna
                )
                if datos_animal.get("nombre_comun"):
                    print(f"Fauna identificada por Gemini; intento {intento + 1}")
                    break
                raise ValueError("Gemini no incluyó nombre_comun en el JSON.")
            except Exception as exc:
                print(f"Gemini fauna intento {intento + 1}/2 falló: {exc}")
                datos_animal = None
                if intento == 0:
                    await asyncio.sleep(2)

    # Respaldo: Qwen 3.8 27B es multimodal y acepta imágenes en Groq.
    if not datos_animal:
        print("BIOPUEBLA_FAUNA_V2: activando Groq Vision con qwen/qwen3.8-27b")
        try:
            datos_animal = await identificar_animal_con_groq(
                contenido_imagen, mime_type, prompt_fauna
            )
        except Exception as exc:
            print(f"Groq fauna también falló: {exc}")
            return {
                "exito": False,
                "mensaje": (
                    "No se pudo analizar la imagen porque los servicios de IA "
                    "no están disponibles en este momento. Intenta de nuevo más tarde."
                ),
            }

    nombre_comun = str(datos_animal.get("nombre_comun") or "Desconocido").strip()
    if nombre_comun.lower() == "ninguno":
        return {
            "exito": False,
            "mensaje": "No se detectó ningún animal claramente en la imagen.",
        }

    nombre_cientifico = str(
        datos_animal.get("nombre_cientifico") or "Desconocido"
    ).strip()
    descripcion = str(datos_animal.get("descripcion") or "Sin descripción.").strip()

    try:
        probabilidad = float(
            str(datos_animal.get("probabilidad", 90)).replace("%", "").strip()
        )
    except (TypeError, ValueError):
        probabilidad = 90.0
    probabilidad = max(0.0, min(probabilidad, 100.0))

    try:
        url_publica = await subir_foto(contenido_imagen, "fauna")

        await db.consultaplanta.create(
            data={
                "nombre_cientifico": nombre_cientifico,
                "nombre_comun": nombre_comun,
                "probabilidad": probabilidad,
                "foto_url": url_publica,
                "latitud": latitud,
                "longitud": longitud,
                "descripcion": descripcion,
            }
        )

        return {
            "exito": True,
            "mensaje": "Animal identificado y guardado",
            "datos": {
                "nombre_cientifico": nombre_cientifico,
                "nombre_comun": nombre_comun.capitalize(),
                "probabilidad": f"{probabilidad:.2f}%",
                "descripcion": descripcion,
            },
        }
    except Exception as exc:
        print(f"Error guardando consulta de fauna: {exc}")
        return {
            "exito": False,
            "mensaje": (
                "El animal se identificó, pero ocurrió un problema al guardar "
                "la consulta o subir la foto."
            ),
            "datos": {
                "nombre_cientifico": nombre_cientifico,
                "nombre_comun": nombre_comun.capitalize(),
                "probabilidad": f"{probabilidad:.2f}%",
                "descripcion": descripcion,
            },
        }


# ============================================================
# ENDPOINT 3: HISTORIAL
# ============================================================
@app.get("/historial/")
async def obtener_historial():
    try:
        historial = await db.consultaplanta.find_many(
            order={"fecha_escaneo": "desc"}
        )
        return [
            {
                "id": item.id,
                "nombre_cientifico": item.nombre_cientifico,
                "nombre_comun": item.nombre_comun,
                "probabilidad": item.probabilidad,
                "fecha_escaneo": (
                    item.fecha_escaneo.isoformat() if item.fecha_escaneo else None
                ),
                "foto_url": item.foto_url,
                "latitud": item.latitud,
                "longitud": item.longitud,
                "descripcion": getattr(item, "descripcion", None),
            }
            for item in historial
        ]
    except Exception as exc:
        print(f"Error al cargar el historial: {exc}")
        raise HTTPException(status_code=500, detail="No se pudo cargar el historial.")
