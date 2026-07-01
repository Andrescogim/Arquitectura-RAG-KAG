import json
import os
import time
import tracemalloc
import psutil
from functools import wraps


def build_prompt(prompt_base, info_adicional):
    
    prompt = prompt_base.format(**info_adicional)
    return prompt


def lectura_json(ruta, archivo):
    # file = str(ruta/archivo)
    ruta_archivo = f"{ruta / archivo}"
    with open(ruta_archivo, "r", encoding='utf-8') as archivo:
        archivo_leido = json.load(archivo)
    return archivo_leido


def medir_recursos(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        proceso = psutil.Process(os.getpid())
        
        ram_inicial_mb = proceso.memory_info().rss / (10**6)
        start_time = time.perf_counter()
        # tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        # _, pico_memoria = tracemalloc.get_traced_memory()
        # pico_memoria_mb = pico_memoria / 10**6
        # tracemalloc.stop()
        info_extendida = proceso.memory_info()
        pico_memoria_mb = info_extendida.peak_wset / (10**6)
        
        mediciones = {
            "funcion": func.__name__,
            "tiempo_ejecucion": round(end_time - start_time, 4),
            "RAM incial": round(ram_inicial_mb, 4),
            "RAM maxima funcion": round((pico_memoria_mb - ram_inicial_mb), 4),
            "RAM maxima total": round(pico_memoria_mb, 4)
        }
        medir_recursos.acumulado.append(mediciones)
        
        return result
    return wrapper

medir_recursos.acumulado = []