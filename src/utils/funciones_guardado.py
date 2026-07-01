from datetime import datetime as dt
import json


def guardar_resultados(resultados, nombre_archivo, ruta):
    
    ahora = dt.now()
    fecha_hora = ahora.strftime("%Y-%m-%d_%H-%M")
    out_name = f"{nombre_archivo}_{fecha_hora}.json"
    out_file = ruta / out_name
    resultados_json = json.dumps(resultados, indent=4, ensure_ascii=False)
    with open(out_file, "w", encoding="utf-8") as archivo:
        archivo.write(resultados_json)
    return resultados_json



def guardar_registro(ruta, comentarios, reg_in, reg_fin, metricas_totales, modelo_LLM, prompt, opciones_LLM, parametros_retrieval):
    """
        Guarda registro de ejecucion.
        En un txt se guarda:
        - Fecha y hora de ejecucion
        - LLM Utilizado
        - Prompt utilizado
    """

    # path = "C:/Users/andre/OneDrive/Escritorio/TFM/KAG/AI-KG/Project/KAG-Graph-over-AI-KG/outputs/registro/"
    
    ahora = dt.now()
    fecha_hora = ahora.strftime("%Y-%m-%d_%H-%M")
    out_name = f"Registro_{fecha_hora}.txt"
    # out_file = f"{path}{out_name}"
    out_file = ruta / out_name

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"fecha_ejecucion: {fecha_hora}\n\n")
        f.write(f"Comentarios: {comentarios}\n\n")
        f.write(f"Nº inicio registros: {reg_in}\n\n")
        f.write(f"Nº ffin registros: {reg_fin}\n\n")
        f.write(f"Metricas: {json.dumps(metricas_totales, indent = 2, ensure_ascii = False)}\n\n")
        f.write(f"modelo_utilizado: {modelo_LLM}\n\n")
        f.write(f"parametros LLM: {opciones_LLM}\n\n")
        f.write(f"parametros generales: {parametros_retrieval}\n\n")
        f.write(f'prompt_utilizado: \n"{prompt}"')

    return 1


def guardar_df_as_csv(df, nombre_archivo, ruta):
    
    ahora = dt.now()
    fecha_hora = ahora.strftime("%Y-%m-%d_%H-%M")
    out_name = f"{nombre_archivo}_{fecha_hora}.csv"
    out_file = ruta / out_name
    df.to_csv(out_file, index=False)

