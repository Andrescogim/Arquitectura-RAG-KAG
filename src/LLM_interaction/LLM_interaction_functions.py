import ollama
from pydantic import BaseModel

class Tripleta(BaseModel):
  subject: str
  relation: str
  object: str

class TripletasList(BaseModel):
  tripletas: list[Tripleta]
  
  
def modelos_disponibles():
    """Muestra los modelos disponibles"""
    
    for model in ollama.list().models:
        print(f"Modelo: {model.model} ; N_Parametros = {model.details.parameter_size}")



def chat(modelo, prompt, opciones):

    response = ollama.chat(
        model = modelo,
        messages = [{
            'role': 'user',
            'content': prompt,
            }],
        options = opciones,
    )
    return response.message.content


def generate(modelo, prompt, opciones, formato=None):

    # response = ollama.generate(
    #     model = modelo,
    #     prompt = prompt,
    #     options = opciones,
    #     format = TripletasList.model_json_schema(),
    # )
    parametros = {
        "model" : modelo,
        "prompt" : prompt,
        "options" : opciones,
        }
    if formato is not None:
        parametros[format] = TripletasList.model_json_schema()
    
    response = ollama.generate(**parametros)

    return response.response


def reset(modelo):

    ollama.chat(
        model = modelo,
        messages = [],
        keep_alive=0
    )
