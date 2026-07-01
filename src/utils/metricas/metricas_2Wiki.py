
import re
import string
from collections import Counter
import json
from groq import Groq
from pydantic import BaseModel, Field


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1, precision, recall



def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))



def respuesta_en_nodos_encontrados(nodos, ground_truth):
    # normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    nodos_lista = [normalize_answer(node['name']) for entidad in nodos for node in nodos[entidad]]
    return normalized_ground_truth in nodos_lista


def suporting_facts_en_subgrafo(nodos, sup_facts, new = None):
    """
    Comprobar cuantos de los supporting facts estan en la entidades
    del subgrafo recuperado.
    Devuelve Nº de sup_facts y nº de entidades igual a sup_facts
    """
    if new == 1:
        nodos_lista = [normalize_answer(node) for node in nodos]
    else:
        nodos_lista = [normalize_answer(node['name']) for entidad in nodos for node in nodos[entidad]]
    # sup_facts_lista = [normalize_answer(sf[0]) for sf in eval(sup_facts)]
    sup_facts_lista = [normalize_answer(sf[0]) for sf in sup_facts]
    comunes = Counter(nodos_lista) & Counter(sup_facts_lista)
    return len(sup_facts_lista), len(comunes)



def metricas_totales(resultados):
    
    N_resultados = len(resultados)
    metricas_totales = {}
    
    em = 0
    precision = 0
    recall = 0
    f1 = 0
    
    for k,v in resultados.items():
        em += resultados[k]['em']
        precision += resultados[k]['precision']
        recall += resultados[k]['recall']
        f1 += resultados[k]['f1']
        
    metricas_totales["em"] = em / N_resultados
    metricas_totales["precision"] = precision / N_resultados
    metricas_totales["recall"] = recall / N_resultados
    metricas_totales["f1"] = f1 / N_resultados
    
    return metricas_totales


class EvaluacionRAG(BaseModel):
    strict_hallucination: int = Field(
        description="1 if the answer is 100% supported by the triplets OR is an 'I don't know' evasion (never guess). 0 if it claims ANY unsupported fact."
    )
    incorrect_evasion: int = Field(
        description="1 if the model evades with 'I don't know' despite the triplets having the clear answer points. 0 in any other case."
    )
    context_recall: int = Field(
        description="1 if the retrieved triplets contain the necessary information to successfully answer the question. 0 if information is missing."
    )
    context_relevance: int = Field(
        description="Number of the retrieved triplets that are highly specific to the question."
    )
    explanation: str = Field(
        description="A concise single-sentence summary justifying the scores assigned."
    )
    
    
def metricas_semanticas_groq(client_groq, modelo, pregunta, respuesta, tripletas):
    prompt_sistema = f"""
    You are an expert logic and quality control auditor for Knowledge Graph RAG systems.
    Your task is to evaluate the provided case and output your verdict strictly adhering to this JSON schema:
    {json.dumps(EvaluacionRAG.model_json_schema())}
    
    CRITICAL INSTRUCTIONS:
    - Return ONLY the raw JSON object. Do not include markdown blocks (```json) or introductory text.
    - Evaluate strict_hallucination and incorrect_evasion as mutually exclusive errors regarding 'I don't know' responses.
    """
    prompt_usuario = f"""
    DATA TO AUDIT:
    - Retrieved Graph Triplets: {tripletas}
    - User Question: {pregunta}
    - System Answer: {respuesta}
    """
    
    completion = client_groq.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        # model="openai/gpt-oss-120b",
        model = modelo,
        temperature=0.0, 
        response_format={"type": "json_object"}
    )
    
    # Parseamos el string JSON directamente a un diccionario de Python
    # resultado_dict = json.loads(completion.choices.message.content)
    return json.loads(completion.choices[0].message.content)


def fusionar_metricas(output, metricas_groq):
    for i, key in enumerate(output):
        output[key]["alucinacion"] = metricas_groq[i]["strict_hallucination"]
        output[key]["Desconocimiento erroneo"] = metricas_groq[i]["incorrect_evasion"]
        output[key]["Context recall"] = metricas_groq[i]["context_recall"]
        output[key]["Context relevance"] = metricas_groq[i]["context_relevance"]
        output[key]["explicacion"] = metricas_groq[i]["explanation"]
    return output