
import re
import string
from collections import Counter

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
    sup_facts_lista = [normalize_answer(sf[0]) for sf in eval(sup_facts)]
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