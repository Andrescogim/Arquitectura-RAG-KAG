from collections import Counter
import numpy as np


def extraer_top_k_entities(query_result, k):
    entities_found = []
    for entity in range(k):
        entities_found.append(query_result[entity]['name'])
    
    return entities_found


def formatear_tripletas(relaciones):
    tripletas_str = []

    for entidad in relaciones:
        for tripleta in relaciones[entidad]:
            origen = tripleta['origen']
            destino = tripleta['destino']
            relacion = tripleta['relacion']

            trip_str = f"{origen} -> {relacion} -> {destino}"
            
            tripletas_str.append(trip_str)
            
            tripletas_formateadas = "\n".join(tripletas_str)
            
    return tripletas_formateadas


def filtrado_parcial(results_parciales, min_score):
                
    filtrado_parcial=[]
    for k,v in results_parciales.items():
        for ent in v:
            if ent['score'] > min_score:
                filtrado_parcial.append(ent['name'])
    return filtrado_parcial


def filtrado_fuzzy(results_fuzzy, min_score):
    
    filtrado_fuzzy=[]
    for k,v in results_fuzzy.items():
        for ent in v:
            if ent['score'] > min_score:
                filtrado_fuzzy.append(ent['name'])

    return filtrado_fuzzy


def combinar_entis_parcial_fuzzy(entis_parcial, entis_fuzzy, n_final):
    conteo_fuzzy_parcial = Counter(entis_parcial + entis_fuzzy)
    entidades_mas_comunes = [k for k,v in conteo_fuzzy_parcial.most_common(n_final)]
    return entidades_mas_comunes


def union_entidades(entis_exactas, entidades_text_index, entidades_embeddings):
    entidades_finales = list(set(entis_exactas + entidades_text_index + entidades_embeddings))
    return entidades_finales



def formatear_tripletas_extendidas_old(relaciones):
    tripletas_formateadas = [(" -> ".join(rel)) for rel in relaciones]
            
    return tripletas_formateadas

def formatear_tripletas_extendidas(relaciones):
    rels_format = {}
    for enti, triplets in relaciones.items():
        rels_string = [(" -> ".join(rel)) for rel in triplets]
        rels_format[enti] = rels_string
            
    return rels_format


def reranking_tripletas_old(question, tripletas, reranker):
    # reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    pairs = [[question, relacion] for relacion in tripletas ]
    scores_rerank = reranker.predict(pairs)
    tripletas_reranked={}
    for idx, elem in enumerate(tripletas):
        tripletas_reranked[elem] = scores_rerank[idx].item()
    return tripletas_reranked


def reranking_tripletas(question, tripletas, reranker):
    # reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    all_tripletas = []
    for entidad in tripletas:
        # all_tripletas.append(rel for rel in tripletas[entidad])
        all_tripletas += [rel for rel in tripletas[entidad]]
    
    pairs = [[question, relacion] for relacion in all_tripletas ]
    scores_rerank = reranker.predict(pairs)
    tripletas_reranked={}
    for idx, elem in enumerate(all_tripletas):
        tripletas_reranked[elem] = scores_rerank[idx].item()
    return tripletas_reranked



def reranking_tripletas_aux(question, relaciones, reranker):
    """
    Se guardan las 2 tripletas con mayor score de cada entidad
    """
    tripletas_finales = []
    for enti in relaciones:
        pairs = [[question, relacion] for relacion in relaciones[enti] ]
        scores = reranker.predict(pairs)
        if len(scores) <= 2:
            # tripletas_finales.append(relaciones[enti])
            tripletas_finales += relaciones[enti]
        else:
            indices_maximos = np.argpartition(relaciones[enti], -2)[-2:]
            # tripletas_finales.append([relaciones[enti][idx] for idx in indices_maximos])
            tripletas_finales += [relaciones[enti][idx] for idx in indices_maximos]
    return tripletas_finales



def filtrar_tripletas_reranked(tripletas_reranked):
    tripletas_filt = [k for k,v in tripletas_reranked.items() if v > 0.75]
    return tripletas_filt