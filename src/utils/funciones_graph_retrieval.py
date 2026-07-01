from collections import Counter
import numpy as np
import pandas as pd


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
        
        # rels_format[enti] = rels_string
        rels_string_norm = [rel.replace("_", " ") for rel in rels_string]
        rels_format[enti] = rels_string_norm
            
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



def filtrar_tripletas_reranked(tripletas_reranked, score_min):
    tripletas_filt = [k for k,v in tripletas_reranked.items() if v > score_min]
    return tripletas_filt



# NUEVAS FUNCIONES CON PANDAS
def subgrafo_a_pandas(subgrafo):
    na_row = {
        "entidad_incial": "",
        "nodo_1": "",
        "nodo_2": "",
        "nodo_3": "",
        "rel_1": "",
        "rel_2": "",
        "direccion_1": "",
        "direccion_2": "",
        "grado_1": "",
        "grado_2": "",
        "grado_3": ""
        }
    rels_pandas = []
    for path in subgrafo:
        row = na_row.copy()
        row["entidad_incial"] = path['entidad_inicial']
        row["nodo_1"] = path['nodos_names'][0]
        row["grado_1"] = path['degrees'][0]
        for i in range(1, len(path['nodos_names']) ):
            row[f"nodo_{i+1}"] = path['nodos_names'][i]
            row[f"grado_{i+1}"] = path['degrees'][i]
            row[f"rel_{i}"] = path['relaciones_types'][i-1]
            row[f"direccion_{i}"] = path['relaciones_direccion'][i-1]
        rels_pandas.append(row)
    df_subgrafo = pd.DataFrame(rels_pandas)
    return df_subgrafo


def filtrado_subgrafo(df_subgrafo, n_rel_max):
    """
    Filtra:
        - Caminos con 2 saltos, cuyo 2º nodo tiene mas de N relaciones
        - Caminos donde las 2 relaciones son iguales y 1ª OUT y 2ª IN
    """
    # df_subgrafo_filt = df_subgrafo.loc[(df_subgrafo['grado_2'] < n_rel_max) | (df_subgrafo['rel_2'].isna())]
    # df_subgrafo_filt = df_subgrafo.drop_duplicates(["nodo_1", "nodo_2", "nodo_3"])
    # df_subgrafo_filt = df_subgrafo.drop_duplicates()
    # df_subgrafo_filt = df_subgrafo_filt.loc[(df_subgrafo_filt['grado_2'] < n_rel_max) | (df_subgrafo_filt['rel_2'] == "")]
    df_subgrafo_filt = df_subgrafo.loc[(df_subgrafo['grado_2'] < n_rel_max) | (df_subgrafo['rel_2'] == "")]
    df_subgrafo_filt = df_subgrafo_filt.loc[~((df_subgrafo_filt['rel_1'] == df_subgrafo_filt['rel_2']) & (df_subgrafo_filt['direccion_1'] =='OUT') & (df_subgrafo_filt['direccion_2'] =='IN'))].reset_index(drop=True)
    return df_subgrafo_filt
    
    
def construir_string(fila):
    if fila['direccion_1'] == 'OUT':
        inicio = 'nodo_1'
        final = 'nodo_2'
    elif fila['direccion_1'] == 'IN':
        inicio = 'nodo_2'
        final = 'nodo_1'
    tripleta_formateada = f"{fila[inicio]} which {fila['rel_1'].replace('_', ' ')} is {fila[final]}."
    if fila['rel_2'] != "":
        if fila['direccion_2'] == 'OUT':
            inicio = 'nodo_2'
            final = 'nodo_3'
        elif fila['direccion_2'] == 'IN':
            inicio = 'nodo_3'
            final = 'nodo_2'
        tripleta_formateada += f" {fila[inicio]} which {fila['rel_2'].replace('_', ' ')} is {fila[final]}"
    # Construimos y retornamos el string completo
    return tripleta_formateada


def reranking_tripletas_pandas(question, df_subgrafo, reranker):
    pairs = [[question, frase] for frase in list(df_subgrafo['tripleta_formateada']) ]
    scores_rerank = reranker.predict(pairs)
    df_subgrafo["rerank_score"] = scores_rerank.tolist()
    return df_subgrafo


def reranking_relaciones_pandas(question, df_subgrafo, reranker):
    rels_1 = list(df_subgrafo['rel_1'])
    rels_2 = list(df_subgrafo['rel_2'])
    rels_to_rerank = []
    for i, rel in enumerate(rels_1):
        rel_to_rerank = rel.replace("_", " ")
        if rels_2[i] != "":
            rel_to_rerank += f" , {rels_2[i].replace('_', ' ')}"
        rels_to_rerank.append(rel_to_rerank)
    pairs = [[question, relacion] for relacion in rels_to_rerank ]
    scores_rerank = reranker.predict(pairs)
    df_subgrafo["rerank_score_rels"] = scores_rerank.tolist()
    return df_subgrafo, rels_to_rerank


def escalado_rerank_rels(df_subgrafo):
    """# Con rank poner valores entre 0 y 1 pero distribuidos normalmente para ue no salgan valores muy pequeños
    min_rank = df_subgrafo["rerank_score_rels"].rank().min()
    max_rank = df_subgrafo["rerank_score_rels"].rank().max()
    df_subgrafo["rerank_score_rels_escalado"] = (df_subgrafo["rerank_score_rels"].rank() - min_rank) / (max_rank - min_rank)"""
    # ESCALADO HACIENDO PRIMERO LOG - CONSERVA MEJOR LAS DISANCIAS ENTRE RERANKS
    df_subgrafo['rerank_score_rels_log'] = np.log(df_subgrafo['rerank_score_rels'])
    min_log = min(df_subgrafo['rerank_score_rels_log'])
    max_log = max(df_subgrafo['rerank_score_rels_log'])
    if min_log == max_log:
        df_subgrafo['rerank_score_rels_escalado'] = 1
    else:
        df_subgrafo['rerank_score_rels_escalado'] = (df_subgrafo['rerank_score_rels_log'] - min_log) / (max_log - min_log)
    return df_subgrafo


def ponderacion_score_reranking_rels(df_subgrafo, peso_tripleta, peso_rel):
    # df_subgrafo["score_rerank_ponderado"] = df_subgrafo["rerank_score"] * peso_tripleta + df_subgrafo["rerank_score_rels"] * peso_rel
    df_subgrafo["score_rerank_ponderado"] = df_subgrafo["rerank_score"] * peso_tripleta + df_subgrafo["rerank_score_rels_escalado"] * peso_rel
    return df_subgrafo


def filtrar_por_reranker_pandas(df_subgrafo, n_maximos, min_score, col_score):
    """
    Filtrar los n_maximos de cada entidad
    """
    df_subgrafo = df_subgrafo[df_subgrafo[col_score] >= min_score]
    df_subgrafo = df_subgrafo.drop_duplicates(["nodo_1", "nodo_2", "nodo_3"])
    # df_maximos = df_subgrafo.groupby('entidad_incial').apply(lambda x: x.nlargest(n_maximos, 'rerank_score'), include_groups=False)
    df_maximos = df_subgrafo.groupby("entidad_incial").apply(lambda x: x.nlargest(n_maximos, col_score), include_groups=False)
    df_maximos = df_maximos.sort_values(col_score, ascending = False).reset_index(drop=True)
    return df_maximos