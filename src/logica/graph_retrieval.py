from src.utils.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.utils.funciones_graph_retrieval import (
    extraer_top_k_entities,
    filtrado_parcial,
    filtrado_fuzzy,
    combinar_entis_parcial_fuzzy,
    union_entidades,
    subgrafo_a_pandas,
    filtrado_subgrafo,
    construir_string,
    reranking_tripletas_pandas,
    reranking_relaciones_pandas,
    escalado_rerank_rels,
    ponderacion_score_reranking_rels,
    filtrar_por_reranker_pandas
)
from src.utils.funciones_retrieval_generales import extraer_entidades_ner
from src.utils.funciones_generales import build_prompt
from src.utils.LLM_interaction import LLM_interaction_functions as llm_funcs
from src.utils.metricas.metricas_2Wiki import (
    f1_score,
    exact_match_score,
    respuesta_en_nodos_encontrados,
    suporting_facts_en_subgrafo,
    metricas_totales
    )



def extraer_buscar_entidades(con_Neo4j, question, ner_model, fulltext_index, vector_index, embed_model, min_score_parcial, min_score_fuzzy, n_final_fuzz_parc, n_res_embeddings):
    
    # ner_model = "en_core_web_sm"
    entidades_ner = extraer_entidades_ner(question, ner_model)
    entis_exactas = con_Neo4j.busqueda_exacta_entidades(entidades_ner)
    # fulltext_index_name = "entidadesIndex"
    res_busqueda_parcial = con_Neo4j.busqueda_parcial_entidades(fulltext_index, entidades_ner)
    res_busqueda_fuzzy = con_Neo4j.busqueda_fuzzy_entidades(fulltext_index, entidades_ner)
            
    # min_score_parcial = 2
    filt_busqueda_parcial = filtrado_parcial(res_busqueda_parcial, min_score_parcial)
    # min_score_fuzzy = 2
    filt_busqueda_fuzzy = filtrado_fuzzy(res_busqueda_fuzzy, min_score_fuzzy)
    # n_final = 3
    entidades_text_index = combinar_entis_parcial_fuzzy(filt_busqueda_parcial, filt_busqueda_fuzzy, n_final_fuzz_parc)
    
    # n_resultados_embedding = 3
    res_busqueda_embeddings = con_Neo4j.query_a_embedding(vector_index, embed_model, question, n_res_embeddings)
    entidades_embeddings = extraer_top_k_entities(res_busqueda_embeddings, 2)
    
    entidades_finales = union_entidades(entis_exactas, entidades_text_index, entidades_embeddings)
    
    return entidades_finales


def extraer_relaciones(con_Neo4j, question, entidades_finales, n_saltos, reranker, n_rel_max, peso_tripleta, peso_rel, n_maximos, min_score):
      
    # subgrafo_raw = extraer_subgrafo_completo_nueva(entidades_finales, n_saltos)
    subgrafo_raw = con_Neo4j.extraer_subgrafo_completo(entidades_finales, n_saltos)
    subgrafo_df = subgrafo_a_pandas(subgrafo_raw)
    # n_rel_max = 10
    subgrafo_df_filt = filtrado_subgrafo(subgrafo_df, n_rel_max)
    subgrafo_df_filt['tripleta_formateada'] = subgrafo_df_filt.apply(construir_string, axis=1)
    subgrafo_df_reranked = reranking_tripletas_pandas(question, subgrafo_df_filt, reranker)
    
    # NUEVO RERANKING DE SOLO RELACIONES
    subgrafo_df_reranked, rels_to_rerank = reranking_relaciones_pandas(question, subgrafo_df_reranked, reranker)
    subgrafo_df_reranked = escalado_rerank_rels(subgrafo_df_reranked)
    
    # peso_tripleta = 0.7
    # peso_rel = 0.3
    subgrafo_df_reranked = ponderacion_score_reranking_rels(subgrafo_df_reranked, peso_tripleta, peso_rel)
    col_score = "score_rerank_ponderado"
    # n_maximos = 3
    # min_score = 0.1
    subgrafo_df_reranked_filt = filtrar_por_reranker_pandas(subgrafo_df_reranked, n_maximos, min_score, col_score)

    return list(subgrafo_df_reranked_filt["tripleta_formateada"])


def generar_respuesta(question, tripletas, llm, opciones_llm, prompt_base):
    info_prompt = {}
    info_prompt['tripletas_formateadas'] = "\n".join(tripletas)
    info_prompt['question'] = question
    prompt = build_prompt(prompt_base, info_prompt)
    respuesta_llm = llm_funcs.generate(llm, prompt, opciones_llm)
    return respuesta_llm



def calcular_metricas_evaluacion(respuesta_llm, ground_truth, entidades_finales, sup_facts):
    em = exact_match_score(respuesta_llm, ground_truth)
    f1, precision, recall = f1_score(respuesta_llm, ground_truth)
    # respuesta_en_subgrafo = respuesta_en_nodos_encontrados(entidades_finales, ground_truth)
    n_sup_facts, n_ent_in_sup_fact = suporting_facts_en_subgrafo(entidades_finales, sup_facts, 1)
    # return em, f1, precision, recall, respuesta_en_subgrafo, n_sup_facts, n_ent_in_sup_fact
    return em, f1, precision, recall, n_sup_facts, n_ent_in_sup_fact


def contestar_2Wiki_con_grafo(
    con_Neo4j,
    n_registros,
    reranker,
    vector_index,
    ner_model,
    fulltext_index,
    n_saltos,
    score_min_reranker,
    embed_model_st,
    llm_name,
    prompt_base,
    opciones_llm,
    n_rel_max,
    min_score_parcial = 2,
    min_score_fuzzy = 2,
    n_final_fuzz_parc = 3,
    n_resultados_embedding = 3,
    peso_tripleta = 0.7,
    peso_rel = 0.3,
    n_maximos = 3,
    min_score = 0.1,
    ):

    """ 

    """
    
    resultados = {}
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")
    # reranker = CrossEncoder(reranker, max_length=512)
    # embed_model_st = SentenceTransformer(embed_model_st_name)

    # reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

    for idx, registro in enumerate(dataset_2Wiki):
        
        print(f"Registro nº: {idx+1}")

        question = dataset_2Wiki[idx]['question']
        id_reg = dataset_2Wiki[idx]['_id']
        
        # -------------- BUSQUEDA DE ENTIDADES ----------------
        # ner_model = "en_core_web_sm"
        # fulltext_index = "entidadesIndex"
        # min_score_parcial = 2
        # min_score_fuzzy = 2
        # n_final_fuzz_parc = 3
        # n_resultados_embedding = 3
        entidades_finales = extraer_buscar_entidades(con_Neo4j, question, ner_model, fulltext_index, vector_index, embed_model_st, min_score_parcial, min_score_fuzzy, n_final_fuzz_parc, n_resultados_embedding)

        # -------------- EXTRACCION DE RELACIONES ----------------
        # n_saltos = 2
        # score_min_reranker = 0.75
        # n_rel_max = 10
        # peso_tripleta = 0.7
        # peso_rel = 0.3
        # n_maximos = 3
        # min_score = 0.1
        tripletas_finales = extraer_relaciones(con_Neo4j, question, entidades_finales, n_saltos, reranker, n_rel_max, peso_tripleta, peso_rel, n_maximos, min_score)
        
        # -------------- GENERACION DE RESPUESTA ----------------
        respuesta_llm = generar_respuesta(question, tripletas_finales, llm_name, opciones_llm, prompt_base)
        
        # ----------------- EVALUACION RESULTADO -----------------
        ground_truth = dataset_2Wiki[idx]['answer']
        sup_facts = dataset_2Wiki[idx]['supporting_facts']
        em, f1, precision, recall, n_sup_facts, n_ent_in_sup_fact = calcular_metricas_evaluacion(respuesta_llm, ground_truth, entidades_finales, sup_facts)

        # ------------------ OUTPUT RESULTADOS -----------------------
        resultados[id_reg] = {
            'question': question,
            'ground_truth': ground_truth,
            'respuesta_llm': respuesta_llm,
            # 'entidades_encontradas': entidades_encontradas,
            # 'nodos_subgrafo': nodos,
            'entidades_encontradas': entidades_finales,
            'tripletas_formateadas': tripletas_finales,
            'em': em,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            # 'respuesta_en_subgrafo': respuesta_en_subgrafo,
            'Nº supporting_facts en el subgrafo': n_ent_in_sup_fact,
            '% entidades subgrafo en supporting_facts': n_ent_in_sup_fact / n_sup_facts,
        }
    return resultados
