import json
import time
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
import pandas as pd
from src.utils.funciones_generales import build_prompt
from src.utils.LLM_interaction import LLM_interaction_functions as llm_funcs
from src.utils.metricas.metricas_2Wiki import (
    f1_score,
    exact_match_score,
    respuesta_en_nodos_encontrados,
    suporting_facts_en_subgrafo,
    metricas_totales
    )
from src.utils.funciones_generales import medir_recursos



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

@medir_recursos
def contestar_2Wiki_con_grafo_from_csvs(
    con_Neo4j,
    # n_registros,
    split,
    rango_in_data,
    rango_fin_data,
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
    min_score_fuzzy = 3,
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
    resultados_recursos = []

    # dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")
    subset = f"{split}[{rango_in_data}:{rango_fin_data + 1}]"
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_subset = None, split = subset)
    iteracion = rango_in_data
    for idx, registro in enumerate(dataset_2Wiki):
        print("-"*30)
        print(f"Registro nº: {idx+1}")
        

        question = dataset_2Wiki[idx]['question']
        id_reg = dataset_2Wiki[idx]['_id']
        
        ruta_csvs = "outputs/caminos_retrieval/rebel.500.nac.ded/"
        csv_file = f"caminos_finales_{iteracion}.csv"
        subgrafo_df_reranked_filt = pd.read_csv(f"{ruta_csvs}{csv_file}")
        tripletas_finales = list(subgrafo_df_reranked_filt["tripleta_formateada"])
        # -------------- GENERACION DE RESPUESTA ----------------
        print("Generando respuesta")

        respuesta_llm = generar_respuesta(question, tripletas_finales, llm_name, opciones_llm, prompt_base)
        
        # ----------------- EVALUACION RESULTADO -----------------
        ground_truth = dataset_2Wiki[idx]['answer']
        sup_facts = json.loads(dataset_2Wiki[idx]['supporting_facts'])
        evidences = json.loads(dataset_2Wiki[idx]['evidences'])
        question_type = dataset_2Wiki[idx]['type']
        entidades_finales = []
        em, f1, precision, recall, n_sup_facts, n_ent_in_sup_fact = calcular_metricas_evaluacion(respuesta_llm, ground_truth, entidades_finales, sup_facts)

        # ------------------ OUTPUT RESULTADOS -----------------------
        resultados[id_reg] = {
            'question': question,
            'question_type': question_type,
            'ground_truth': ground_truth,
            'respuesta_llm': respuesta_llm,
            # 'entidades_encontradas': entidades_encontradas,
            # 'nodos_subgrafo': nodos,
            # 'entidades_encontradas': entidades_finales,
            'tripletas_formateadas': tripletas_finales,
            'em': em,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            # 'respuesta_en_subgrafo': respuesta_en_subgrafo,
            # 'supporting_facts': sup_facts,
            'evidences': evidences,
            'Nº supporting_facts en el subgrafo': n_ent_in_sup_fact,
            '% entidades subgrafo en supporting_facts': n_ent_in_sup_fact / n_sup_facts,
        }

        iteracion += 1
    return resultados
