import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.load_data.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.conexion_Neo4j.conexion_Neo4j import ConexionNeo4j
from src.graph_retrieval.funciones_graph_retrieval import (
    extraer_top_k_entities,
    filtrado_parcial,
    filtrado_fuzzy,
    combinar_entis_parcial_fuzzy,
    union_entidades,
    formatear_tripletas_extendidas,
    reranking_tripletas,
    filtrar_tripletas_reranked,
    reranking_tripletas_aux,
)
from src.funciones_retrieval_generales import extraer_entidades_ner
from src.funciones_generales import build_prompt
from src.LLM_interaction import LLM_interaction_functions as llm_funcs
from src.metricas.metricas_2Wiki import (
    f1_score,
    exact_match_score,
    respuesta_en_nodos_encontrados,
    suporting_facts_en_subgrafo,
    metricas_totales
    )
from src.output_save.funciones_guardado import guardar_resultados, guardar_registro


def extraer_buscar_entidades(con_Neo4j, question, ner_model, fulltext_index, vector_index, embed_model):
    
    # ner_model = "en_core_web_sm"
    entidades_ner = extraer_entidades_ner(question, ner_model)
    # entis_exactas = database_Neo4j.busqueda_exacta_entidades(entidades_ner)
    entis_exactas = con_Neo4j.busqueda_exacta_entidades(entidades_ner)
    # fulltext_index_name = "entidadesIndex"
    res_busqueda_parcial = con_Neo4j.busqueda_parcial_entidades(fulltext_index, entidades_ner)
    res_busqueda_fuzzy = con_Neo4j.busqueda_fuzzy_entidades(fulltext_index, entidades_ner)
            
    min_score_parcial = 2
    filt_busqueda_parcial = filtrado_parcial(res_busqueda_parcial, min_score_parcial)
    min_score_fuzzy = 2
    filt_busqueda_fuzzy = filtrado_fuzzy(res_busqueda_fuzzy, min_score_fuzzy)
    n_final = 3
    entidades_text_index = combinar_entis_parcial_fuzzy(filt_busqueda_parcial, filt_busqueda_fuzzy, n_final)
    
    n_resultados_embedding = 3
    # res_busqueda_embeddings = database_Neo4j.query_a_embedding(vector_index, embed_model, question, n_resultados_embedding)
    res_busqueda_embeddings = con_Neo4j.query_a_embedding(vector_index, embed_model, question, n_resultados_embedding)
    entidades_embeddings = extraer_top_k_entities(res_busqueda_embeddings, 2)
    
    entidades_finales = union_entidades(entis_exactas, entidades_text_index, entidades_embeddings)
    
    return entidades_finales


def extraer_relaciones(con_Neo4j, question, entidades_finales, n_saltos, reranker):
    
    # n_saltos = 2
    subgrafo = con_Neo4j.extraer_subgrafo_completo(entidades_finales, n_saltos)
    tripletas_formateadas = formatear_tripletas_extendidas(subgrafo)
    tripletas_reranked = reranking_tripletas(question, tripletas_formateadas, reranker)
    tripletas_finales = filtrar_tripletas_reranked(tripletas_reranked)
    if len(tripletas_finales) < 2 or len(tripletas_finales) > 8:
        tripletas_finales = reranking_tripletas_aux(question, tripletas_formateadas, reranker)

    return tripletas_finales


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




def contestar_2Wiki_con_grafo(n_registros, database_Neo, vector_index, embed_model_st, llm_name, prompt_base, opciones_llm):
    """ 

    """
    
    con_Neo4j = ConexionNeo4j(database_Neo)
    resultados = {}
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

    for idx, registro in enumerate(dataset_2Wiki):
        
        print(f"Registro nº: {idx+1}")

        question = dataset_2Wiki[idx]['question']
        id_reg = dataset_2Wiki[idx]['_id']
        
        # -------------- BUSQUEDA DE ENTIDADES ----------------
        """ner_model = "en_core_web_sm"
        entidades_ner = extraer_entidades_ner(question, ner_model)
        entis_exactas = database_Neo4j.busqueda_exacta_entidades(entidades_ner)
        fulltext_index_name = "entidadesIndex"
        res_busqueda_parcial = database_Neo4j.busqueda_parcial_entidades(fulltext_index_name, entidades_ner)
        res_busqueda_fuzzy = database_Neo4j.busqueda_fuzzy_entidades(fulltext_index_name, entidades_ner)
             
        min_score_parcial = 2
        filt_busqueda_parcial = filtrado_parcial(res_busqueda_parcial, min_score_parcial)
        min_score_fuzzy = 2
        filt_busqueda_fuzzy = filtrado_fuzzy(res_busqueda_fuzzy, min_score_fuzzy)
        n_final = 3
        entidades_text_index = combinar_entis_parcial_fuzzy(filt_busqueda_parcial, filt_busqueda_fuzzy, n_final)
        
        n_resultados_embedding = 3
        res_busqueda_embeddings = database_Neo4j.query_a_embedding(vector_index, embed_model_st, question, n_resultados_embedding)
        entidades_embeddings = extraer_top_k_entities(res_busqueda_embeddings, 2)
        
        entidades_finales = union_entidades(entis_exactas, entidades_text_index, entidades_embeddings)"""

        ner_model = "en_core_web_sm"
        fulltext_index = "entidadesIndex"
        entidades_finales = extraer_buscar_entidades(con_Neo4j, question, ner_model, fulltext_index, vector_index, embed_model_st)

        # -------------- EXTRACCION DE RELACIONES ----------------
        """n_saltos = 2
        subgrafo = database_Neo4j.extraer_subgrafo_completo(entidades_finales, n_saltos)
        tripletas_formateadas = formatear_tripletas_extendidas(subgrafo)
        tripletas_reranked = reranking_tripletas(question, tripletas_formateadas, reranker)
        tripletas_finales = filtrar_tripletas_reranked(tripletas_reranked)
        if len(tripletas_finales) < 2 or len(tripletas_finales) > 8:
            tripletas_finales = reranking_tripletas_aux(question, tripletas_formateadas, reranker)"""
        
        n_saltos = 2
        tripletas_finales = extraer_relaciones(con_Neo4j, question, entidades_finales, n_saltos, reranker)
        
        # -------------- GENERACION DE RESPUESTA ----------------
        """info_prompt = {}
        info_prompt['tripletas_formateadas'] = "\n".join(tripletas_finales)
        info_prompt['question'] = question
        prompt = build_prompt(prompt_base, info_prompt)
        respuesta_llm = llm_funcs.generate(llm_name, prompt, opciones_llm)"""
        
        respuesta_llm = generar_respuesta(question, tripletas_finales, llm_name, opciones_llm, prompt_base)
        
        # ----------------- EVALUACION RESULTADO -----------------
        ground_truth = dataset_2Wiki[idx]['answer']
        sup_facts = dataset_2Wiki[idx]['supporting_facts']
        
        """em = exact_match_score(respuesta_llm, ground_truth)
        f1, precision, recall = f1_score(respuesta_llm, ground_truth)
        # respuesta_en_subgrafo = respuesta_en_nodos_encontrados(entidades_finales, ground_truth)
        n_sup_facts, n_ent_in_sup_fact = suporting_facts_en_subgrafo(entidades_finales, sup_facts, 1)"""
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




prompt_base = """
        You are a question answering system.

        You MUST answer the question using ONLY the data of the knowledge graph provided below.
        Do NOT use any external knowledge.
        If you don't know the answer MUST say only: "I don't know".
        Don't explain your answer.

        Knowledge graph:
        {tripletas_formateadas}

        Question:
        {question}

        Answer:
        """


def main():
    comentarios = """
        PRUEBA
    """
    
    n_registros = 300
    database_Neo = "2wiki.prueba2"
    vector_index = "entity_embedding_index"
    embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    llm_name = "phi3:latest"
    opciones_llm = {
        'temperature': 0,
        # 'num_ctx': 1024,
        # 'num_predict': 600,
    }


    resultados = contestar_2Wiki_con_grafo(n_registros, database_Neo, vector_index, embed_model_st, llm_name, prompt_base, opciones_llm)
    
    ruta_resultados = root_dir / "outputs" / "resultados" / "dataset_2Wiki"
    nombre_result = "graph_answer_2Wiki"
    resultados_json = guardar_resultados(resultados, nombre_result, ruta_resultados)
    
    metricas_agg = metricas_totales(resultados)
    ruta_registro = root_dir / "outputs" / "registro" / "dataset_2Wiki"
    guardar_registro(ruta_registro, comentarios, n_registros, metricas_agg, llm_name, prompt_base, opciones_llm)
    
    print(metricas_agg)
    

resultados_json = main()