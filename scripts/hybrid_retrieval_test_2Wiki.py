import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.load_data.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.conexion_Neo4j.conexion_Neo4j import ConexionNeo4j
from src.graph_retrieval.funciones_graph_retrieval import (
    extraer_top_k_entities,
    formatear_tripletas,
)
from src.conexion_qdrant.conexion_qdrant import ConexionQdrant
from src.RAG_retrieval.funciones_RAG_retrieval import(
    extraer_info_nodes,
    extraer_info_points,
    filtrar_nodos_por_score,
    textos_para_prompt
)
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





def contestar_2Wiki_con_hibrido(n_registros, database_Neo, vector_index_name, embed_model_st, collection, embed_model_llama, llm_name, prompt_base, opciones_llm):
    """ 

    """
    
    database_Neo4j = ConexionNeo4j(database_Neo)
    qd_client = ConexionQdrant()
    
    resultados = {}

    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")

    for idx, registro in enumerate(dataset_2Wiki):
        
        print(f"Registro nº: {idx+1}")

        question = dataset_2Wiki[idx]['question']
        id_reg = dataset_2Wiki[idx]['_id']
        
        #------------ GRAPH RETRIEVAL ----------------
        entidades_encontradas = database_Neo4j.query_a_embedding(vector_index_name, embed_model_st, question, 5)
        entidades_filtradas = extraer_top_k_entities(entidades_encontradas, 2)
        nodos, relaciones = database_Neo4j.extraer_subgrafo(entidades_filtradas, 2)
        tripletas_formateadas = formatear_tripletas(relaciones)
        
        
        #------------ RAG RETRIEVAL ----------------
        # nodes = qd_client.retriever_quadrant_llama(collection, embed_model_llama, question)
        # nodes_clean = extraer_info_nodes(nodes)
        # DIRECTO A QDRANT
        points = qd_client.query_qdrant(collection, embed_model_st, question, 5)
        points_clean = extraer_info_points(points)
        
        # nodes_filt = filtrar_nodos_por_score(nodes_clean)
        points_filt = filtrar_nodos_por_score(points_clean)
        
        # textos_retrieval = textos_para_prompt(nodes_filt)
        textos_retrieval = textos_para_prompt(points_filt)
        
        
        #------------ COMBINACION Y RESPUESTA ----------------
        info_prompt = {}
        info_prompt['tripletas_formateadas'] = tripletas_formateadas
        info_prompt['textos_retrieval'] = textos_retrieval
        info_prompt['question'] = question
        prompt = build_prompt(prompt_base, info_prompt)
        
        
        #------------ GENERAR RESPUESTA --------------------
        respuesta_llm = llm_funcs.generate(llm_name, prompt, opciones_llm)
        
        
        #------------ EVALUAR RESPUESTA --------------------
        ground_truth = dataset_2Wiki[idx]['answer']
        sup_facts = dataset_2Wiki[idx]['supporting_facts']
        
        em = exact_match_score(respuesta_llm, ground_truth)
        f1, precision, recall = f1_score(respuesta_llm, ground_truth)
        respuesta_en_subgrafo = respuesta_en_nodos_encontrados(nodos, ground_truth)
        n_sup_facts, n_ent_in_sup_fact = suporting_facts_en_subgrafo(nodos, sup_facts)

        resultados[id_reg] = {
            'question': question,
            'ground_truth': ground_truth,
            'respuesta_llm': respuesta_llm,
            'entidades_encontradas': entidades_encontradas,
            'nodos_subgrafo': nodos,
            'tripletas_formateadas': [tripletas_formateadas.split("\n")],
            'em': em,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'respuesta_en_subgrafo': respuesta_en_subgrafo,
            'Nº supporting_facts en el subgrafo': n_ent_in_sup_fact,
            '% entidades subgrafo en supporting_facts': n_ent_in_sup_fact / n_sup_facts,
        }
    # return json.dumps(resultados, indent=4)
    return resultados






prompt_base = """
        You are a question answering system.

        You MUST answer the question using ONLY the data provided below.
        Data providing come from 2 sources: kwnoledge graph and texts.
        Do NOT use any external knowledge.
        If you don't know the answer MUST say only: "I don't know".
        Don't explain your answer.

        Knowledge graph:
        {tripletas_formateadas}

        Texts:
        {textos_retrieval}

        Question:
        {question}

        Answer:
        """


def main():
    comentarios = """
        PRUEBA HIBRIDO
    """
    
    n_registros = 20
    database_Neo = "2wiki.prueba1"
    vector_index_name = "entity_embedding_index"
    collection = "2wikimultihop_prueba1"
    embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embed_model_llama = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",)

    llm_name = "phi3:latest"
    opciones_llm = {
        'temperature': 0,
        # 'num_ctx': 1024,
        # 'num_predict': 600,
    }

    resultados = contestar_2Wiki_con_hibrido(n_registros, database_Neo, vector_index_name, embed_model_st, collection, embed_model_llama, llm_name, prompt_base, opciones_llm)
    
    ruta_resultados = root_dir / "outputs" / "resultados" / "dataset_2Wiki"
    nombre_result = "Hybrid-Graph-RAG_answer_2Wiki"
    resultados_json = guardar_resultados(resultados, nombre_result, ruta_resultados)
    
    metricas_agg = metricas_totales(resultados)
    ruta_registro = root_dir / "outputs" / "registro" / "dataset_2Wiki"
    guardar_registro(ruta_registro, comentarios, n_registros, metricas_agg, llm_name, prompt_base, opciones_llm)
    
    print(metricas_agg)
    

resultados_json = main()