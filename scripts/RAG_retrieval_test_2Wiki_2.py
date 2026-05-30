import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.load_data.funciones_carga_datos import load_filter_dataset_HuggingFace
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
    metricas_totales
    )
from src.output_save.funciones_guardado import guardar_resultados, guardar_registro




def contestar_2Wiki_con_RAG(n_registros, collection, embed_model, embed_model_st, llm, llm_name, prompt_base, opciones_llm):
    """ 

    """
    # qd_client = QdrantClient(url="http://localhost:6333")
    qd_client = ConexionQdrant()
    
    resultados = {}

    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")

    for idx, registro in enumerate(dataset_2Wiki):
        
        print(f"Registro nº: {idx+1}")

        question = dataset_2Wiki[idx]['question']
        id_reg = dataset_2Wiki[idx]['_id']
        
        # CON LLAMAINDEX
        # nodes, response = query_quadrant_llama(qd_client, collection, embed_model, llm, question)
        nodes = qd_client.retriever_quadrant_llama(collection, embed_model, question)
        nodes_clean = extraer_info_nodes(nodes)
        
        # DIRECTO A QDRANT
        points = qd_client.query_qdrant(collection, embed_model_st, question, 5)
        points_clean = extraer_info_points(points)
        
        nodes_filt = filtrar_nodos_por_score(nodes_clean)
        points_filt = filtrar_nodos_por_score(points_clean)

        # textos_retrieval = textos_para_prompt(points_filt)
        textos_retrieval = textos_para_prompt(nodes_filt)
    
        info_prompt = {}
        info_prompt['textos_retrieval'] = textos_retrieval
        info_prompt['question'] = question
        prompt = build_prompt(prompt_base, info_prompt)
        respuesta_llm = llm_funcs.generate(llm_name, prompt, opciones_llm)
        
        ground_truth = dataset_2Wiki[idx]['answer']
        sup_facts = dataset_2Wiki[idx]['supporting_facts']
        
        em = exact_match_score(respuesta_llm, ground_truth)
        f1, precision, recall = f1_score(respuesta_llm, ground_truth)

        resultados[id_reg] = {
            'question': question,
            'ground_truth': ground_truth,
            'respuesta_llm': respuesta_llm,
            'nodos_recuperados': nodes_filt,
            # 'nodos_recuperados': points_filt,
            'em': em,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    # return json.dumps(resultados, indent=4)
    return resultados






prompt = """

You are a question answering system.

You MUST answer the question using the data ONLY from RETRIEVAL SYSTEM provided below.
Do NOT use any external knowledge.
If the answer is not present, say "I don't know".

[RETRIEVAL]:
{textos_retrieval}

Question:
{question}

Answer:
"""


def main():
    comentarios = """
        PRUEBA
    """
    
    n_registros = 10
    collection = "2wikimultihop_prueba1"
    embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",)
    llm = Ollama(model="gemma:2b")
    llm_name = "phi3:latest"
    opciones_llm = {
        'temperature': 0,
        # 'num_ctx': 1024,
        # 'num_predict': 600,
    }


    resultados = contestar_2Wiki_con_RAG(n_registros, collection, embed_model, embed_model_st, llm, llm_name, prompt, opciones_llm)
    
    ruta_resultados = root_dir / "outputs" / "resultados" / "dataset_2Wiki"
    nombre_result = "RAG_answer_2Wiki"
    resultados_json = guardar_resultados(resultados, nombre_result, ruta_resultados)
    
    metricas_agg = metricas_totales(resultados)
    ruta_registro = root_dir / "outputs" / "registro" / "dataset_2Wiki"
    guardar_registro(ruta_registro, comentarios, n_registros, metricas_agg, llm_name, prompt, opciones_llm)
    
    print(metricas_agg)
    

resultados_json = main()