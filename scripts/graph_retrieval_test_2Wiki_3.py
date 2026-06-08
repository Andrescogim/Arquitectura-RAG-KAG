import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.conexion_Neo4j import ConexionNeo4j
from src.utils.funciones_guardado import guardar_resultados, guardar_registro
from src.logica.graph_retrieval import contestar_2Wiki_con_grafo
from src.utils.metricas.metricas_2Wiki import metricas_totales


prompt_base = """
        You are a question answering system.

        You MUST answer the question using ONLY the data of the [knowledge graph] provided below.
        Do NOT use any external knowledge.
        Don't explain your answer.
        If is possible answer JUST yes or no.
        If you don't know the answer based on the [Knowledge graph] MUST say only: "I don't know".
        
        [Knowledge graph]:
        {tripletas_formateadas}

        Question:
        {question}

        Answer:
        """


def main():
    comentarios = """
        Tripletas con REBEL. Limipias. Rerankr con relaciones mas ponderacion
    """
    
    database_Neo = "2wiki.prueba.rebel.5"
    # # database_Neo = "2wiki.prueba1"
    # # database_Neo = "2wiki.prueba.rebel"
    opciones_llm = {
        'temperature': 0,
        # 'num_ctx': 1024,
        # 'num_predict': 600,
    }
    con_Neo4j = ConexionNeo4j(database_Neo)
    reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
    embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    parametros = {
        "n_registros": 2,
        "reranker": reranker,
        "vector_index": "entity_embedding_index",
        "ner_model": "en_core_web_sm",
        "fulltext_index": "entidadesIndex",
        "n_saltos": 2,
        "score_min_reranker": 0.75,
        "embed_model_st": embed_model_st,
        "llm_name": "phi3:latest",
        "prompt_base": prompt_base,
        "opciones_llm": opciones_llm,
        "n_rel_max": 10,
        "min_score_parcial": 2,
        "min_score_fuzzy": 2,
        "n_final_fuzz_parc": 3,
        "n_resultados_embedding": 3,
        "peso_tripleta": 0.7,
        "peso_rel": 0.3,
        "n_maximos": 3,
        "min_score": 0.1,
    }
    
    resultados = contestar_2Wiki_con_grafo(con_Neo4j, **parametros)
    metricas_agg = metricas_totales(resultados)
    
    ruta_resultados = root_dir / "outputs" / "resultados" / "dataset_2Wiki" / "solo_grafo"
    ruta_registro = root_dir / "outputs" / "registro" / "dataset_2Wiki" / "solo_grafo"
    nombre_result = "graph_answer_2Wiki"
    
    resultados_json = guardar_resultados(resultados, nombre_result, ruta_resultados)
    guardar_registro(
        ruta_registro,
        comentarios,
        parametros["n_registros"],
        metricas_agg,
        parametros["llm_name"],
        prompt_base,
        opciones_llm
        )
    
    print(metricas_agg)


if __name__ == "__main__":
    main()
# resultados_json = main()