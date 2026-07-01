import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.conexion_Neo4j import ConexionNeo4j
from src.utils.funciones_guardado import guardar_resultados, guardar_registro
from src.logica.graph_retrieval import contestar_2Wiki_con_grafo
from src.logica.graph_retrieval_from_files import contestar_2Wiki_con_grafo_from_csvs
from src.utils.metricas.metricas_2Wiki import metricas_totales
from src.utils.funciones_generales import medir_recursos



prompt_base = """
        You are a strict question-answering assistant. 

        1. You MUST answer the question using ONLY the facts provided in the [knowledge graph] section.
        2. Do NOT use any external knowledge or assume anything.
        3. Be direct and concise. Do NOT explain your answer. Give only the exact name, place, date, or "Yes"/"No" as requested.
        4. If the [Knowledge graph] does not contain enough information to answer the question, you MUST reply exactly with: "I don't know".
        
        [Knowledge graph]:
        {tripletas_formateadas}

        Question:
        {question}

        Answer:
        """


def main():
    comentarios = """
        
    """
    
    database_Neo = "2wiki.rebel.500.nac.ded"
    opciones_llm = {
        'temperature': 0,
        # 'num_ctx': 1024,
        # 'num_predict': 600,
    }
    con_Neo4j = ConexionNeo4j(database_Neo)
    # reranker_model = "BAAI/bge-reranker-base"
    reranker_model = "BAAI/bge-reranker-v2-m3"
    reranker = CrossEncoder(reranker_model, max_length=512)
    embed_model_name = "BAAI/bge-small-en-v1.5"
    embed_model_st = SentenceTransformer(embed_model_name)
    ner = spacy.load("en_core_web_sm")
    models = ['qwen2.5:3b-instruct', 'qwen2.5:7b-instruct', 'gemma:7b', 'llama3:latest', 'phi3:latest', 'mistral:latest', 'deepseek-r1:8b']
    # models = ["gemma3:4b", "phi4:14b", 'qwen2.5:3b-instruct', 'qwen2.5:7b-instruct', 'gemma:7b', 'llama3:latest', 'phi3:latest',]
    for model in models:
        parametros = {
            # "n_registros": 100,
            "split": "train",
            "rango_in_data": 0,
            "rango_fin_data": 500,
            "reranker": reranker,
            "vector_index": "entity_embedding_index",
            "ner_model": ner,
            "fulltext_index": "entidadesIndex",
            "n_saltos": 2,
            "score_min_reranker": 0.75,
            "embed_model_st": embed_model_st,
            "llm_name": model,
            # "llm_name": "qwen2.5:7b-instruct",
            # "llm_name": "gemma:7b",
            "prompt_base": prompt_base,
            "opciones_llm": opciones_llm,
            "n_rel_max": 35,
            "min_score_parcial": 2,
            "min_score_fuzzy": 2,
            "n_final_fuzz_parc": 3,
            "n_resultados_embedding": 3,
            "peso_tripleta": 0.65,
            "peso_rel": 0.35,
            "n_maximos": 3,
            "min_score": 0.3,
        }
        
        # resultados = contestar_2Wiki_con_grafo(con_Neo4j, **parametros)
        resultados = contestar_2Wiki_con_grafo_from_csvs(con_Neo4j, **parametros)
        metricas_agg = metricas_totales(resultados)
        
        # ruta_resultados = root_dir / "outputs" / "resultados" / "dataset_2Wiki" / "solo_grafo"
        # ruta_registro = root_dir / "outputs" / "registro" / "dataset_2Wiki" / "solo_grafo"
        ruta_resultados = root_dir / "outputs" / "resultados" / "dataset_2Wiki" / "definitivo"
        ruta_registro = root_dir / "outputs" / "registro" / "dataset_2Wiki" / "definitivo"
        llm_name = parametros["llm_name"].replace(":", "-")
        nombre_result = f"graph_answer_DB_{database_Neo}_{llm_name}_NREG_{parametros['rango_in_data']}-{parametros['rango_fin_data']}"
        
        resultados_json = guardar_resultados(resultados, nombre_result, ruta_resultados)
        parametros_registro = [
            "split",
            "n_saltos",
            "score_min_reranker",
            "min_score_parcial",
            "min_score_fuzzy",
            "n_final_fuzz_parc",
            "n_resultados_embedding",
            "peso_tripleta",
            "peso_rel",
            "n_maximos",
            "min_score",
            ]
        dic_param_reg = {k:v for k,v in parametros.items() if k in parametros_registro}
        dic_param_reg["reranker"] = reranker_model
        dic_param_reg["embed_model"] = embed_model_name
        guardar_registro(
            ruta_registro,
            comentarios,
            # parametros["n_registros"],
            parametros["rango_in_data"],
            parametros["rango_fin_data"],
            metricas_agg,
            parametros["llm_name"],
            prompt_base,
            opciones_llm,
            dic_param_reg
            )
        
        print(metricas_agg)

        ruta_medicion = root_dir / "outputs" / "medicion_recursos"
        nombre_result = nombre_result
        _ = guardar_resultados(medir_recursos.acumulado, nombre_result, ruta_medicion)
        print(f"MODELO: {model} FINALIZADO Y GUARDADOS RESULTADOS")
        print(f"-"*40)

if __name__ == "__main__":
    main()
# resultados_json = main()