import sys
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.utils.conexion_Neo4j import ConexionNeo4j
from src.utils.funciones_construir_grafo import(
    limpiar_tripletas,
    extraccion_tripletas_2wiki_rebel,
    # tripletas_from_evidences_2Wiki_to_neo4j
)


def main():
    n_registros = 2
    database_Neo = "2wiki.prueba.rebel.5"
    reemplazar_database_neo = True
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to("cuda")
    n_window = 4 # Ventana busqueda tripletas nacionalidad
    nlp = spacy.load("en_core_web_sm")
    embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vector_index_name = "entity_embedding_index"
    vec_dim_index = 384
    similarity_func_index = 'cosine'
    text_index_name = "entidadesIndex"
    
    """# parametros = {
    #     n_registros: 2,
    #     "database_Neo": "2wiki.prueba.rebel.5",
    #     "reemplazar_database_neo": False,
    #     "tokenizer_extraccion_tripletas": "Babelscape/rebel-large",
    #     "model_extraccion_tripletas": "Babelscape/rebel-large",
    #     "n_window": 4, # Ventana busqueda tripletas nacionalidad
    #     "modelo_spacy": "en_core_web_sm",
    #     "embed_model_st": "BAAI/bge-small-en-v1.5",
    #     "vector_index_name": "entity_embedding_index",
    #     "vec_dim_index": 384,
    #     "similarity_func_index": 'cosine',
    #     "text_index_name": "entidadesIndex",
        
    # }
    # tokenizer = AutoTokenizer.from_pretrained(parametros["tokenizer_extraccion_tripletas"])
    # model = AutoModelForSeq2SeqLM.from_pretrained(parametros["model_extraccion_tripletas"]).to("cuda")
    # "nlp" = spacy.load(parametros["modelo_spacy"])
    # "embed_model_st" = SentenceTransformer(parametros["embed_model_st"])"""
    
    
    # ------------------ OBTENCION DATASET ----------------------
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")
    print("OBTENCION DATASET OK")
    
    # ------------------ CONEXION NEO4J ----------------------
    conn_Neo4j = ConexionNeo4j(database_Neo)
    
    # ----------------- CREAR LA DATABASE -------------------
    if reemplazar_database_neo == True:
        conn_Neo4j.crear_reemplazar_database()
    else:
        conn_Neo4j.crear_database()
    print("CREACION DATABASE EN Neo4j OK")

    # ----------------- EXTRACCION DE TRIPLETAS -----------------
    print("COMENZANDO EXTRACCION Y CARGA DE TRIPLETAS...")
    # TRIPLETAS DIRECTAMENTE DE LAS EVIDENCIAS
    # sumary = tripletas_from_evidences_2Wiki_to_neo4j(conn_Neo4j, dataset_2Wiki)

    # TRIPLETAS EXTRAIDAS CON REBEL
    tripletas_all, all_triples_nacionalidad = extraccion_tripletas_2wiki_rebel(dataset_2Wiki, tokenizer, model, nlp, n_window)
    
    # ----------------- INSERCION DE TRIPLETAS EN NEO4J -----------------
    tripletas_limpias = list(set(limpiar_tripletas(tripletas_all)))
    tripletas_limpias_nac = list(set(limpiar_tripletas(all_triples_nacionalidad)))
    summary = conn_Neo4j.insertar_triplets_batch(tripletas_limpias)
    summary = conn_Neo4j.insertar_triplets_batch(tripletas_limpias_nac)
    print("EXTRACCION Y CARGA DE TRIPLETAS OK")

    # ----------------- CREACION DE EMBEDDINGS Y TEXT INDEX -----------------
    print("COMENZANDO CREACION DE EMBEDDINGS Y TEXT INDEX EN Neo4j")
    entidades = conn_Neo4j.extraer_all_entidades_neo4j()
    sumary = conn_Neo4j.añadir_embeddings_como_propiedad_neo4j(entidades, embed_model_st)
    sumary = conn_Neo4j.crear_vector_index_neo4j(vector_index_name, vec_dim_index, similarity_func_index)
    sumary = conn_Neo4j.crear_fulltext_index(text_index_name)
    print("CREACION DE EMBEDDINGS Y TEXT INDEX EN Neo4j OK ")


if __name__ == "__main__":
    main()