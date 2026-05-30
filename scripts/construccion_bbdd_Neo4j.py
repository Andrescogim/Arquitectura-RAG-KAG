import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.load_data.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.conexion_Neo4j.conexion_Neo4j import ConexionNeo4j



def tripletas_from_evidences_2Wiki_to_neo4j(con_Neo, data_2wiki):
    
    for el in data_2wiki:
        for tripleta in eval(el["evidences"]):
            subj = tripleta[0]
            obj = tripleta[2]
            rel = tripleta[1].replace(" ", "_")
            # sumary = insert_triplet(driver, "2wiki.prueba1", subj, obj, rel)
            sumary = con_Neo.insertar_tripleta(subj, obj, rel)
    return sumary


n_registros = 200
dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_registros, "train")
database_Neo = "2wiki.prueba1"
database_Neo4j = ConexionNeo4j(database_Neo)
sumary = tripletas_from_evidences_2Wiki_to_neo4j(database_Neo4j, dataset_2Wiki)

entidades = ConexionNeo4j.extraer_entidades_neo4j()

embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")
sumary = ConexionNeo4j.añadir_embeddings_como_propiedad_neo4j(entidades, embed_model_st)
vector_index_name = "entity_embedding_index"
sumary = ConexionNeo4j.crear_vector_index_neo4j(vector_index_name)


