import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.conexion_Neo4j import ConexionNeo4j
from src.utils.funciones_construir_grafo import(
    obtencion_entidades_de_tripletas,
)
from src.utils.funciones_generales import(
    lectura_json,
)


def main():
    
    database_Neo = "2wiki.rebel.gold.500"
    embed_model_st = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # ------------------ CONEXION NEO4J ----------------------
    conn_Neo4j = ConexionNeo4j(database_Neo)
    
    # ------------------ CARGA TRIPLETAS ----------------------
    ruta_tripletas = root_dir / "outputs" / "tripletas_generadas" / "dataset_2Wiki"


    print("LEYENDO TRIPLETAS")
    archivos = [
        "tripletas_Rebel_2Wiki_registros_0-100_2026-06-12_08-16.json",
        "tripletas_Rebel_2Wiki_registros_101-200_2026-06-13_07-50.json",
        "tripletas_Rebel_2Wiki_registros_201-300_2026-06-13_21-54.json",
        "tripletas_Rebel_2Wiki_registros_301-400_2026-06-14_05-00.json",
        "tripletas_Rebel_2Wiki_registros_401-500_2026-06-14_10-52.json",
    ]
    tripletas_all = []
    for archivo in archivos:
        tripletas = lectura_json(ruta_tripletas, archivo)
        tripletas_all += tripletas

    # ------------------ INSERCION TRIPLETAS ----------------------
    print("INSERTANDO TRIPLETAS EN NEO4J")
    summary = conn_Neo4j.insertar_triplets_batch(tripletas_all)

    # ----------------- CREACION DE EMBEDDINGS -----------------
    print("CREACION DE EMBEDDINGS Neo4j")

    entidades = obtencion_entidades_de_tripletas(tripletas_all)
    sumary = conn_Neo4j.añadir_embeddings_como_propiedad_neo4j(entidades, embed_model_st)
    print("INSERCION DE TRIPLETAS FINALIZADA")


if __name__ == "__main__":
    main()