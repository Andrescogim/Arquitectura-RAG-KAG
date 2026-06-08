import sys
from pathlib import Path

import spacy

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.conexion_Neo4j import ConexionNeo4j
from src.utils.funciones_deduplicacion import (
    calcular_rank_B25s,
    encontrar_candidatos_iguales_principal,
    get_columnas,
    calcular_distancias,
    excluir_por_numeros,
    excluir_por_nombre,
    filtrar_fusionables,
    unir_candidatos,
    grupos_fusion,
    seleccionar_nodo_principal
)


nlp = spacy.load("en_core_web_sm")
database_Neo = "2wiki.prueba.rebel.5"
conn_Neo4j = ConexionNeo4j(database_Neo)

entidades = conn_Neo4j.extraer_all_entidades_neo4j()
n_candidatos = 3

df_deduplicacion = calcular_rank_B25s(entidades, n_candidatos)
df_deduplicacion = encontrar_candidatos_iguales_principal(df_deduplicacion, n_candidatos)
cols_distancias, cols_exc_num, cols_exc_nom = get_columnas(n_candidatos)
df_deduplicacion[cols_distancias] = df_deduplicacion.apply(calcular_distancias, args=(nlp, n_candidatos), axis = 1, result_type='expand')
df_deduplicacion[cols_exc_num] = df_deduplicacion.apply(excluir_por_numeros, args=(n_candidatos,), axis = 1, result_type='expand')
df_deduplicacion[cols_exc_nom] = df_deduplicacion.apply(excluir_por_nombre, args=(nlp, n_candidatos), axis = 1, result_type='expand')
df_deduplicacion = filtrar_fusionables(df_deduplicacion, n_candidatos)
df_deduplicacion_final = unir_candidatos(df_deduplicacion, n_candidatos)
grupos = grupos_fusion(df_deduplicacion_final)
grados = conn_Neo4j.obtener_grados_nodos(df_deduplicacion_final)
nodos_fusion = seleccionar_nodo_principal(grupos, grados)
nodos_fusionados = conn_Neo4j.fusionar_nodos(nodos_fusion)

print(f"Nº de nodos fusionados: {df_deduplicacion_final.shape[0]}")
print(f"Nº de nodos tras fusion: {nodos_fusionados}")