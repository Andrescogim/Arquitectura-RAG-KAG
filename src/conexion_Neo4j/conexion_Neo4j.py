from neo4j import GraphDatabase

class ConexionNeo4j:
    def __init__(self, database):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
            database = database
        )


    def insertar_tripleta(self, subj, obj, rel):
        
        """Inserta tripleta en Neo4j"""
        
        query_base = f"""
            MERGE (a:Entity {{name: $subj}})
            MERGE (b:Entity {{name: $obj}})
            MERGE (a)-[:{rel}]->(b)
        """
        summary = self.driver.execute_query(
            query_base,
            subj = subj,
            obj = obj,
            rel = rel,
            # database_ = database
        )
        return summary


    def ejecutar_query(self, query):
        records, summary, keys = self.driver.execute_query(query)
        return records, summary, keys


    def normalizar_tripleta(self, tripleta, separador):
        """
        Normaliza tripletas a formato correcto para Neo4j
        Tripleta puede ser lista o tupla
        2º Elemento --> Relacion        
        """
        trip = tripleta.split(separador)
        subj = trip[0].replace("–", "_")
        rel_aux = trip[1].replace("–", "_").replace("/", "_")
        rel = f"_{rel_aux}"
        obj = trip[2].replace("–", "_")
        return subj, rel, obj


    def query_a_embedding(self, vector_index_name, embed_model, query, n_resultados):
        
        query_embedding = embed_model.encode(query).tolist()
        query_base = f"""
        CALL db.index.vector.queryNodes(
            $vector_index_name,
            $n_resultados,
            $embedding
        )
        YIELD node, score
        RETURN node.name AS name, score
        """
        result = self.driver.execute_query(
            query_base,
            vector_index_name = vector_index_name,
            n_resultados = n_resultados,
            embedding = query_embedding,
        )
        return result[0]
    
    
    def extraer_subgrafo(self, entidades, n_saltos):
        query_base = """
            MATCH (n:Entity)
            WHERE n.name IN $entidades
            CALL apoc.path.subgraphAll(n, {
                maxLevel: $k
            })
            YIELD nodes, relationships
            RETURN 
            [node IN nodes | {name: node.name}] AS nodes,
            [rel in relationships | {
            origen: startNode(rel).name,
            destino: endNode(rel).name,
            relacion: type(rel)
            }
            ] AS relationships
        """
        subgrafo_raw = self.driver.execute_query(query_base, entidades = entidades, k = n_saltos)
        subgrafo_clean = [record for record in subgrafo_raw]
        # nodes = subgrafo_clean[0][0]['nodes']
        # rels = subgrafo_clean[0][0]['relationships']
        nodes = {}
        rels = {}
        for i, node in enumerate(subgrafo_clean[0]):
            nodes[f"nodos_entidad_{entidades[i]}"] = node['nodes']
            rels[f"relaciones_entidad_{entidades[i]}"] = node['relationships']

        return nodes, rels