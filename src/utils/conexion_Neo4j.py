from neo4j import GraphDatabase

class ConexionNeo4j:
    def __init__(self, database):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
        )
        self.database = database


    def crear_database(self):
        query = f"""
            CREATE DATABASE $db IF NOT EXISTS WAIT
        """
        # WAIT para que espere hasta que este creada la db
        summary = self.driver.execute_query(
            query,
            db = self.database,
            database_ = "system"
        )


    def crear_reemplazar_database(self):
        query = f"""
            CREATE OR REPLACE DATABASE $db WAIT
        """
        # WAIT para que espere hasta que este creada la db
        summary = self.driver.execute_query(
            query,
            db = self.database,
            database_ = "system"
        )


    def extraer_all_entidades_neo4j(self):
        query = """
            MATCH (n:Entity)
            RETURN n.name AS name
        """
        entidades = self.driver.execute_query(
            query,
            database_ = self.database
            )
        entidades_name = []
        for entidad in entidades[0]:
            entidades_name.append(entidad['name'])
        return entidades_name
    
    
    
    def extraer_all_relaciones_neo4j(self):
        query = """
            MATCH ()-[r]->()
            RETURN COLLECT(DISTINCT type(r)) AS relationshipTypes
        """
        relaciones = self.driver.execute_query(
            query,
            database_ = self.database
            )
        # relaciones_types = []
        # for relacion in relaciones[0]:
        #     relaciones_types.append(relacion['name'])
        return relaciones


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
            database_ = self.database
        )
        return summary


    def ejecutar_query(self, query):
        records, summary, keys = self.driver.execute_query(
            query,
            database_ = self.database
            )
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
            database_ = self.database,
        )
        return result[0]
    
    

    def busqueda_exacta_entidades(self, entidades):
        
        query = """
            MATCH (n:Entity)
            WHERE n.name IN $entis
            RETURN n.name
        """
        records, summary, keys = self.driver.execute_query(
            query,
            entis = entidades,
            database_ = self.database
            )
        entis_exactas = [record["n.name"] for record in records]
        return entis_exactas




    def busqueda_parcial_entidades(self, index_name, entidades):
        
        res_busqueda_parcial = {}
        for ent in entidades:
            query = """
                CALL db.index.fulltext.queryNodes($index_name, $entidad) 
                YIELD node, score
                RETURN node.name, score
            """
            records, summary, keys = self.driver.execute_query(
                query,
                index_name = index_name,
                entidad = ent,
                database_ = self.database
                )
            resultado = [{"name": record['node.name'], "score": record['score']} for record in records]
            res_busqueda_parcial[ent] = resultado
        return res_busqueda_parcial





    def busqueda_fuzzy_entidades(self, index_name, entidades):
        
        res_busqueda_fuzzy = {}
        for ent in entidades:
            # ent_fuzzy = f"{ent}~"
            ent_fuzzy = ent + "~"
            query = """
                CALL db.index.fulltext.queryNodes($index_name, $entidad) 
                YIELD node, score
                RETURN node.name, score
            """
            records, summary, keys = self.driver.execute_query(
                query,
                index_name = index_name,
                entidad = ent_fuzzy,
                database_ = self.database
                )
            # res_busqueda_fuzzy[ent] = records
            resultado = [{"name": record['node.name'], "score": record['score']} for record in records]
            res_busqueda_fuzzy[ent] = resultado
        return res_busqueda_fuzzy




    
    def extraer_subgrafo(self, entidades, n_saltos):
        """
        Extrae todas las relaciones de las entidades hasta vecinos de 2 saltos
        SOLO tripletas (2 entidades-1relacion)
        """
        
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
        subgrafo_raw = self.driver.execute_query(
            query_base,
            entidades = entidades,
            k = n_saltos,
            database_ = self.database)
        subgrafo_clean = [record for record in subgrafo_raw]
        # nodes = subgrafo_clean[0][0]['nodes']
        # rels = subgrafo_clean[0][0]['relationships']
        nodes = {}
        rels = {}
        for i, node in enumerate(subgrafo_clean[0]):
            nodes[f"nodos_entidad_{entidades[i]}"] = node['nodes']
            rels[f"relaciones_entidad_{entidades[i]}"] = node['relationships']

        return nodes, rels


    def extraer_subgrafo_completo_1(self, entidades, n_saltos):
        """
        Extrae todas las relaciones de las entidades hasta vecinos de 2 saltos
        TODOS los saltos - caminos completos DESDE entidad origen
        """
        query_base = f"""
            MATCH path = (n:Entity)-[*1..{n_saltos}]-(m)
            WHERE n.name IN $entis
            RETURN
                elementId(n) AS ID,
                n.name AS entidad_inicial,
                [node IN nodes(path) | node.name] AS nodos_names,
                [rel IN relationships(path) | type(rel)] AS relaciones_types
            """

        subgrafo_raw, summary, key = self.driver.execute_query(
            query_base,
            entis = entidades,
            database_ = self.database
            )
        lista_relaciones = []
        for rec in subgrafo_raw:
            relacion = [rec["entidad_inicial"]]
            for rel, node in zip(rec["relaciones_types"], rec["nodos_names"][1:]):
                relacion.append(rel)
                relacion.append(node)
            lista_relaciones.append(relacion)

        return lista_relaciones

    def extraer_subgrafo_completo_2(self, entidades, n_saltos):
        """
        Extrae todas las relaciones de las entidades hasta vecinos de 2 saltos
        TODOS los saltos - caminos completos DESDE entidad origen
        """
        # driver_aux = GraphDatabase.driver(
        #     "bolt://localhost:7687",
        #     auth=("neo4j", "password"),
        #     database = database_Neo
        # )
        query_base = f"""
            MATCH path = (n:Entity)-[*1..{n_saltos}]-(m)
            WHERE n.name IN $entis
            RETURN
                elementId(n) AS ID,
                n.name AS entidad_inicial,
                [node IN nodes(path) | node.name] AS nodos_names,
                [rel IN relationships(path) | type(rel)] AS relaciones_types
            """

        subgrafo_raw, summary, key = self.driver.execute_query(
            query_base,
            entis = entidades,
            database_ = self.database
            )
        tripletas_extendidas = {}
        for rec in subgrafo_raw:
            relacion = [rec["entidad_inicial"]]
            for rel, node in zip(rec["relaciones_types"], rec["nodos_names"][1:]):
                relacion.append(rel)
                relacion.append(node)
            tripletas_extendidas.setdefault(rec['entidad_inicial'], []).append(relacion)
        return tripletas_extendidas


    def extraer_subgrafo_completo(self, entidades, n_saltos):
        query_subgrafo = f"""
            MATCH path = (n:Entity)-[*1..{n_saltos}]-(m)
            WHERE n.name IN $entis
            RETURN
                elementId(n) AS ID,
                n.name AS entidad_inicial,
                [node IN nodes(path) | node.name] AS nodos_names,
                [node IN nodes(path) | COUNT{{(node)--()}}] AS degrees,
                [rel IN relationships(path) | type(rel)] AS relaciones_types,
                [i IN range(0, size(relationships(path)) - 1) |
                CASE 
                    WHEN startNode(relationships(path)[i]) = nodes(path)[i]
                    THEN "OUT"
                    ELSE "IN"
                END
                ] AS relaciones_direccion
            """
        subgrafo_raw, summary, keys = self.driver.execute_query(
            query_subgrafo,
            entis = entidades,
            k = 2,
            database_ = self.database
            )
        return subgrafo_raw



    def añadir_embeddings_como_propiedad_neo4j(self, entidades, embed_model):

        entidades_embeddings = embed_model.encode(entidades)
        # Lista de diccionarios para Neo4j
        entis_embeddings_list = [{'name': k, "embedding":v } for k,v in zip(entidades, entidades_embeddings.tolist())]
        query_embeddings = """
            UNWIND $data AS row
            MATCH(n:Entity {name: row.name})
            SET n.embedding = row.embedding
        """
        sumary = self.driver.execute_query(
            query_embeddings,
            data = entis_embeddings_list,
            database_ = self.database
            )
        return sumary
    
    
    def crear_vector_index_neo4j(self, vector_index_name, n_vector, similarity_function):
    
        query_crear_embeddings = """
            CREATE VECTOR INDEX $vector_index_name IF NOT EXISTS
            FOR (n:Entity)
            ON (n.embedding)
            OPTIONS {
            indexConfig: {
                `vector.dimensions`: $n_vector,
                `vector.similarity_function`: $similarity_function
                }
            }
        """
        sumary = self.driver.execute_query(
            query_crear_embeddings,
            vector_index_name = vector_index_name,
            n_vector = n_vector,
            similarity_function = similarity_function,
            database_ = self.database
            )
        return sumary
    


    def crear_fulltext_index(self, index_name):
        
        query = """
            CREATE FULLTEXT INDEX $index_name FOR (n:Entity) ON EACH [n.name]
        """
        summary = self.driver.execute_query(
            query,
            index_name = index_name,
            database_ = self.database
            )
        return summary


    def insertar_triplets_batch(self, tripletas):
        
        """
        Inserta tripletas en Neo4j en batch.
        input: tripletas -> lista de tuplas (subj, rel, obj)

        """
        
        query_base = """
            UNWIND $tripletas as tripleta
            MERGE (a:Entity {name: tripleta[0]})
            MERGE (b:Entity {name: tripleta[2]})
            WITH a, b, tripleta
            CALL apoc.create.relationship(a, tripleta[1], {}, b)
            YIELD rel
            RETURN rel;
        """
        summary = self.driver.execute_query(
            query_base,
            tripletas = tripletas,
            database_ = self.database
        )
        return summary


def obtener_grados_nodos(self, df):
    
    entidades = df['Entidad'].tolist()
    similares = df['similar'].tolist()
    set_entidades = list(set(entidades + similares)) 
    
    query = """
        UNWIND $entidades as entidad
        MATCH (n:Entity {name: entidad})
        RETURN n.name AS name, COUNT{(n)--()} AS n_relaciones
    """
    records, summary, key = self.driver.execute_query(query, entidades = set_entidades, database_ = self.database)
    grados = {rec['name']: rec['n_relaciones'] for rec in records}
    return grados



def fusionar_nodos(self, nodos_fusion):
    query_fusion = """
        UNWIND $grupos AS grupo
        
        MATCH (principal:Entity {name: grupo.nodo_principal})

        MATCH (secundarios:Entity)
        WHERE secundarios.name IN grupo.nodos_a_fusionar 

        WITH principal, collect(secundarios) AS nodos_secundarios, grupo.nodos_a_fusionar AS names_fusionados

        CALL apoc.refactor.mergeNodes([principal] + nodos_secundarios, {
        properties: {
            name: "discard",
            `\\*`: "combine"
        },
        mergeRels: true
        }) YIELD node

        SET node.fusionados = names_fusionados
        RETURN count(node) AS fusiones_realizadas
    """
    records, summary, key = self.driver.execute_query(query_fusion, grupos = nodos_fusion, database_ = self.database)
    return records[0]["fusiones_realizadas"]