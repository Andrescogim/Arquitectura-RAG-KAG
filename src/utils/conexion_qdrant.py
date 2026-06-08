from qdrant_client import QdrantClient

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex


class ConexionQdrant:
    def __init__(self):
        self.qd_client = QdrantClient(
            url="http://localhost:6333"
            )


    def query_qdrant(self, collection, embed_model, query, n_resultados):

        query_vector = embed_model.encode(query).tolist()
        results = self.qd_client.query_points(
            collection_name = collection,
            query = query_vector,
            limit = n_resultados,
        )
        return results.points



    def retriever_quadrant_llama(self, collection, embed_model, query):
        
        vector_store = QdrantVectorStore(
            client = self.qd_client,
            collection_name = collection,
            # text_key="texto_referencia",  # Para decirle como se ha guardado el texto en Qdrant
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store = vector_store,
            embed_model = embed_model,
        )

        retriever = index.as_retriever(similarity_top_k = 5)
        nodes = retriever.retrieve(query)

        return nodes



    def query_quadrant_llama(self, collection, embed_model, llm, query):
        
        vector_store = QdrantVectorStore(
            client = self.qd_client,
            collection_name = collection,
            # text_key="texto_referencia",  # Para decirle como se ha guardado el texto en Qdrant
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store = vector_store,
            embed_model = embed_model,
        )

        retriever = index.as_retriever()
        nodes = retriever.retrieve(query)

        query_engine = index.as_query_engine(llm = llm)
        response = query_engine.query(query)
        return nodes, response