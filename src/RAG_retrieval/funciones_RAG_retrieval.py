import json


def extraer_info_nodes(nodes):
    """
    De momnento solo lo basico.
    Luego se puede extraer mas cosas (como relationships)
    """
    
    nodes_clean = {}

    for node in nodes:
        node_id = node.node.id_
        nodes_clean[node_id] = {
            "text": node.node.text,
            "score": node.score
        }
    return nodes_clean



def extraer_info_points(points):
    """
    De momnento solo lo basico.
    Luego se puede extraer mas cosas (como relationships)
    """
    
    points_clean = {}
    
    for point in points:
        node_content = json.loads(point.payload['_node_content'])
        node_id = node_content["id_"]

        points_clean[node_id] = {
            "text": node_content["text"],
            "score": point.score
        }
    return points_clean



def filtrar_nodos_por_score(nodos):
    nodos_filt = {k: v for k,v in nodos.items() if v["score"]>0.6}
    return nodos_filt



def textos_para_prompt(resultados):
    textos = [v["text"] for k,v in resultados.items()]
    return "\n".join(textos)
