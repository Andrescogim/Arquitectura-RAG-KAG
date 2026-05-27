

def extraer_top_k_entities(query_result, k):
    entities_found = []
    for entity in range(k):
        entities_found.append(query_result[entity]['name'])
    
    return entities_found



def formatear_tripletas(relaciones):
    tripletas_str = []

    for entidad in relaciones:
        for tripleta in relaciones[entidad]:
            origen = tripleta['origen']
            destino = tripleta['destino']
            relacion = tripleta['relacion']

            trip_str = f"{origen} --> {relacion} --> {destino}"
            
            tripletas_str.append(trip_str)
            
            tripletas_formateadas = "\n".join(tripletas_str)
            
    return tripletas_formateadas

