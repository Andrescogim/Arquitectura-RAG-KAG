import spacy

def extraer_entidades_ner_old(text, ner_model):
    
    # ner = spacy.load('en_core_web_sm')
    ner = spacy.load(ner_model)
    if "entity_ruler" in ner.pipe_names:
        ner.remove_pipe("entity_ruler")
    ner_custom = ner.add_pipe("entity_ruler", before="ner")

    patron_mayuscula = [
        {
            "IS_TITLE": True,  # Detecta entidades cuando empiezan por mayuscula
            "OP": "+"          # Se repite 1 o más veces
        }
    ]
    patterns = [
        {
            "label": "NOMBRE",          # El nombre de la entidad que quieres asignar
            "pattern": patron_mayuscula  # El patrón que definiste arriba
        }
    ]


    ner_custom.add_patterns(patterns)
    
    entidades_extraidas = ner(text)
    entidades_ner = [ent.text for ent in entidades_extraidas.ents]
    entidades_ner = entidades_ner[1:]
    
    return list(set(entidades_ner))


def extraer_entidades_ner(text, ner_model):
    """
    Extraccion de entidades con spacy.
    Se añade regla para que trate las palabras que comienzan en mayuscula como entidades.
    Primera letra a minuscula para no confundir con entidad.
    """

    text = text[0].lower() + text[1:]
    ner = spacy.load(ner_model)
    if "entity_ruler" in ner.pipe_names:
        ner.remove_pipe("entity_ruler")
    ner_custom = ner.add_pipe("entity_ruler", before="ner")

    patron_mayuscula = [
        {
            "IS_TITLE": True,  # Detecta entidades cuando empiezan por mayuscula
            "OP": "+"          # Se repite 1 o más veces
        }
    ]
    patterns = [
        {
            "label": "NOMBRE",          # El nombre de la entidad que se asigna
            "pattern": patron_mayuscula
        }
    ]


    ner_custom.add_patterns(patterns)
    
    entidades_extraidas = ner(text)
    entidades_ner = [ent.text for ent in entidades_extraidas.ents]

    
    return list(set(entidades_ner))