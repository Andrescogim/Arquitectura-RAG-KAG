import spacy
import json

def limpiar_texto(texto):
    texto = texto.replace("(", " (").replace("  ", " ")
    return texto


def limpiar_entidad(entidad):
    entidad_clean = entidad.lower().replace(".", " ").replace("- ", "-").replace(" -", "-").replace("   ", " ").replace("  ", " ").strip("-").strip()
    return entidad_clean

def limpiar_relacion(relacion):
    relacion_clean = relacion.lower().strip().replace(" ", "_")
    return relacion_clean


def limpiar_tripletas(tripletas):
    tripletas_clean = []
    
    for trip in tripletas:
        s = limpiar_entidad(trip[0])
        o = limpiar_entidad(trip[2])
        r = limpiar_relacion(trip[1])
        tripletas_clean.append((s, r, o))
    return tripletas_clean


def extract_triplets_tuples(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                if subject != object_:
                    # triplets.append((subject.strip().lower(), relation.strip().replace(" ", "_"), object_.strip().lower()))
                    triplets.append((subject, relation, object_))
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                if subject != object_:
                    # triplets.append((subject.strip().lower(), relation.strip().replace(" ", "_"), object_.strip().lower()))
                    triplets.append((subject, relation, object_))
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        if subject != object_:
            # triplets.append((subject.strip().lower(), relation.strip().replace(" ", "_"), object_.strip().lower()))
            triplets.append((subject, relation, object_))
    return triplets



def extraer_tripletas_rebel(text, tokenizer, model):
    # text = limpiar_texto(text)
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 8,
        "num_return_sequences": 8,
    }
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt').to("cuda")
    generated_tokens = model.generate(
        # model_inputs["input_ids"].to(model.device),
        # attention_mask=model_inputs["attention_mask"].to(model.device),
        model_inputs["input_ids"].to("cuda"),
        attention_mask=model_inputs["attention_mask"].to("cuda"),
        **gen_kwargs,
    )
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    tripletas_tup_list = []
    for sentence in decoded_preds:
        tripletas_tup = extract_triplets_tuples(sentence)
        tripletas_tup_list += tripletas_tup
    return list(set(tripletas_tup_list))

# TRIPLETAS NACIONALIDAD

def extraer_nacionalidades(text, nlp, n_window):

    doc = nlp(text)
    # N_tokens = 3
    triplets_nacionalidad = []
    window_ctxt = []
    
    nationality = [ent.text for ent in doc.ents if ent.label_=="NORP"]
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            window = min(ent.end + n_window, len(doc))
            for token in doc[ent.end:window]:
                if token.text in nationality:
                    triplets_nacionalidad.append((ent.text.lower(), "originally_from", token.text.lower()))
                    # triplets_nacionalidad.append((ent.text, "country of origin", token.lemma_))
                    window_ctxt.append(f"{ent.text} {doc[ent.end:window]}")
                    break
    return triplets_nacionalidad, window_ctxt


def extraccion_tripletas_2wiki_rebel(dataset_2Wiki, tokenizer, model, nlp, n_window):
    tripletas_all = []
    all_triples_nacionalidad = []
    for idx, reg in enumerate(dataset_2Wiki, start = 1):
        print(f"Extrayendo tripletas. Registro {idx}/{len(dataset_2Wiki)}")
        parrafos = [" ".join(txt[1]) for txt in json.loads(reg['context']) ]  # Union por parrafos de los textos (quitando los titulos)
        for par in parrafos:
            par = limpiar_texto(par)
            tripletas = extraer_tripletas_rebel(par, tokenizer, model)
            tripletas_all += tripletas
            triplets_nacionalidad, window_ctxt = extraer_nacionalidades(par, nlp, n_window)
            all_triples_nacionalidad += triplets_nacionalidad
    return tripletas_all, all_triples_nacionalidad


def tripletas_from_evidences_2Wiki_to_neo4j(con_Neo, data_2wiki):
    for el in data_2wiki:
        for tripleta in eval(el["evidences"]):
            subj = tripleta[0]
            obj = tripleta[2]
            rel = tripleta[1].replace(" ", "_")
            # sumary = insert_triplet(driver, "2wiki.prueba1", subj, obj, rel)
            sumary = con_Neo.insertar_tripleta(subj, obj, rel)
    return sumary
