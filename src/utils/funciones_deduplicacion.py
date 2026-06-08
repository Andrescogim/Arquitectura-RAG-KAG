import pandas as pd
import bm25s
import textdistance
import regex as re


def calcular_rank_B25s(entidades, n_candidatos):
   entidades_token = bm25s.tokenize(entidades, stopwords="en")
   retriever = bm25s.BM25()
   retriever.index(entidades_token)
   results, scores = retriever.retrieve(entidades_token, k=n_candidatos)
   cols_sim = []
   cols_score = []
   for i in range(1, n_candidatos+1):
      cols_sim.append(f"similar_{i}")
      cols_score.append(f"score_{i}")
      
   df_resultados = pd.DataFrame(data=results, columns=cols_sim)
   df_scores = pd.DataFrame(scores, columns=cols_score)
   df_resultados = df_resultados.map(lambda x: entidades[x])
   df_rank_25 = pd.concat([df_resultados, df_scores], axis=1)
   df_rank_25['Entidad'] = entidades
   return df_rank_25


def encontrar_candidatos_iguales_principal(df, n_candidatos):
    
    for i in range(1, n_candidatos+1):
        df[f"entidad_similar_{i}_iguales"] = False
        df.loc[df["Entidad"] == df[f"similar_{i}"], f"entidad_similar_{i}_iguales"] = True
    return df


def get_columnas(n_candidatos):
    cols_jaro = []
    cols_leven = []
    cols_jaccard = []
    cols_overlap = []
    cols_exc_num = []
    cols_exc_nom = []
    for i in range(1, n_candidatos+1):
        cols_jaro.append(f"dist_jaro_{i}")
        cols_leven.append(f"dist_leven_{i}")
        cols_jaccard.append(f"dist_jaccard_{i}")
        cols_overlap.append(f"dist_overlap_{i}")
        cols_exc_num.append(f"excluir_num_{i}")
        cols_exc_nom.append(f"excluir_nom_{i}")
    cols_distancias = cols_jaro + cols_leven + cols_jaccard + cols_overlap
    
    return cols_distancias, cols_exc_num, cols_exc_nom


def calcular_distancias(fila, nlp, n_candidatos):
    doc_ent = nlp(fila['Entidad'])
    ent_sw_removed = " ".join([token.text for token in doc_ent if not token.is_stop and not token.is_punct])
    
    distancias_jaro = []
    distancias_leven = []
    distancias_jaccard = []
    distancias_overlap = []
    for i in range(1, n_candidatos+1):
        doc_sim = nlp(fila[f'similar_{i}'])
        sim_sw_removed = " ".join([token.text for token in doc_sim if not token.is_stop and not token.is_punct])
        dist_jaro = textdistance.jaro_winkler(ent_sw_removed, sim_sw_removed)
        dist_leven = textdistance.levenshtein.normalized_similarity(ent_sw_removed, sim_sw_removed)
        dist_jaccard = textdistance.jaccard.normalized_similarity(ent_sw_removed, sim_sw_removed)
        dist_overlap = textdistance.overlap.normalized_similarity(ent_sw_removed, sim_sw_removed)
        
        distancias_jaro.append(dist_jaro)
        distancias_leven.append(dist_leven)
        distancias_jaccard.append(dist_jaccard)
        distancias_overlap.append(dist_overlap)
    
    return distancias_jaro + distancias_leven + distancias_jaccard + distancias_overlap


def excluir_por_numeros(fila, n_candidatos):
    
    regex_romanos = r"\b(v[ii]{0,3}|i[vx]|x[lcvi]{0,4}|i{1,3})\b"
    romanos_ent = set(re.findall(regex_romanos, fila["Entidad"]))
    
    regex_numeros = r"\d+"
    numeros_ent = set(re.findall(regex_numeros, fila["Entidad"]))
    
    excluir = [False, False, False]
    
    for i in range(1, n_candidatos+1):
        romanos_sim = set(re.findall(regex_romanos, fila[f"similar_{i}"]))
        # if romanos_ent and romanos_sim and romanos_ent != romanos_sim:
        if romanos_ent != romanos_sim:
            excluir[i-1] = True
                
        numeros_sim = set(re.findall(regex_numeros, fila[f"similar_{i}"]))
        if numeros_ent != numeros_sim:
            excluir[i-1] = True
    
    return excluir


def excluir_por_nombre(fila, nlp, n_candidatos):
    
    doc_ent = nlp(fila['Entidad'])
    ents_ent = {ent.text: ent.label_ for ent in doc_ent.ents}
    palabras_ent = {ent.text for ent in doc_ent.ents}
    labels_excluir = ["PERSON", "GPE", "ORG"]
    
    exclusion = [False, False, False]
    
    for i in range(1, n_candidatos+1):
        if len(fila['Entidad'].split()) == len(fila[f'similar_{i}'].split()):

            doc_sim = nlp(fila[f'similar_{i}'])
            ents_sim = {ent.text: ent.label_ for ent in doc_sim.ents}
            palabras_sim = {ent.text for ent in doc_sim.ents}
            
            diferencias = palabras_ent.symmetric_difference(palabras_sim)

            for palabra_dif in diferencias:
                label_e = ents_ent.get(palabra_dif)
                label_sim = ents_sim.get(palabra_dif)
                if label_e in labels_excluir or label_sim in labels_excluir:
                    exclusion[i-1] = True

    return exclusion


def filtrar_fusionables(df, n_similares):
    
    for i in range(1, n_similares+1):
        col_jaro = f"dist_jaro_{i}"
        col_leven = f"dist_leven_{i}"
        col_jaccard = f"dist_jaccard_{i}"
        col_overlap = f"dist_overlap_{i}"
        col_score = f"score_{i}"
        
        col_ex_num = f"excluir_num_{i}"
        col_ex_nom = f"excluir_nom_{i}"
        col_sim_igual = f"entidad_similar_{i}_iguales"
        
        col_exclusion = f"exclusion_{i}"
        col_fusion = f"fusion_{i}"
        
        df[col_exclusion] = False
        df.loc[(df[col_ex_num] == True) | (df[col_ex_nom] == True) | (df[col_sim_igual] == True), col_exclusion] = True
        
        filtro_1 = (
            (
                (
                    (df[col_jaro].between(0.8, 0.999)) & (df[col_leven].between(0.75, 0.999))
                    ) |
                (
                    (df[col_jaro].between(0.9, 0.999)) & (df[col_leven].between(0.71, 0.999))
                    )
                ) &
            (df[col_score] > 4.5) &
            (df[col_exclusion] == False) 
            )
        
        filtro_2 = (
            (df[col_jaro].between(0.9, 0.999)) &
            (df[col_leven].between(0.8, 0.999)) &
            (df[col_score] > 2) &
            (df[col_exclusion] == False) 
            )
        
        filtro_3 = (
            (df[col_overlap] == 1) &
            (df[col_jaro] >= 0.9) &
            (df[col_score] >= 4.5) &
            (df[col_exclusion] == False) 
            )
        
        df[col_fusion] = False
        df.loc[filtro_1, col_fusion] = True
        df.loc[filtro_2, col_fusion] = True
        df.loc[filtro_3, col_fusion] = True
        
    return df

def unir_candidatos(df, n_candidatos):
    df_union = pd.DataFrame()
    for i in range(1, n_candidatos+1):
        cols_union = ['Entidad', f'similar_{i}',f'score_{i}',f'dist_jaro_{i}', f'dist_leven_{i}',f'dist_jaccard_{i}', f'dist_overlap_{i}',f'fusion_{i}']
        df_filt = df.loc[df[f'fusion_{i}'] == True, cols_union]
        rename_cols = {
            f'similar_{i}': "similar",
            f'score_{i}': "score",
            f'dist_jaro_{i}': "dist_jaro",
            f'dist_leven_{i}': "dist_leven",
            f'dist_jaccard_{i}': "dist_jaccard",
            f'dist_overlap_{i}': "dist_overlap",
            f'fusion_{i}': "fusion"
            }
        df_filt = df_filt.rename(columns = rename_cols)
        df_union = pd.concat([df_union, df_filt], ignore_index=True)
    df_union = df_union.drop_duplicates(["Entidad", "similar"]).sort_values("Entidad").reset_index(drop=True)
    return df_union


def grupos_fusion(df):
    entidades_1 = df['Entidad'].tolist()
    entidades_2 = df['similar'].tolist()
    grupos = []

    for ent_1, ent_2 in zip(entidades_1, entidades_2):
        encontrados = []
        for grupo in grupos:
            if ent_1 in grupo or ent_2 in grupo:
                encontrados.append(grupo)

        if not encontrados:
            grupos.append({ent_1, ent_2})
        else:
            nuevo_grupo = {ent_1, ent_2}
            for g in encontrados:
                nuevo_grupo.update(g)
                grupos.remove(g)
            grupos.append(nuevo_grupo)

    return [list(g) for g in grupos]


def seleccionar_nodo_principal(grupos, grados):
    nodos_fusion = {}
    for grupo in grupos:
        
        nodo_principal = max(grupo, key = len)
        for ent in grupo:
            if grados[ent] > grados[nodo_principal]:
                nodo_principal = ent
        nodos_fusion[nodo_principal] = [sec for sec in grupo if sec != nodo_principal]
        
    nodos_fusion_neo4j = [
        {
            "nodo_principal": grupo,
            "nodos_a_fusionar": nodos_fusion[grupo]
        }
        for grupo in nodos_fusion
    ]
    return nodos_fusion_neo4j