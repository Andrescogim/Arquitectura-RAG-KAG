import sys
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.append(str(root_dir))

from src.utils.funciones_carga_datos import load_filter_dataset_HuggingFace
from src.utils.funciones_construir_grafo import(
    extraccion_tripletas_2wiki_rebel,
    limpiar_tripletas,
    tripletas_from_evidences_2Wiki
)
from src.utils.funciones_guardado import guardar_resultados
from src.utils.funciones_generales import medir_recursos


@medir_recursos
def extraer_tripletas_2Wiki(rango_in_data, rango_fin_data, split, tokenizer, model, nlp, n_window_nac):
    subset = f"{split}[{rango_in_data}:{rango_fin_data + 1}]"
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_subset = None, split = subset)
    tripletas_rebel, triplestas_nacionalidad = extraccion_tripletas_2wiki_rebel(dataset_2Wiki, tokenizer, model, nlp, n_window_nac)
    tripletas_limpias = list(set(limpiar_tripletas(tripletas_rebel)))
    tripletas_limpias_nac = list(set(limpiar_tripletas(triplestas_nacionalidad)))
    return tripletas_limpias, tripletas_limpias_nac



def main():
    

    rango_in_data = 0
    rango_fin_data = 1
    split = "train"
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to("cuda")
    nlp = spacy.load("en_core_web_sm")
    n_window = 4 # Ventana busqueda tripletas nacionalidad

    """tripletas_limpias, tripletas_limpias_nac = extraer_tripletas_2Wiki(rango_in_data, rango_fin_data, split, tokenizer, model, nlp, n_window)

    ruta_tripletas = root_dir / "outputs" / "tripletas_generadas" / "dataset_2Wiki"
    nombre_result = f"tripletas_Rebel_2Wiki_registros_{rango_in_data}-{rango_fin_data}"
    nombre_result_nac = f"tripletas_Nacionalidad_2Wiki_registros_{rango_in_data}-{rango_fin_data}"
    # nombre_result = f"PRUEBA RECURSOS"
    # nombre_result_nac = f"PRUEBA RECURSOS"
    _ = guardar_resultados(tripletas_limpias, nombre_result, ruta_tripletas)
    _ = guardar_resultados(tripletas_limpias_nac, nombre_result_nac, ruta_tripletas)"""
    
    rango_in_data = 0
    rango_fin_data = 500
    subset = f"{split}[{rango_in_data}:{rango_fin_data + 1}]"
    dataset_2Wiki = load_filter_dataset_HuggingFace("xanhho/2wikimultihopqa", n_subset = None, split = subset)
    tripletas = tripletas_from_evidences_2Wiki(dataset_2Wiki)
    tripletas_limpias = list(set(limpiar_tripletas(tripletas)))
    nombre_result = f"tripletas_2wiki_gold_500"
    ruta_tripletas = root_dir / "outputs" / "tripletas_generadas" / "dataset_2Wiki"
    _ = guardar_resultados(tripletas_limpias, nombre_result, ruta_tripletas)
    
    # ruta_medicion = root_dir / "outputs" / "medicion_recursos"
    # nombre_result = f"PRUEBA RECURSOS"
    # _ = guardar_resultados(medir_recursos.acumulado, nombre_result, ruta_medicion)

if __name__ == "__main__":
    main()