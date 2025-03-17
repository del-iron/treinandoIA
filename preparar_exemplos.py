from transformers import AutoTokenizer
from datasets import Dataset
import json

# Carregar o tokenizer do modelo BERTimbau
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/bert-base-cased-squad-v1.1-portuguese")

# Carregar o JSON
with open('toinhaRespondeMoodle.json', 'r') as f:
    dados = json.load(f)

# Transformar em dataset
dataset = Dataset.from_dict({
    'pergunta': [item['pergunta'] for item in dados],
    'contexto': [item['contexto'] for item in dados],
    'resposta': [item['resposta'] for item in dados]
})

def encontrar_posicoes_resposta(exemplo, tokenizer):
    # Encontrar a posição da resposta no contexto original
    resposta = exemplo['resposta']
    contexto = exemplo['contexto']
    start_char = contexto.find(resposta)  # Posição inicial da resposta
    end_char = start_char + len(resposta)  # Posição final da resposta

    # Tokenizar o contexto
    tokenized = tokenizer(
        exemplo['pergunta'],
        contexto,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True  # Mapeamento de caracteres para tokens
    )

    # Converter posições de caracteres para posições de tokens
    offset_mapping = tokenized['offset_mapping']
    start_token = None
    end_token = None

    for i, (start, end) in enumerate(offset_mapping):
        if start <= start_char < end:
            start_token = i
        if start < end_char <= end:
            end_token = i
        if start_token is not None and end_token is not None:
            break

    # Adicionar as posições ao exemplo tokenizado
    tokenized['start_positions'] = start_token
    tokenized['end_positions'] = end_token

    return tokenized

def preparar_exemplos(exemplo):
    return encontrar_posicoes_resposta(exemplo, tokenizer)

# Aplicar a função ao dataset
dataset_treinado = dataset.map(preparar_exemplos, batched=False)

# Salvar o dataset tokenizado (opcional)
dataset_treinado.save_to_disk("moodle_dataset")