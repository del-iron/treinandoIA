from transformers import Trainer, TrainingArguments
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
import torch

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

# Função para preparar os exemplos
def preparar_exemplos(exemplo):
    # Tokenizar pergunta e contexto
    tokenized = tokenizer(
        exemplo['pergunta'],
        exemplo['contexto'],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True  # Garantir que retorna o mapeamento de deslocamento
    )

    # Encontrar posições de início e fim da resposta
    resposta = exemplo['resposta']
    start_char = exemplo['contexto'].find(resposta)
    end_char = start_char + len(resposta)

    # Tokenizar as posições de resposta
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

    # Adicionar as posições de início e fim ao tokenizado
    tokenized['start_positions'] = start_token
    tokenized['end_positions'] = end_token

    return tokenized

# Aplicar a função de preparação aos dados
dataset = dataset.map(preparar_exemplos, batched=False)

# Dividir em treino e avaliação
dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# Definir argumentos de treinamento
args = TrainingArguments(
    output_dir="./results",  # Diretório para salvar os resultados
    eval_strategy="epoch",  # Avaliar a cada época
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# Criar o modelo (exemplo de modelo de QA)
model = AutoModelForQuestionAnswering.from_pretrained("pierreguillou/bert-base-cased-squad-v1.1-portuguese")

# Definir o dispositivo manualmente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Criar o Trainer
trainer = Trainer(
    model=model,  # O modelo que você está treinando
    args=args,  # Os argumentos de treinamento
    train_dataset=train_dataset,  # O dataset de treinamento
    eval_dataset=eval_dataset,  # O dataset de avaliação
)

# Iniciar o treinamento
trainer.train()