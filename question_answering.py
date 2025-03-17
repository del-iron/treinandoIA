from transformers import AutoTokenizer, AutoModelForQuestionAnswering
modelo_nome = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
modelo = AutoModelForQuestionAnswering.from_pretrained(modelo_nome)