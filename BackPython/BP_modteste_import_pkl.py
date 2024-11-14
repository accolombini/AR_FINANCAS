import pickle

# Carregar o arquivo .pkl
with open('BackPython/MODELS/short_term_rf_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Verificar as dimensões e alguns valores do conteúdo
print("Tipo de dado:", type(model_data))
print("Forma do array:", model_data.shape)
# Exibindo os primeiros 5 valores
print("Primeiros 5 valores:", model_data[:5])
