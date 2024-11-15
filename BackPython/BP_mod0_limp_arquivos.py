''' 
    |||>Script para limpar diretórios
'''

# Importando bibliotecas necessárias para limpeza de diretórios e arquivos
import os
import shutil

# Diretórios e arquivos a serem limpos
dirs_to_clean = ["BackPython/DADOS", "BackPython/MODELS"]

for directory in dirs_to_clean:
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove o diretório e todo o conteúdo
        print(f"Diretório {directory} apagado.")
    else:
        print(f"Diretório {directory} não encontrado. Pulando.")
