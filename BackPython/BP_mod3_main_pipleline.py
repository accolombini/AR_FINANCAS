# BP_mod3_main_pipeline.py
import os
import subprocess

# Diretório onde o módulo 3 está localizado
MODULE_DIR = "BackPython/"


def run_dashboard():
    """
    Executa o dashboard do Módulo 3 usando subprocess.

    Lança uma exceção se o script do dashboard falhar.
    """
    dashboard_script = os.path.join(MODULE_DIR, "BP_mod3_dashboard.py")
    print("Iniciando o Dashboard do Módulo 3...")

    try:
        # Executar o script do dashboard
        subprocess.run(["python", dashboard_script], check=True)
        print("Dashboard encerrado com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o Dashboard do Módulo 3: {e}")
        raise


def main():
    """
    Pipeline principal para o Módulo 3.

    Executa o dashboard e gerencia mensagens de status.
    """
    print("Executando o Pipeline do Módulo 3...")

    # Executar o dashboard
    try:
        run_dashboard()
    except Exception as e:
        print(f"Erro no Pipeline do Módulo 3: {e}")
    else:
        print("Pipeline do Módulo 3 executado com sucesso.")


if __name__ == "__main__":
    main()
