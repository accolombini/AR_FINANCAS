# BP_mod3_main_pipeline.py
import os
import subprocess

# Diretório onde o módulo 3 está localizado
MODULE_DIR = "BackPython/"

# Executar o dashboard do módulo 3


def run_dashboard():
    dashboard_script = os.path.join(MODULE_DIR, "BP_mod3_dashboard.py")
    print("Iniciando o Dashboard do Módulo 3...")

    # Executar o script do dashboard
    subprocess.run(["python", dashboard_script], check=True)
    print("Dashboard encerrado.")


def main():
    print("Executando o Pipeline do Módulo 3...")

    # Executar o dashboard
    run_dashboard()


if __name__ == "__main__":
    main()
