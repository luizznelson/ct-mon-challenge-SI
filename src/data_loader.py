import zipfile
import gdown
from glob import glob
import os

def download_data():
    """
    Baixa e extrai os dados do Google Drive, se ainda não existirem.
    """
    file_id = '1JcR3KRQDwJyzqTbCiYF9IHRLYd0_8ITC'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_zip = 'data/open-data.zip'
    extracted_path = 'data/Train'

    os.makedirs('data', exist_ok=True)

    # Verifica se os dados já foram extraídos
    if os.path.exists(extracted_path) and os.path.isdir(extracted_path):
        print("[INFO] Dados já extraídos. Pulando download e extração.")
        return

    # Verifica se o ZIP já foi baixado
    if not os.path.exists(output_zip):
        print("[INFO] Baixando os dados...")
        gdown.download(url, output_zip, quiet=False)
    else:
        print("[INFO] Arquivo ZIP já existe. Pulando download.")

    # Extração do ZIP
    print("[INFO] Extraindo os arquivos...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall('./data/')
    print("[INFO] Dados extraídos com sucesso.")

def get_training_files():
    """
    Retorna lista de arquivos de treinamento
    """
    return glob("./data/Train/dash/*")

def get_test_files():
    """
    Retorna lista de arquivos de teste
    """
    return glob("./data/Test/*")
