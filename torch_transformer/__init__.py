import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

directory_path_from_main = './torch_transformer'
SPM_MODEL_FILE_PATH = f'{directory_path_from_main}/sentencepiece_model_en_ja.model'