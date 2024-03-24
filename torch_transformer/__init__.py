import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPM_MODEL_FILE_PATH = f'./torch_transformer/sentencepiece_model_en_ja.model'