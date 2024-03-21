from torch_transformer import DEVICE
import streamlit as st
import pandas as pd
import torch
from torch_transformer.transformer import Transformer
from torch_transformer.translator import Translator

# SentencePiece 英語・日本語 同一モデル 10epoch：
state_dict = torch.load(f'torch_transformer/Transformer_dim512_3e-05_10_state_dict.pt', map_location=DEVICE)
model = Transformer(16000, 16000, 512, 8)
model.load_state_dict(state_dict)
translator = Translator(model)


def translate(ja_text:str) -> str:
    translated_sentence = translator.beam_translate(ja_text)
    return translated_sentence

st.write('### \\\(^o^)/ 100本ノック100問目 翻訳サーバー \\\(^o^)/')
ja_text = st.text_input(label='日本語を入力')

if st.button(label='翻訳'):
    markdown = st.markdown("<p style='text-align: center;'>translating...</h1>", unsafe_allow_html=True)
    translated_text = translate(ja_text)
    markdown.empty()
    st.write(translated_text)
