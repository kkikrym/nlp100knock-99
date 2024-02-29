import streamlit as st
import pandas as pd


def translate(ja_text:str) -> str:
    return 'translated sentence: ' + ja_text

st.write('### \\\(^o^)/ 100本ノック100問目 翻訳サーバー \\\(^o^)/')
ja_text = st.text_input(label='日本語を入力')

if st.button(label='翻訳'):
    translated_text = translate(ja_text)
    st.write(translated_text)
