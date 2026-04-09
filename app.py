import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import os


st.set_page_config(
    page_title="arXiv Article Classifier",
    layout="wide"
)

st.title("arXiv Article Classifier")
st.markdown("Определить научную область статьи по заголовку и абстракту")

@st.cache_resource
def load_model():
    cache_dir = "./model_cache"
    with st.spinner("Скачивание модели (Это занимает несколько минут)"):
        snapshot_download(
            repo_id="adpokh/arxiv-model",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
    with st.spinner("Загрузка модели в память"):
        model = DistilBertForSequenceClassification.from_pretrained(cache_dir)
        tokenizer = DistilBertTokenizer.from_pretrained(cache_dir)
        label_encoder = joblib.load(os.path.join(cache_dir, "label_encoder.pkl"))
    
    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_model()

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Название статьи")
    
with col2:
    abstract = st.text_area("Абстракт (опционально)", height=200)


if st.button("Классифицировать", type="primary"):
    if not title.strip() and not abstract.strip():
        st.error("Введите хотя бы название статьи или абстракт")
    else:
        text = ""
        if title.strip():
            text = title
        if abstract.strip():
            text += " [SEP] " + abstract
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]

        indices = np.argsort(probabilities)[::-1]
        cumulative = 0
        top_indices = []
        
        for idx in indices:
            cumulative += probabilities[idx]
            top_indices.append(idx)
            if cumulative >= 0.95:
                break
        
        st.success("Результат классификации:")
        for idx in top_indices:
            label = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx] * 100
            st.markdown(f"**{label}**: {prob:.1f}%")
            st.progress(float(prob / 100))

st.markdown("---")
st.caption("Модель: DistilBERT-base-uncased | Классы: Physics, CS, Mathematics, Biology, Statistics, EE, Economics")
