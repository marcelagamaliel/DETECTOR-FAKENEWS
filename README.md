# Detector de Fake News — PLN + Machine Learning (TF-IDF + Random Forest)
## Objetivo do Projeto

Este projeto implementa um sistema completo para detecção automática de Fake News em língua portuguesa, utilizando técnicas de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina.

A aplicação conta com:

- Pipeline de preparação dos dados (Fake.br Corpus)

- Treinamento e avaliação de modelos (TF-IDF + ML e Embeddings)

- API desenvolvida com FastAPI

- Interface web para consulta em tempo real

---
## Dataset Utilizado

**Fake.Br-Corpus**  
Link oficial: https://github.com/roneysco/Fake.br-Corpus.
O dataset contém cerca de **7.000 registros**

## Estrutura do Repositório
```text
detector-fakeNews/
│
├── data/
│   ├── Fake.br-Corpus-master/
│   └── processed/ (gerado após make_dataset.py)
│
├── models/  (gerado após treinamento)
│
├── notebooks/  (exploração, análises e comparações)
│
├── src/
│   ├── api/  (API + frontend)
│   ├── preprocessing/ (scripts de preparação)
│   └── models/ (treino/avaliação)
│
└── requirements.txt
```
---

## Instalação do Ambiente

### 1️. Clone o repositório
    git clone https://github.com/marcelagamaliel/DETECTOR-FAKENEWS.git
    cd detector-fakeNews

### 2. Crie e ative o ambiente virtual (Linux/Mac)
    python3 -m venv .venv
    source .venv/bin/activate
#### Windows
    py -m venv .venv
    .venv\Scripts\activate
### 3.Instale as dependências 
    pip install -r requirements.txt

## Preparação dos dados
### 1. Gerar os splits train/val/test
    python src/preprocessing/make_dataset.py

Isso criará:
- data/processed/train.csv    
- data/processed/val.csv
- data/processed/test.csv

## Treinar os Modelos
### 1. Baseline (TF-IDF + SVM/Random Forest)
    python src/models/train_baseline.py

### 2. Embeddings - Opcional (Apenas se quiser analisar notebooks)
    python src/models/train_embeddings.py

## Avaliar os Modelos
### TF-IDF
    python src/models/evaluate_baseline.py
### Embeddings - Opcional (Apenas se quiser analisar notebooks)
    python src/models/evaluate_embeddings.py

Os resultados serão salvos em: 
- models/results_baseline.csv
- models/results_embeddings.csv

## Executar a API
### Rodar a API FastAPI
    uvicorn src.api.app:app --reload
    
A API estará disponível em: http://127.0.0.1:8000 <p>
E a interface Web em: http://127.0.0.1:8000/static/index.html

## Usar a Interface Web
1. Abra o navegador
2. Entre em: http://127.0.0.1:8000/static/index.html
3. Cole uma notícia
4. Clique em **Analisar**
5. Veja o resultado da predição ("FAKE" ou "REAL") + probabilidade
    
##  Reproduzir do Zero (Linux/Mac)
Se quiser rodar tudo novamente:
```
rm -rf data/processed models
python src/preprocessing/make_dataset.py
python src/models/train_baseline.py
python src/models/evaluate_baseline.py
uvicorn src.api.app:app --reload
```
### Windows 
```
rmdir /s /q data\processed
rmdir /s /q models
python src/preprocessing/make_dataset.py
python src/models/train_baseline.py
python src/models/evaluate_baseline.py
uvicorn src.api.app:app --reload
```

## Avaliar métricas e comparações
#### Executar o Jupyter
    jupyter notebook
#### Abrir e executar
- notebooks/01_eda_fakebr.ipynb
- notebooks/02_data_checks.ipynb
- notebooks/03_baseline_models.ipynb
- notebooks/04_embedding_models.ipynb
- notebooks/05_comparativo_final.ipynb 
