"""
text_preprocess.py
Funções simples/seguras de limpeza de texto (para uso posterior).
"""
def clean_text_minimal(text):
    """Normalização mínima: string, lower, strip."""
    return str(text).lower().strip()
