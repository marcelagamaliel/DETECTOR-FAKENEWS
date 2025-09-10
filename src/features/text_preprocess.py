def clean_text(text: str):
    """Garantir que o texto esteja padronizado (caso existam sujeiras no CSV)."""
    return str(text).lower().strip()
