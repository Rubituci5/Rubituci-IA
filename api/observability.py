from fastapi import FastAPI


def setup_observability(app: FastAPI) -> None:
    """Inicializa hooks de observabilidade da aplicação."""
    return None


def get_metrics():
    """Retorna métricas da aplicação."""
    return {}
