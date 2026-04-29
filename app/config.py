from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Información de la App
    PROJECT_NAME: str = "Mi Proyecto FastAPI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    OPEN_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""
    APP_ENV: str = ""
    LOG_LEVEL: str = ""

    # Configuración de Pydantic para leer el archivo .env
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True
    )

# Instanciamos para importar 'settings' directamente en otros archivos
settings = Settings()
