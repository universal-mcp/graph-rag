from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # database_url: str
    azure_openai_endpoint: str
    azure_openai_api_key: str

    embedding_model_name: str
    embedding_api_version: str

    # chat_model_name: str
    
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = 'ignore'

settings = Settings()