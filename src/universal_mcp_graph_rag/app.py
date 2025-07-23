import asyncio
import shutil
from pathlib import Path

import numpy as np
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag import LightRAG, QueryParam
from loguru import logger
from openai import AzureOpenAI
from tqdm import tqdm

from universal_mcp.applications import APIApplication
from universal_mcp.integrations import Integration
from universal_mcp_graph_rag.settings import settings

AZURE_OPENAI_ENDPOINT = settings.azure_openai_endpoint
AZURE_OPENAI_API_KEY = settings.azure_openai_api_key
EMBEDDING_MODEL_NAME = settings.embedding_model_name
AZURE_EMBEDDING_API_VERSION = settings.embedding_api_version

setup_logger("lightrag", level="INFO")

WORKING_DIR = Path("./rag_storage")

async def llm_model_func(
    prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs
) -> str:
    """Performs a chat completion using Azure OpenAI."""
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model="gpt-4o", 
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    """Generates embeddings for a list of texts using Azure OpenAI."""
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


class GraphRagApp(APIApplication):
    """
    An application for building and querying a knowledge graph-based RAG system.
    """

    def __init__(self, integration: Integration = None, **kwargs):
        """
        Initializes the GraphRagApp.
        """
        super().__init__(name="context", integration=integration, **kwargs)
        self.rag: LightRAG | None = None

    async def initialize(self):
        """
        Initializes the underlying LightRAG instance. This method sets up
        all necessary components, including storage and external API clients.
        """
        logger.info("Initializing LightRAG instance...")
        self.rag = LightRAG(
            working_dir=str(WORKING_DIR),
            llm_model_func=llm_model_func,
            graph_storage="Neo4JStorage",
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=embedding_func,
            ),
            max_parallel_insert=8,
        )
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        logger.info("LightRAG instance initialized successfully.")

    async def _ensure_initialized(self):
        """A helper to initialize the RAG instance on demand."""
        if not self.rag:
            await self.initialize()

    async def search(self, query: str):
        """
        Search your collection of documents for results semantically similar to the input query. 
        Collection description: contains research reports on financial markets received on email listed.equities@360.one
        Make sure to include inline citations and a numbered list of sources at the end of the response.
        
        Args:
            query (str): The question to ask the RAG system.
        """
        await self._ensure_initialized()
        logger.info(f"Querying RAG with: '{query}'")
        param = QueryParam(mode="mix", enable_rerank=True)
        result = await self.rag.aquery(query, param=param)
        
        return result

    def list_tools(self):
        """Lists the available tools for this application."""
        return [
            self.search, 
            ]

    async def finalize(self):
        """Gracefully finalizes storage connections."""
        if self.rag:
            await self.rag.finalize_storages()
            logger.info("RAG storage connections finalized.")

