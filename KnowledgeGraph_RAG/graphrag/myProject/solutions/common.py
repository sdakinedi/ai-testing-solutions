

"""
Shared components: environment loading, driver creation,
LLM creation, embedder creation.
Does NOT contain domain constants or utilities anymore.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# Load environment variables immediately
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------
# Neo4j config helpers
# ---------------------------------------------------------------------

def get_neo4j_config():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    return uri, user, pwd


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def get_driver():
    uri, user, pwd = get_neo4j_config()
    return GraphDatabase.driver(uri, auth=(user, pwd))


# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

def get_llm(model_name="gpt-3.5-turbo"):
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in environment.")
    return OpenAILLM(
        model_name=model_name,
        api_key=api_key,
    )


# ---------------------------------------------------------------------
# Embeddings for retrievers
# ---------------------------------------------------------------------

def get_embedder(model="text-embedding-ada-002"):
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
    )
