import os
import openai
from langchain_openai.chat_models import AzureChatOpenAI
from autogen import ConversableAgent

from helper import get_openai_api_key
from openai import OpenAI
import pandas as pd
import json
import duckdb
from pydantic import BaseModel, Field
from IPython.display import Markdown

import phoenix as px
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from openinference.instrumentation import TracerProvider

# initialize the OpenAI client
openai_api_key = get_openai_api_key()
client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-4o-mini"

# Add Phoenix API Key for tracing
PHOENIX_API_KEY = "692a40cd57ca8e0f86a:57e4727"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# configure the Phoenix tracer
from phoenix.otel import register

OpenAI.api_key  = os.environ['OPENAI_API_KEY']

PROJECT_NAME = "evaluating-autogen-agent"
tracer_provider = register(
  project_name=PROJECT_NAME, # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed dependencies
)

OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)

tracer = tracer_provider.get_tracer(__name__)


import warnings
warnings.filterwarnings('ignore')

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_config = {"model": "gpt-4o-mini"}


agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

#prompt = "tell me a joke"

prompt = "decompose the user story: Book a ride from A to B on Uber App \
         into 3 Scenarios with 3 ACs in given , when , then format."

reply = agent.generate_reply(
    messages=[{"content": prompt, "role": "user"}]
)
print(reply)
