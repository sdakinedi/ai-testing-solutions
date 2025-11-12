import warnings
warnings.filterwarnings('ignore')

import json
import phoenix as px
import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from openinference.instrumentation import TracerProvider
from tqdm import tqdm
from phoenix.evals import (
    TOOL_CALLING_PROMPT_TEMPLATE, 
    llm_classify,
    OpenAIModel
)
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from openinference.instrumentation import suppress_tracing

import nest_asyncio
nest_asyncio.apply()

PROJECT_NAME = "evaluating-agent-19-03-2025"

# Add Phoenix API Key for tracing
PHOENIX_API_KEY = "692a40cd57ca8e0f86a:57e4727"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer_provider = register(
  project_name=PROJECT_NAME, # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed dependencies
)

OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)

tracer = tracer_provider.get_tracer(__name__)

from router import start_main_span, run_agent, tools
from tools import *


## extract the spans from the generate_visualization tool from the Phoenix client
query = SpanQuery().where(
    "name =='generate_visualization'"
).select(
    generated_code="output.value"
)

# The Phoenix Client can take this query and return the dataframe.
code_gen_df = px.Client().query_spans(query, 
                                      project_name=PROJECT_NAME, 
                                      timeout=None)

code_gen_df.head()

# Check if the code is runnable. this is the logic that will be used to evaluate the code
# if the code is runnable, then the score is 1, else the score is 0
import logging

# Configure logging
logging.basicConfig(filename="code_execution.log", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def code_is_runnable(output: str) -> bool:
    """Check if the code is runnable and log errors if it fails."""
    output = output.strip()
    output = output.replace("```python", "").replace("```", "")

    try:
        exec(output)
        return True
    except Exception as e:
        logging.error(f"Code execution failed: {e}\nCode:\n{output}")
        return False


code_gen_df["label"] = code_gen_df["generated_code"].apply(code_is_runnable).map({True: "runnable", False: "not_runnable"})
code_gen_df["score"] = code_gen_df["label"].map({"runnable": 1, "not_runnable": 0})


code_gen_df.head()

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Runnable Code Eval", dataframe=code_gen_df),
)