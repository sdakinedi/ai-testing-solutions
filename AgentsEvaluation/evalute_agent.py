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

#add questions to ask the agent
agent_questions = [
    "What was the most popular product SKU?",
    "What was the total revenue across all stores?"
    # "Which store had the highest sales volume?",
    # "Create a bar chart showing total sales by store",
    # "What percentage of items were sold on promotion?",
    # "What was the average transaction value?"`1   `
]

for question in tqdm(agent_questions, desc="Processing questions"):
    try:
        ret = start_main_span([{"role": "user", "content": question}])
    except Exception as e:
        print(f"Error processing question: {question}")
        print(e)
        continue

print(TOOL_CALLING_PROMPT_TEMPLATE)


#########################
######### Querying the Required Spans from Phenoix
######## This is needed to label the spans as LLM or not and then use them to evaluate the agent
#########################

query = SpanQuery().where(
    # Filter for the `LLM` span kind.
    # The filter condition is a string of valid Python boolean expression.
    "span_kind == 'LLM'",
).select(
    question="input.value",
    tool_call="llm.tools"
)

# The Phoenix Client can take this query and return the dataframe.
tool_calls_df = px.Client().query_spans(query, 
                                        project_name=PROJECT_NAME, 
                                        timeout=None)
tool_calls_df = tool_calls_df.dropna(subset=["tool_call"])

tool_calls_df.head()


#########################
######### Evaluating Tool Calling capability of the router agent
######## This is needed to evaluate the agent
#########################
with suppress_tracing():
    tool_call_eval = llm_classify(
        dataframe = tool_calls_df,
        template = TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace("{tool_definitions}", 
                                                                 json.dumps(tools).replace("{", '"').replace("}", '"')),
        rails = ['correct', 'incorrect'],
        model=OpenAIModel(model="gpt-4o"),
        provide_explanation=True
    )

tool_call_eval['score'] = tool_call_eval.apply(lambda x: 1 if x['label']=='correct' else 0, axis=1)

tool_call_eval.head()

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Tool Calling Eval", dataframe=tool_call_eval),
)