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
openai.api_key  = os.environ['OPENAI_API_KEY']


llm_config1 = {"model": "gpt-3.5-turbo"}
llm_config2 = {"model": "gpt-4o-mini"}


##### Enable conversation between 2 different agents
BA = ConversableAgent(
    name="Cathy",
    system_message=
    "Your name is Cathy(Business Analyst) with solid experience in understanding requirements , \
    , features, user stories and decomposing them into Acceptance criteria etc \
    You can start decomposing the user story given to you as input into Scenarios and ACs in Gherkin format... ",
    llm_config=llm_config2,
    human_input_mode="NEVER",
)

PO = ConversableAgent(
    name="Joe",
    system_message=
    "Your name is Joe and you are a product owner "
    "YOu can start identifying the requirements , features required for your Application context \
    and let BA work on their work of identifying the US, Scenarios, TCs etc",
    llm_config=llm_config1,
    human_input_mode="NEVER",
)

userstory = "Book a ride from A to B on Uber App"
context = "Uber is a ride booking service app..."

message = f"I am Joe. Cathy, let's work together on the feature {userstory} and {context}"

print(message)

# chat_result = PO.initiate_chat(
#     recipient=BA, 
#     message=message,
#     max_turns=2,
# )


from opentelemetry.trace import StatusCode

def initiate_chat_with_tracing(agent, recipient, message, max_turns):
    """
    Initiates a chat with OpenTelemetry tracing to capture spans.
    """
    with tracer.start_as_current_span("PO_initiate_chat", openinference_span_kind="agent") as span:
        # Add span attributes
        span.set_attribute("PO.name", agent.name)
        span.set_attribute("Recipient.name", recipient.name)
        span.set_attribute("Message.content", message)
        span.set_attribute("Max.turns", max_turns)
        span.set_input(value=message)        
        try:
            chat_result = agent.initiate_chat(
                recipient=recipient, 
                message=message,
                max_turns=max_turns,
            )
            # Mark span status as success
            span.set_output(value=chat_result)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            # Capture exceptions in the tracing spans
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

        
        
        return chat_result


chat_result = initiate_chat_with_tracing(
    agent=PO,
    recipient=BA, 
    message=message,
    max_turns=4,
)


