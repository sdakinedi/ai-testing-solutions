from phoenix.otel import register
import os
import openai

# Add Phoenix API Key for tracing
PHOENIX_API_KEY = "692a40cd57ca8e0f86a:57e4727"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# configure the Phoenix tracer
from phoenix.otel import register

openai.api_key  = os.environ['OPENAI_API_KEY']

tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed dependencies
)

import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku."}],
)
print(response.choices[0].message.content)