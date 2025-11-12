from smolagents import (
   CodeAgent,
   DuckDuckGoSearchTool,
   VisitWebpageTool,
   HfApiModel,
)

# // add your Hugging Face token here
token = "<your_hugging_face_token>"

hf_model = HfApiModel(token=token)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=hf_model,
    add_base_tools=True
)

feature = "Booking a ride on Uber"
story = "As a user, I want to book a ride on Uber so that I can get to my destination."


from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

tracer_provider = register(project_name="my-smolagents-app") # creates a tracer provider to capture OTEL traces
SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider) # automatically captures any smolagents calls as traces

agent.run("What is the today's conversion rate of USD to INR?") 