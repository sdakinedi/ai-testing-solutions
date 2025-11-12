#from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

import os
import openai

# Set your OpenAI API key
openai.api_key  = os.environ['OPENAI_API_KEY']

# Load documents from the data directory
documents = SimpleDirectoryReader("C:\\Users\\Satyaprasad_Dakinedi\\Desktop\\data").load_data()

# create the index
index = VectorStoreIndex.from_documents(documents)

# Setup the Query Engine
query_engine = index.as_query_engine()

# Send Your First Request
response = query_engine.query("What is the role of Scrum Master")
print(response)

#######################
### Setup Feedback Functions
#######################

from trulens_eval import Tru
import numpy as np
#from trulens.core import Feedback
#from trulens.core import Select
#from trulens.feedback.provider import OpenAI
from trulens.providers.openai import OpenAI

from trulens_eval import Feedback
from trulens_eval import Select

tru = Tru()

provider = OpenAI()

from trulens_eval.app import App
context = App.select_context(query_engine)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on(context.collect())  # Collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance)
    .on_input_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

# Construct the app
#Wrap the custom RAG with TruCustomApp, add list of feedbacks for eval

from trulens_eval import TruLlama

# Initialize the recorder
tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance])


# Use the recorder as a context manager to log queries
with tru_query_engine_recorder as recording:
    response = query_engine.query("What did the author do growing up?")
    print(response)


# The record of the app invocation can be retrieved from the `recording`:
rec = recording.get()  # Use .get if only one record

# Display the record
display(rec)

# Run the dashboard to visualize feedback
tru.run_dashboard()


# Retrieve the record from the recording context
rec = recording.get()  # Use .get if only one record

# Display the feedback record
display(rec)


# The results of the feedback functions can be rertireved from
# `Record.feedback_results` or using the `wait_for_feedback_result` method. The
# results if retrieved directly are `Future` instances (see
# `concurrent.futures`). You can use `as_completed` to wait until they have
# finished evaluating or use the utility method:

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

records.head()

#tru.get_leaderboard(app_ids=["LlamaIndex_App1"])


tru.run_dashboard()


