from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# configure Azure OpenAI service client 
client = AzureOpenAI(
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"], 
  api_key=os.environ['AZURE_OPENAI_API_KEY'],  
  api_version = "2023-10-01-preview"
  )

deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']

# add your completion code
question = input("Ask about a country or city: ")
prompt = f"""
You are going to be a travel planner.

Whenever questions regarding countries or locations are asked, you should provide the latest information. 
You should also suggest possible itineraries and popular restaurants.

If the information you have is very old (earlier than 2022), you should mention that the information is old and may not be accurate.

Provide answer for the question: {question}
"""
messages = [{"role": "user", "content": prompt}]  
# make completion
completion = client.chat.completions.create(model=deployment, messages=messages, temperature=0.5)

# print response
print(completion.choices[0].message.content)
