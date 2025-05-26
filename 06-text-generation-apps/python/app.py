from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# configure Azure OpenAI service client 
client = AzureOpenAI(
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"], 
  api_key=os.environ['AZURE_OPENAI_API_KEY'],  
  api_version = "2024-02-01"
  )

deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']


no_recipes = input("No of recipes (for example, 5): ")
ingredients = input("List of ingredients (for example, chicken, potatoes, and carrots): ")
filter = input("Filter (for example, vegetarian, vegan, or gluten-free): ")

prompt = f"Show me {no_recipes} recipes for a dish with the following ingredients: {ingredients}. Per recipe, list all the ingredients used, no {filter}"
messages = [{"role": "user", "content": prompt}]

# make completion
completion = client.chat.completions.create(model=deployment, messages=messages)

# print response
print(completion.choices[0].message.content)

old_prompt_result = completion.choices[0].message.content
prompt = "Produce a shopping list for the generated recipes and please don't include ingredients that I already have."

new_prompt = f"{old_prompt_result} {prompt}"
messages = [{"role": "user", "content": new_prompt}]
completion = client.chat.completions.create(model=deployment, messages=messages, max_tokens=500)

# print response
print("Shopping list:")
print(completion.choices[0].message.content)