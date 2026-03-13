from google import genai
from dotenv import load_dotenv
import os

# Substitua pela sua chave
load_dotenv()

api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

print("Modelos disponíveis para esta chave:")
for modelo in client.models.list():
    # Filtra para mostrar apenas os modelos que geram conteúdo (ignora modelos de embedding puro)
    if 'generateContent' in modelo.supported_actions:
        print(f"- {modelo.name}")