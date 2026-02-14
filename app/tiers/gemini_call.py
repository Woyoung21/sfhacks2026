from google.genai import Client
import os
from dotenv import load_dotenv
load_dotenv()

key=os.getenv("GOOGLE_API_KEY")
if not key:
	print("API Key not found. do you have GOOGLE_API_KEY in .env?")

client = Client(api_key=key)

def make_call(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        
        return response.text
    
    except Exception as e:
        return f"Error making API call: {str(e)}"
