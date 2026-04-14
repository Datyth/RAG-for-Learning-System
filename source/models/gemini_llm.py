import os
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(os.getenv("GEMINI_API_KEY"))


from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiLLM:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None, temperature = 0.9, max_output_tokens = 1024):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model = self.load_model()

    def load_model(self):
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
        try:
            print(f'Loading Gemini model: {self.model_name}')
            model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            print('Gemini model loaded successfully')
            return model
        except Exception as e:
            print(f"Error loading Gemini model {self.model_name}: {e}")
            raise e
            
if __name__ == "__main__":
    # gemini_llm = GeminiLLM(api_key = gemini_api_key)
    print("ok")