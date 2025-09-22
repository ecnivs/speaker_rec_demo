import logging
from google.genai import Client

class Gemini:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Client()
        self.model = "gemini-2.5-flash"

    def get_response(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=query
        )
        text = getattr(response, "text", None)
        return text
