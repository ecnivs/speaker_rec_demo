import logging
from google.genai import Client
from .prompt import Prompt

class Llm:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Client()
        self.model = "gemini-2.5-flash"
        self.prompt = Prompt()

    def get_response(self, speaker: str, query: str) -> dict:
        prompt = self.prompt.build(speaker=speaker, query=query)
        response_obj = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        text = getattr(response_obj, "text", "")
        lang, plain, transcripted = None, None, None

        for line in text.splitlines():
            if line.startswith("LANG:"):
                lang = line[len("LANG:"):].strip()
            elif line.startswith("PLAIN:"):
                plain = line[len("PLAIN:"):].strip()
            elif line.startswith("TRANSCRIPTED:"):
                transcripted = line[len("TRANSCRIPTED:"):].strip()

        return {
            "language": lang,
            "response": plain or text,
            "transcripted_response": transcripted or text
        }
