import logging
from google.genai import Client

class Gemini:
    def __init__(self, persona):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Client()
        self.model = "gemini-2.5-flash"
        self.persona = persona

    def get_response(self, query: str) -> dict:
        prompt = self.persona.build_prompt(query)
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
