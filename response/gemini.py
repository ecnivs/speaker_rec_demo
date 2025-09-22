import logging
from google.genai import Client

class Gemini:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Client()
        self.model = "gemini-2.5-flash"

    def get_response(self, query: str) -> dict:
        prompt = f"""
        You are a voice assistant. Detect the correct language for the response.
        Provide output in this format:
        LANG: <language code>
        PLAIN: <plain text response>
        TRANSCRIPTED: <response with tone/emphasis suitable for speaking>
        Query: {query}
        """

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
