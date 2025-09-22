import logging
from google.genai import Client

class Gemini:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Client()
        self.model = "gemini-2.5-flash"

    def get_response(self, query: str) -> dict:
        prompt = f"""
    You are a voice assistant. Respond to the user query below.
    Provide output in this format (plain text first, then transcripted version):
    PLAIN: <plain text response>
    TRANSCRIPTED: <response with tone/emphasis suitable for speaking>
    Query: {query}
    """

        response_obj = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        text = getattr(response_obj, "text", "")
        plain, transcripted = None, None

        for line in text.splitlines():
            if line.startswith("PLAIN:"):
                plain = line[len("PLAIN:"):].strip()
            elif line.startswith("TRANSCRIPTED:"):
                transcripted = line[len("TRANSCRIPTED:"):].strip()

        return {
            "response": plain or text,
            "transcripted_response": transcripted or text
        }
