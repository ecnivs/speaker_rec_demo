import asyncio
from dotenv import load_dotenv
import logging
from audio import SpeechToText, TextToSpeech
import threading
import shutil
from contextlib import contextmanager
import tempfile
from response import Gemini

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    force=True)

# -------------------------------
# Temporary Workspace
# -------------------------------
@contextmanager
def new_workspace():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class Core:
    def __init__(self, workspace) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.lock: threading.Lock = threading.Lock()
        self.stt: SpeechToText = SpeechToText(core=self, model_size="small")
        self.tts: TextToSpeech = TextToSpeech(workspace=workspace)
        self.gemini: Gemini = Gemini()
        self.query = None

    async def run(self):
        threading.Thread(target=self.stt.listen, daemon=True).start()
        while True:
            with self.lock:
                if self.query:
                    response = self.gemini.get_response(self.query)
                    self.logger.info(response)
                    self.tts.speak(transcript=response["transcripted_response"])

            await asyncio.sleep(0.1)

if __name__ == "__main__":
    load_dotenv()

    with new_workspace() as workspace:
        core: Core = Core(workspace=workspace)
        asyncio.run(core.run())
