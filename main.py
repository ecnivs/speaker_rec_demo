from dotenv import load_dotenv
import logging

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    force=True)

class Core:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

if __name__ == "__main__":
    load_dotenv()
