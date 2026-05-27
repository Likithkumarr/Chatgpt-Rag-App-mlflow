import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db7"
SQLITE_PATH = "Users.db"


# Forces the database to ALWAYS be created right next to your config file
AZURE = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "chat": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    "embed": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
}