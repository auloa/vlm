import os

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Authenticate with HuggingFace Hub
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise OSError("HF_TOKEN not found. Add it to your .env file.")


login(token=hf_token, add_to_git_credential=False)



