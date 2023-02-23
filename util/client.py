import openai
import os

EMBEDDING_MODEL = 'text-embedding-ada-002'
TOKEN_LIMIT = 8191


def init_openai():
    org_id = os.getenv("OPENAI_ORGANIZATION_ID")
    api_key = os.getenv("OPENAI_API_KEY")

    if org_id is None or len(org_id) == 0:
        print("OPENAI_ORGANIZATION_ID is empty")
        return

    if api_key is None or len(api_key) == 0:
        print("OPENAI_API_KEY is empty")
        return

    openai.organization = org_id
    openai.api_key = api_key

    return openai


def create_embeddings(openai_client, input):
    return openai_client.Embedding.create(
        input=input, model=EMBEDDING_MODEL)['data']
