import openai
import os


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
