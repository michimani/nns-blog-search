import json
import os
import faiss
import openai
import numpy as np
from ctoken import str_to_tokens

# BLOG_INDEX_FILE = 'data/index.json'
BLOG_INDEX_FILE = 'data/index_len5.json'


def load_blog_indexes():
    with open(BLOG_INDEX_FILE) as f:
        indexes = json.load(f)
        return indexes


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


DIMENSION = 1536
nss_index = None


def init_nns_index():
    global nss_index
    nss_index = faiss.IndexFlatL2(DIMENSION)


NSS_INSEX_FILE = 'data/nss_index.faiss'


def save_nns_index():
    faiss.write_index(nss_index, NSS_INSEX_FILE)


EMBEDDING_MODEL = 'text-embedding-ada-002'
TOKEN_LIMIT = 8191
CONTENTS_LIMIET = 3000


def create_embeddings(blog_index):
    if len(blog_index['contents']) > CONTENTS_LIMIET:
        blog_index['contents'] = blog_index['contents'][:3000]

    bi_str = json.dumps(blog_index)

    _, count = str_to_tokens(bi_str)
    if count > TOKEN_LIMIT:
        print('over token limit. token_count:{} content_len:{} title_len:{} url_len:{}'.format(
            count, len(blog_index['contents']), len(blog_index['title']), len(blog_index['permalink'])))
        return

    embeddings = openai.Embedding.create(
        input=bi_str, model=EMBEDDING_MODEL)['data']

    embeddings = np.array([x["embedding"]
                           for x in embeddings], dtype=np.float32)

    nss_index.add(embeddings)


if __name__ == '__main__':
    blog_indexes = load_blog_indexes()
    print('There are {} indexes.'.format(len(blog_indexes)))

    init_openai()
    init_nns_index()

    i = 1
    for bi in blog_indexes:
        create_embeddings(bi)
        print('{}: created nns index'.format(i))
        i += 1

    save_nns_index()
