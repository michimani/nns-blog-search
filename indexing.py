import json
import time
import faiss
import traceback
import numpy as np
from client import init_openai
from ctoken import str_to_tokens

BLOG_INDEX_FILE = 'data/index.json'
# BLOG_INDEX_FILE = 'data/index_len5.json'


def load_blog_indexes():
    with open(BLOG_INDEX_FILE) as f:
        indexes = json.load(f)
        return indexes


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
        return False

    try:
        embeddings = openai_client.Embedding.create(
            input=bi_str, model=EMBEDDING_MODEL)['data']

        embeddings = np.array([x["embedding"]
                               for x in embeddings], dtype=np.float32)

        nss_index.add(embeddings)
    except Exception:
        print(traceback.format_exc())

        return False

    return True


openai_client = None

if __name__ == '__main__':
    blog_indexes = load_blog_indexes()
    print('There are {} indexes.'.format(len(blog_indexes)))

    openai_client = init_openai()
    init_nns_index()

    i = 1
    for bi in blog_indexes:
        if create_embeddings(bi):
            print('{}: created nns index'.format(i))
        else:
            print('{}: failed to create nns index'.format(i))
        i += 1

        time.sleep(2)

    save_nns_index()
