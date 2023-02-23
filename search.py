import traceback
import sys
import numpy as np
from util.client import init_openai, create_embeddings, TOKEN_LIMIT
from util.ctoken import str_to_tokens
from util.blog import load_blog_indexes
from util.nns import load_nns_index


K = 3


def search(query, k):
    _, count = str_to_tokens(query)
    if count > TOKEN_LIMIT:
        print('over token limit. token_count:{} query_len:{}'.format(
            count, len(query)))
        return

    try:
        blog_indexes = load_blog_indexes()

        embeddings = create_embeddings(openai_client, query)

        query_embedding = np.array(
            [embeddings[0]["embedding"]], dtype=np.float32)

        _, idxs = nns_index.search(query_embedding, k)
        for idx in idxs[0]:
            blog_content = blog_indexes[idx]
            print('-----------\nTitle: {}\nURL: {}\n'.format(
                blog_content['title'], blog_content['permalink']))

    except Exception:
        print(traceback.format_exc())
        return


openai_client = None
nns_index = None

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print('first parameter is required for query')
        exit

    query = args[1]
    k = K
    if len(args) > 2 and args[2].isnumeric():
        k = int(args[2])

    nns_index = load_nns_index()
    openai_client = init_openai()

    search(query, k)
