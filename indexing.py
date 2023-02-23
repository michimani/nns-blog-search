import traceback
import numpy as np
from util.client import init_openai, create_embeddings, TOKEN_LIMIT
from util.ctoken import str_to_tokens
from util.blog import load_blog_indexes
from util.nns import init_nns_index, save_nns_index


def create_nns_index(sentences):
    try:
        embeddings = create_embeddings(openai_client, sentences)

        embeddings = np.array([x["embedding"]
                               for x in embeddings], dtype=np.float32)

        nns_index.add(embeddings)
    except Exception:
        print(traceback.format_exc())

        return False

    return True


openai_client = None
nns_index = None

if __name__ == '__main__':
    blog_indexes = load_blog_indexes()
    print('There are {} indexes.'.format(len(blog_indexes)))

    openai_client = init_openai()
    nns_index = init_nns_index()

    sentences = []
    for bi in blog_indexes:
        sentence = bi['contents']
        _, count = str_to_tokens(sentence)
        if count > TOKEN_LIMIT:
            sentence = sentence[:2500]
        sentences.append(sentence)

    create_nns_index(sentences)

    save_nns_index(nns_index)
