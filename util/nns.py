import traceback
import faiss

DIMENSION = 1536
NNS_INDEX_FILE = 'data/nns_index.faiss'


def init_nns_index():
    return faiss.IndexFlatL2(DIMENSION)


def load_nns_index():
    try:
        return faiss.read_index(NNS_INDEX_FILE)
    except Exception:
        print(traceback.format_exc())
        return None


def save_nns_index(index):
    faiss.write_index(index, NNS_INDEX_FILE)
