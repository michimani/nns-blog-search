import json

BLOG_INDEX_FILE = 'data/index.json'


def load_blog_indexes():
    with open(BLOG_INDEX_FILE) as f:
        indexes = json.load(f)
        return indexes
