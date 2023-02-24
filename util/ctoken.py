import tiktoken

enc = tiktoken.get_encoding('cl100k_base')


def str_to_tokens(s: str):
    t = enc.encode(s)
    return t, len(t)
