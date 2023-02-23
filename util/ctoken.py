import tiktoken

enc = tiktoken.get_encoding('gpt2')


def str_to_tokens(s: str):
    t = enc.encode(s)
    return t, len(t)
