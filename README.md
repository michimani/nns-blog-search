nns-blog-search
===

Example implementation of search in a blog using nearest neighbor search.

# Preparing

Create environment using conda.

```bash
conda env create -f=nns-bs.yaml
```

# Create index for NNS

```bash
python indexing.py
```

# Search contents

```bash
python search.py 'ポエム'
```

# License

[MIT](https://github.com/michimani/nns-blog-search/blob/main/LICENSE)

# Author

[michimani210](https://twitter.com/michimani210)