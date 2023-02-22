.PHONY: exportenv

exportenv:
	conda env export >| nns-bs-local.yaml \
	&& sed '/prefix: /d' nns-bs-local.yaml > nns-bs.yaml