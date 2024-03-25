.PHONY : all docs doctests lint test _clean_docs

all : lint docs doctests test

_clean_docs :
	rm -rf docs/_build jupyter_execute

docs : _clean_docs
	sphinx-build -vnW . docs/_build

doctest : _clean_docs
	sphinx-build -b doctest . docs/_build

lint :
	black --check .

test :
	pytest -v
