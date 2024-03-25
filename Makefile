.PHONY : all docs doctests lint test

all : lint docs doctests test

docs :
	rm -rf docs/_build
	sphinx-build -nW . docs/_build

doctest :
	rm -rf docs/_build
	sphinx-build -b doctest . docs/_build

lint :
	black --check .

test :
	pytest -v
