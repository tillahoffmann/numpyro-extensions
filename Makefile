.PHONY : docs doctests lint test

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
