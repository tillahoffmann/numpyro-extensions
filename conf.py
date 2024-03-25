project = "numpyro-extensions"
html_theme = "sphinx_book_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
exclude_patterns = [
    "venv",
]
add_module_names = False
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
}
