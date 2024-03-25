# Global config.
project = "numpyro-extensions"
html_theme = "sphinx_book_theme"
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]
exclude_patterns = [
    ".pytest_cache",
    "**/.ipynb_checkpoints",
    "**/.jupyter_cache",
    "**/jupyter_execute",
    "playground",
    "README.md",
    "venv",
]

# Autodoc.
add_module_names = False

# Intersphinx.
intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
}

# Jupyter notebooks in Sphinx.
nb_execution_mode = "cache"
myst_dmath_double_inline = True
myst_enable_extensions = [
    "dollarmath",
]
