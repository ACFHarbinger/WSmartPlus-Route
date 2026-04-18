"""
Custom CSS stylesheet for the Streamlit dashboard.

Loads modular CSS files from the 'css' directory and wraps them
via a Jinja2 template to ensure a clean Python AST.
"""

import os

import jinja2


def _load_css() -> str:
    """Read modular CSS files and wrap them using a template."""
    # 1. Setup paths and Jinja2 environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_dir = os.path.join(base_dir, "css")
    template_dir = os.path.join(os.path.dirname(base_dir), "templates")

    loader = jinja2.FileSystemLoader(template_dir)
    env = jinja2.Environment(loader=loader)

    css_files = ["layout.css", "kpi.css", "status.css"]
    css_blocks = []

    # 2. Read the raw CSS content
    for filename in css_files:
        filepath = os.path.join(css_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                css_blocks.append(f.read())
        except FileNotFoundError:
            # Silent fallback if a file is missing
            continue

    # 3. Render the final string through the HTML wrapper
    # This removes all literal <style> tags from this Python file
    template = env.get_template("style_container.html")
    return template.render(css_blocks=css_blocks)


# Expose the loaded CSS for Streamlit consumption
CUSTOM_CSS = _load_css()
