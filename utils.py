import inspect
import re
from IPython.display import Markdown

def show_doc(obj, name=None, title_level=3):
    """
    Generates a Markdown description of a given class or function,
    displaying its custom title or name, signature, and docstring in a formatted manner.
    
    Parameters:
    - obj: The class or function to document.
    - name: Optional; A custom title for the documentation. If not provided, the object's name is used.
    
    Returns:
    - A Markdown representation of the class or function documentation.
    """
    # Determine the anchor name based on the provided title or the object's name
    anchor_name = name if name else obj.__name__
    # Replace spaces with dashes and make lowercase for URL compatibility
    anchor_id = anchor_name.replace(" ", "-").lower()
    
    # Use the custom title if provided, otherwise fall back to the object's name
    display_title = name if name else obj.__name__
    
    # Get the signature, excluding the name from the signature for separate display
    try:
        signature = inspect.signature(obj)
    except ValueError:  # Handle cases where signature retrieval is not applicable
        signature = ''
    
    # Get the docstring
    docstring = inspect.getdoc(obj)
    if not docstring:
        docstring = "No documentation available."
    
    # Wrap the docstring in <pre> tags to preserve formatting
    formatted_docstring = f"<pre>{docstring}</pre>"
    
    title_marker = "#" * title_level
    
    # Format the markdown output
    markdown_output = f"{title_marker} <a id=\"{anchor_id}\">{display_title}</a>\n\n> {obj.__name__} `{signature}`\n\n{formatted_docstring}\n\n"
    
    # Display the markdown
    return Markdown(markdown_output)

def get_version_from_init() -> str:
    """
    Reads the version number from the __init__.py file and returns it as string.
    """
    with open("gingado/__init__.py", "r") as f:
        version_file = f.read()

    # Use regular expression to extract the version string
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
