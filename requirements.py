import os
import ast
from importlib_metadata import version, PackageNotFoundError

def get_imports(filepath):
    with open(filepath) as f:
        root = ast.parse(f.read(), filepath)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            yield node.module.split('.')[0]

def get_all_imports(directory):
    imports = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                imports.update(get_imports(filepath))
    return imports

project_directory = '.'  # replace with the path to your project directory
imports = get_all_imports(project_directory)

# Check if each package is installed and get its version
with open('requirements.txt', 'w') as f:
    for package in imports:
        try:
            __import__(package)
            try:
                package_version = version(package)
                f.write(f"{package}=={package_version}\n")
            except PackageNotFoundError:
                f.write(f"{package}\n")
        except ImportError:
            pass
