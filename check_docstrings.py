import ast
import os
import sys

def check_path(path):
    missing = []
    if os.path.isfile(path):
        if not path.endswith(".py"):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Syntax error in {path}")
            return []
            
        if not ast.get_docstring(tree):
            missing.append(f"{path}: (Module) Missing docstring")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue
                
                if not ast.get_docstring(node):
                    kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    missing.append(f"{path}: ({kind}) {node.name} Missing docstring")
    return missing

def check_docstrings_recursive(directory):
    missing = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            missing.extend(check_path(os.path.join(root, file)))
    return missing

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_docstrings.py <path1> [path2 ...]")
        sys.exit(1)
    
    all_missing = []
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            all_missing.extend(check_path(arg))
        elif os.path.isdir(arg):
            all_missing.extend(check_docstrings_recursive(arg))
        else:
            print(f"Skipping invalid path: {arg}")

    if all_missing:
        print("Missing Docstrings Found:")
        for item in all_missing:
            print(item)
    else:
        print("No missing docstrings found!")
