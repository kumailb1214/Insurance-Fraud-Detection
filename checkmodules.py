import os
import ast

project_dir = '.'  # Change to your project root if needed
used_modules = set()

# Walk through all files in project
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    node = ast.parse(f.read(), filename=filepath)
                    for n in ast.walk(node):
                        if isinstance(n, ast.Import):
                            for alias in n.names:
                                used_modules.add(alias.name.split('.')[0])
                        elif isinstance(n, ast.ImportFrom):
                            if n.module:
                                used_modules.add(n.module.split('.')[0])
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

# Print used modules
print("\n📦 Modules used in workspace:")
for mod in sorted(used_modules):
    print(f" - {mod}")
