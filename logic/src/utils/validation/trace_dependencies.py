"""
A tool to visually trace the origin and usages of a Python component,
outputting an interactive graph with clickable, natively nested UML-style info panels.
"""

import argparse
import ast
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

try:
    from pyvis.network import Network
except ImportError:
    print("Error: Required libraries not found.")
    print("Please run: pip install pyvis networkx")
    exit(1)


class ASTScopeVisitor(ast.NodeVisitor):
    """Custom AST visitor to capture definitions as a nested tree and group imports."""

    def __init__(self):
        self.tree = {"name": "<module>", "children": []}
        self.stack = [self.tree]

        self.imports_grouped = defaultdict(list)
        self.imports_direct = []
        self.imports_graph = {}

        self.flat_defs = set()
        self.seen_in_scope = set()

    def _add_node(self, name: str):
        # Prevent duplicates in the exact same scope
        scope_key = id(self.stack[-1])
        sig = (scope_key, name)
        if sig not in self.seen_in_scope:
            node = {"name": name, "children": []}
            self.stack[-1]["children"].append(node)
            self.seen_in_scope.add(sig)
            return node
        return None

    def visit_ClassDef(self, node):
        new_node = self._add_node(f"class {node.name}")
        self.flat_defs.add(node.name)
        if new_node:
            self.stack.append(new_node)
            self.generic_visit(node)
            self.stack.pop()

    def visit_FunctionDef(self, node):
        new_node = self._add_node(f"def {node.name}")
        self.flat_defs.add(node.name)
        if new_node:
            self.stack.append(new_node)
            self.generic_visit(node)
            self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        new_node = self._add_node(f"async def {node.name}")
        self.flat_defs.add(node.name)
        if new_node:
            self.stack.append(new_node)
            self.generic_visit(node)
            self.stack.pop()

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._add_node(target.id)
                self.flat_defs.add(target.id)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        level = node.level
        prefix = "." * level + module if level > 0 else module
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.imports_graph[local_name] = (prefix, alias.name)

            display_name = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            self.imports_grouped[prefix].append(display_name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.imports_graph[local_name] = (alias.name, alias.name)

            display_name = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            self.imports_direct.append(display_name)
        self.generic_visit(node)


class DependencyGrapher:
    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.all_files: Set[str] = set()

        # Graph tracing mapping
        self.definitions: Dict[str, Set[str]] = defaultdict(set)
        self.imports: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)

        # UI mapping
        self.ui_definitions: Dict[str, Dict] = {}
        self.ui_imports_direct: Dict[str, List[str]] = defaultdict(list)
        self.ui_imports_grouped: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

        self.nodes: Set[Tuple[str, str]] = set()
        self.edges: List[Tuple[str, str, str]] = []

    def _python_path_to_filepath(self, module_path: str) -> Optional[str]:
        target_suffix = module_path.replace(".", os.sep) + ".py"
        target_init_suffix = os.path.join(module_path.replace(".", os.sep), "__init__.py")

        for filepath in self.all_files:
            if filepath.endswith(target_suffix) or filepath.endswith(target_init_suffix):
                return filepath
        return None

    def scan_project(self):
        for root, _, files in os.walk(self.project_root):
            if any(part.startswith(".") or part in ("venv", "__pycache__", "env") for part in root.split(os.sep)):
                continue
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self.all_files.add(filepath)

        for filepath in self.all_files:
            self._parse_file(filepath)

    def _parse_file(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
        except (SyntaxError, UnicodeDecodeError):
            return

        visitor = ASTScopeVisitor()
        visitor.visit(tree)

        self.definitions[filepath] = visitor.flat_defs
        self.imports[filepath] = visitor.imports_graph
        self.ui_definitions[filepath] = visitor.tree
        self.ui_imports_direct[filepath] = visitor.imports_direct
        self.ui_imports_grouped[filepath] = dict(visitor.imports_grouped)

    def trace_backward(
        self, current_file: str, target_name: str, visited: Optional[Set[Tuple[str, str]]] = None
    ) -> Optional[str]:
        if visited is None:
            visited = set()

        trace_key = (current_file, target_name)
        if trace_key in visited:
            return None
        visited.add(trace_key)

        self.nodes.add((current_file, "intermediate"))

        if target_name in self.definitions.get(current_file, set()):
            self.nodes.discard((current_file, "intermediate"))
            self.nodes.add((current_file, "origin"))
            return current_file

        if target_name in self.imports.get(current_file, {}):
            source_module, original_name = self.imports[current_file][target_name]
            source_file = self._python_path_to_filepath(source_module)

            if source_file:
                self.edges.append((current_file, source_file, "backward"))
                return self.trace_backward(source_file, original_name, visited)
            else:
                ext_name = f"External: {source_module}"
                self.nodes.add((ext_name, "origin"))
                self.edges.append((current_file, ext_name, "backward"))
        return None

    def trace_forward(self, target_origin_file: str, target_original_name: str):
        for filepath in self.all_files:
            if filepath == target_origin_file:
                continue

            for local_alias in self.imports.get(filepath, {}):
                temp_visited = set()
                current_check = filepath
                current_name = local_alias
                path_edges = []
                found_origin = False

                while True:
                    trace_key = (current_check, current_name)
                    if trace_key in temp_visited:
                        break
                    temp_visited.add(trace_key)

                    if current_check == target_origin_file and current_name == target_original_name:
                        found_origin = True
                        break

                    if current_name in self.imports.get(current_check, {}):
                        src_mod, orig_nm = self.imports[current_check][current_name]
                        src_file = self._python_path_to_filepath(src_mod)
                        if src_file:
                            path_edges.append((src_file, current_check, "forward"))
                            current_check = src_file
                            current_name = orig_nm
                        else:
                            break
                    else:
                        break

                if found_origin:
                    self.nodes.add((filepath, "usage"))
                    self.edges.extend(path_edges)

    def _render_defs_html(self, children: List[Dict], limit=10, is_top=True) -> str:
        """Recursively formats the nested definitions into HTML lists with details tags."""
        if not children:
            return "<li><i>(None)</i></li>" if is_top else ""

        html = ""
        for i, child in enumerate(children):
            if i == limit:
                html += f"<details><summary>... (+{len(children) - limit} more)</summary><ul class='uml-list'>"

            html += f"<li><span class='bullet'>+</span> {child['name']}"
            if child["children"]:
                html += "<ul>" + self._render_defs_html(child["children"], limit=50, is_top=False) + "</ul>"
            html += "</li>"

        if len(children) > limit:
            html += "</ul></details>"
        return html

    def _render_imports_html(self, direct: List[str], grouped: Dict[str, List[str]], limit=10) -> str:
        """Formats the imports grouped by module with details tags."""
        items = []
        for imp in direct:
            items.append(f"<li><span class='bullet'>+</span> import {imp}</li>")

        for prefix, names in grouped.items():
            group_html = f"<li><span class='bullet'>+</span> from <b>{prefix}</b>:<ul>"
            for name in names:
                group_html += f"<li><span class='bullet'>-</span> import {name}</li>"
            group_html += "</ul></li>"
            items.append(group_html)

        if not items:
            return "<li><i>(None)</i></li>"

        html = ""
        for i, item in enumerate(items):
            if i == limit:
                html += f"<details><summary>... (+{len(items) - limit} more items)</summary><ul class='uml-list'>"
            html += item
        if len(items) > limit:
            html += "</ul></details>"
        return html

    def _build_node_uml(self, filepath: str) -> str:
        if filepath.startswith("External:"):
            name = filepath.replace("External: ", "")
            return f"<div class='uml-header'>&lt;&lt;external_library&gt;&gt;<br/><b>{name}</b></div><div class='uml-body'><div class='uml-section'>Third-party or built-in module.</div></div>"

        display_path = os.path.relpath(filepath, self.project_root)

        # Data
        defs_tree = self.ui_definitions.get(filepath, {"children": []})
        direct_imps = self.ui_imports_direct.get(filepath, [])
        grouped_imps = self.ui_imports_grouped.get(filepath, {})

        # HTML Gen
        html = f"<div class='uml-header'>&lt;&lt;module&gt;&gt;<br/><b>{os.path.basename(filepath)}</b></div>"
        html += "<div class='uml-body'>"
        html += f"<div class='uml-section'><b>Path:</b><br/>{display_path}</div>"

        html += "<div class='uml-section-title'>+ defines:</div><ul class='uml-list'>"
        html += self._render_defs_html(defs_tree["children"], limit=10)
        html += "</ul>"

        html += "<div class='uml-section-title'>+ imports:</div><ul class='uml-list'>"
        html += self._render_imports_html(direct_imps, grouped_imps, limit=8)
        html += "</ul></div>"

        return html

    def _build_edge_uml(self, source: str, target: str, direction: str) -> str:
        src_label = os.path.relpath(source, self.project_root) if not source.startswith("External:") else source
        tgt_label = os.path.relpath(target, self.project_root) if not target.startswith("External:") else target

        html = "<div class='uml-header'>&lt;&lt;import_dependency&gt;&gt;<br/><b>Connection</b></div>"
        html += "<div class='uml-body'>"
        if direction == "backward":
            html += f"<div class='uml-section'>+ <b>Importer:</b><br>{src_label}</div>"
            html += f"<div class='uml-section'>+ <b>Provider:</b><br>{tgt_label}</div>"
        else:
            html += f"<div class='uml-section'>+ <b>Provider:</b><br>{src_label}</div>"
            html += f"<div class='uml-section'>+ <b>Importer:</b><br>{tgt_label}</div>"
        html += "</div>"
        return html

    def generate_graph(self, target_file: str, target_name: str):
        print("Scanning project...")
        self.scan_project()
        self.nodes.add((target_file, "target"))

        print("Tracing backwards (Origin)...")
        origin_file = self.trace_backward(target_file, target_name)

        print("Tracing forwards (Usages)...")
        if origin_file:
            self.trace_forward(origin_file, target_name)
        else:
            self.trace_forward(target_file, target_name)

        self._render_pyvis(target_file)

    def _render_pyvis(self, target_node_id: str):
        net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", directed=True)
        net.force_atlas_2based(gravity=-50)

        for filepath, n_type in self.nodes:
            label = (
                filepath.replace("External: ", "")
                if filepath.startswith("External:")
                else os.path.basename(filepath).replace(".py", "")
            )
            uml_content = self._build_node_uml(filepath)

            color = (
                "#ff4b4b"
                if filepath == target_node_id
                else "#4caf50"
                if n_type == "origin"
                else "#2196f3"
                if n_type == "usage"
                else "#9e9e9e"
            )
            net.add_node(filepath, label=label, color=color, size=25, title="👆 Click for UML details", uml=uml_content)

        unique_edges = set(self.edges)
        for source, target, direction in unique_edges:
            uml_content = self._build_edge_uml(source, target, direction)
            color = "#e67e22" if direction == "backward" else "#9b59b6"
            net.add_edge(source, target, color=color, arrows="to", title="👆 Click for UML details", uml=uml_content)

        output_file = "dependency_graph.html"
        net.show(output_file, notebook=False)
        self._inject_uml_panel(output_file)
        print(f"\nSuccess! Interactive graph saved to: {output_file}")

    def _inject_uml_panel(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()

        injection = """
        <style>
        #uml-panel {
            display: none;
            position: absolute;
            top: 30px;
            right: 30px;
            width: 420px;
            max-height: 90vh;
            overflow-y: auto;
            background-color: #ffffff;
            border: 2px solid #000;
            font-family: 'Consolas', 'Courier New', monospace;
            color: #000;
            box-shadow: 6px 6px 15px rgba(0,0,0,0.6);
            z-index: 1000;
        }
        .uml-header {
            text-align: center;
            font-weight: bold;
            padding: 8px;
            border-bottom: 2px solid #000;
            background-color: #5ab3d9;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .uml-body { padding: 12px; background-color: #ffffff; }
        .uml-section { margin-bottom: 10px; border-bottom: 1px dashed #444; padding-bottom: 5px; word-wrap: break-word;}
        .uml-section-title { font-weight: bold; margin-top: 10px; }
        .uml-list { margin-top: 5px; font-size: 0.9em; list-style-type: none; margin-left: 0; padding-left: 0; white-space: nowrap;}
        .uml-list ul { list-style-type: none; padding-left: 18px; margin: 2px 0 6px 0; border-left: 1px dashed #ccc; }
        .bullet { font-weight: bold; color: #333; margin-right: 4px; }
        details { margin: 2px 0; }
        details > summary {
            cursor: pointer;
            color: #0066cc;
            font-weight: bold;
            list-style: none;
            margin-top: 4px;
        }
        details > summary::-webkit-details-marker { display: none; }
        details > summary:hover { text-decoration: underline; }
        #close-btn {
            position: absolute; right: 5px; top: 5px;
            cursor: pointer; font-weight: bold; color: #000; z-index: 20;
        }
        </style>
        <div id="uml-panel">
            <span id="close-btn" onclick="document.getElementById('uml-panel').style.display='none'">X</span>
            <div id="uml-content"></div>
        </div>

        <script type="text/javascript">
        setTimeout(function() {
            if (typeof network !== 'undefined') {
                network.on("click", function(params) {
                    var panel = document.getElementById("uml-panel");
                    var content = document.getElementById("uml-content");
                    if (params.nodes.length > 0) {
                        var nodeId = params.nodes[0];
                        var node = nodes.get(nodeId);
                        if (node && node.uml) {
                            content.innerHTML = node.uml;
                            panel.style.display = "block";
                        }
                    } else if (params.edges.length > 0) {
                        var edgeId = params.edges[0];
                        var edge = edges.get(edgeId);
                        if (edge && edge.uml) {
                            content.innerHTML = edge.uml;
                            panel.style.display = "block";
                        }
                    } else {
                        panel.style.display = "none";
                    }
                });
            }
        }, 1000);
        </script>
        """
        html_content = html_content.replace("</body>", injection + "</body>")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Generate an interactive dependency graph for a component.")
    parser.add_argument("project_root", help="Root directory to scan")
    parser.add_argument("target_file", help="The file where the component is located")
    parser.add_argument("target_name", help="The name of the component (function/class)")
    args = parser.parse_args()

    target_file = os.path.abspath(args.target_file)
    if not os.path.exists(target_file):
        print(f"Error: Could not find target file {target_file}")
        return

    grapher = DependencyGrapher(args.project_root)
    grapher.generate_graph(target_file, args.target_name)


if __name__ == "__main__":
    main()
