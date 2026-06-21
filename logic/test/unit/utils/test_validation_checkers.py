import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 1. Test check_relative_imports
from logic.src.utils.validation.check_relative_imports import (
    format_relative_import,
    analyze_file as analyze_relative,
    print_stats_table as print_stats_relative,
    main as main_relative,
)

# 2. Test check_unused_imports
from logic.src.utils.validation.check_unused_imports import (
    analyze_file as analyze_unused,
    main as main_unused,
)

# 3. Test check_nested_imports
from logic.src.utils.validation.check_nested_imports import (
    is_type_checking_block,
    is_import_error_try_block,
    is_suppress_import_error_block,
    is_constant_expression,
    is_header_assignment,
    is_constant_guarded_if,
    is_header_setup_call,
    analyze_file as analyze_nested,
    print_stats_table as print_stats_nested,
    main as main_nested,
)

# 4. Test check_circular_imports
from logic.src.utils.validation.check_circular_imports import (
    file_to_module,
    collect_module_map,
    resolve_to_module,
    build_graph,
    tarjan_sccs,
    generate_html,
    main as main_circular,
)

# 5. Test count_loc
from logic.src.utils.validation.count_loc import (
    get_docstring_lines as get_doc_count,
    analyze_file as analyze_count,
    group_by_directory as group_count,
    main as main_count,
)

# 6. Test tree_loc
from logic.src.utils.validation.tree_loc import (
    analyze_file as analyze_tree,
    print_tree,
    main as main_tree,
)

# 7. Test trace_dependencies
import os
import ast
from logic.src.utils.validation.trace_dependencies import (
    ASTScopeVisitor,
    DependencyGrapher,
    main as main_trace,
)

# 8. Test visualize_module_graph
from logic.src.utils.validation.visualize_module_graph import (
    file_to_module,
    collect_module_map,
    resolve_to_module,
    build_graph,
    get_layer,
    find_violations,
    condense_to_packages,
    generate_html as generate_html_visualize,
    main as main_visualize,
)

# 9. Test check_interface_compliance
from logic.src.utils.validation.check_interface_compliance import (
    collect_python_files,
    base_name,
    has_decorator,
    parse_file,
    build_registry,
    get_required_abstract_methods,
    get_implemented_methods,
    check_compliance,
    main as main_compliance,
)



def test_relative_imports_format():
    import ast
    node = ast.parse("from . import a, b as c").body[0]
    assert format_relative_import(node) == "from . import a, b as c"


def test_relative_imports_analyze(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("from .a import b\nfrom ..c import d as e\nimport sys\n")
    results = analyze_relative(f)
    assert len(results) == 2
    assert results[0][0] == 1
    assert results[0][1] == 1
    assert results[1][0] == 2
    assert results[1][1] == 2

    # Syntax error file
    f_err = tmp_path / "err.py"
    f_err.write_text("invalid python code ...")
    assert analyze_relative(f_err) == []


def test_relative_imports_print_stats(capsys):
    results = {"package/file.py": [(1, 1, "from .a import b")]}
    print_stats_relative(results, Path("package"))


def test_relative_imports_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("from .a import b\n")

    with patch("sys.argv", ["check_relative_imports.py", str(tmp_path)]):
        main_relative()
    captured = capsys.readouterr()
    assert "Found 1 relative import" in captured.out

    # Test non-existent dir
    with patch("sys.argv", ["check_relative_imports.py", "nonexistent_dir"]), pytest.raises(SystemExit) as exc:
        main_relative()
    assert exc.value.code == 1


def test_unused_imports_analyze(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("import sys\nimport os\nprint(sys.argv)\n")
    results = analyze_unused(f)
    assert len(results) == 1
    assert results[0] == (2, "os")

    f_star = tmp_path / "star.py"
    f_star.write_text("from os import *\n")
    assert analyze_unused(f_star) == []

    f_err = tmp_path / "err.py"
    f_err.write_text("invalid python code ...")
    assert analyze_unused(f_err) == []

    f_factory = tmp_path / "factory.py"
    f_factory.write_text("class MyFactory:\n    import math\n")
    assert len(analyze_unused(f_factory, ignore_factories=False)) == 1
    assert len(analyze_unused(f_factory, ignore_factories=True)) == 0


def test_unused_imports_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("import os\n")

    with patch("sys.argv", ["check_unused_imports.py", str(tmp_path)]):
        main_unused()
    captured = capsys.readouterr()
    assert "Unused Imports Found" in captured.out

    f_clean = tmp_path / "clean.py"
    f_clean.write_text("import sys\nprint(sys.argv)\n")
    with patch("sys.argv", ["check_unused_imports.py", str(f_clean)]):
        main_unused()
    captured = capsys.readouterr()
    assert "No unused imports found" in captured.out

    with patch("sys.argv", ["check_unused_imports.py", "nonexistent.py"]), pytest.raises(SystemExit) as exc:
        main_unused()
    assert exc.value.code == 1


def test_nested_imports_helpers():
    import ast
    node_tc = ast.parse("if TYPE_CHECKING:\n    import sys").body[0]
    assert is_type_checking_block(node_tc)

    node_try = ast.parse("try:\n    import foo\nexcept ImportError:\n    pass").body[0]
    assert is_import_error_try_block(node_try)

    node_suppress = ast.parse("with contextlib.suppress(ImportError):\n    import bar").body[0]
    assert is_suppress_import_error_block(node_suppress)

    node_const = ast.parse("X = 42").body[0].value
    assert is_constant_expression(node_const)

    node_header = ast.parse("logger = get_pylogger(__name__)").body[0]
    assert is_header_assignment(node_header)

    node_if = ast.parse("if SOME_FLAG:\n    pass").body[0]
    assert is_constant_guarded_if(node_if)

    node_call = ast.parse("matplotlib.use('Agg')").body[0]
    assert is_header_setup_call(node_call)


def test_nested_imports_analyze(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text("import sys\ndef foo():\n    import math\n")
    results = analyze_nested(f)
    assert len(results) == 1
    assert results[0] == (3, "math")

    # Syntax error file
    f_err = tmp_path / "err.py"
    f_err.write_text("invalid python code ...")
    assert analyze_nested(f_err) == []


def test_nested_imports_print_stats():
    results = {"package/file.py": [(1, "math")]}
    print_stats_nested(results, Path("package"))


def test_nested_imports_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("def foo():\n    import math\n")

    with patch("sys.argv", ["check_nested_imports.py", str(tmp_path)]):
        main_nested()
    captured = capsys.readouterr()
    assert "Found 1 nested imports" in captured.out

    # Test non-existent dir
    with patch("sys.argv", ["check_nested_imports.py", "nonexistent_dir"]):
        main_nested()
    captured = capsys.readouterr()
    assert "does not exist" in captured.out


def test_circular_imports_helpers(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    f = root / "module" / "sub.py"
    f.parent.mkdir()
    f.write_text("")
    assert file_to_module(f, root) == "module.sub"

    known = {"module.sub", "module.other"}
    assert resolve_to_module("sub", 1, "module.other", known) == "module.sub"


def test_circular_imports_build_graph(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    a = root / "a.py"
    b = root / "b.py"
    a.write_text("import b\n")
    b.write_text("import a\n")

    graph = build_graph(root, set())
    assert "a" in graph
    assert "b" in graph
    assert "b" in graph["a"]
    assert "a" in graph["b"]


def test_circular_imports_tarjan():
    graph = {
        "a": {"b"},
        "b": {"c"},
        "c": {"a", "d"},
        "d": set(),
    }
    sccs = tarjan_sccs(graph)
    assert len(sccs) == 1
    assert set(sccs[0]) == {"a", "b", "c"}


def test_circular_imports_html(tmp_path):
    cycles = [["a", "b"]]
    graph = {"a": {"b"}, "b": {"a"}}
    output = tmp_path / "graph.html"

    with patch("logic.src.utils.validation.check_circular_imports.Network") as mock_net:
        generate_html(cycles, graph, output)
        assert mock_net.called


def test_circular_imports_main(tmp_path, capsys):
    root = tmp_path / "src"
    root.mkdir()
    a = root / "a.py"
    b = root / "b.py"
    a.write_text("import b\n")
    b.write_text("import a\n")

    with patch("sys.argv", ["check_circular_imports.py", str(root)]), pytest.raises(SystemExit) as exc:
        main_circular()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Found 1 circular import group" in captured.out

    # Clean run
    a.write_text("")
    b.write_text("")
    with patch("sys.argv", ["check_circular_imports.py", str(root)]), pytest.raises(SystemExit) as exc:
        main_circular()
    assert exc.value.code == 0

    # Non-existent dir
    with patch("sys.argv", ["check_circular_imports.py", "nonexistent_dir"]), pytest.raises(SystemExit) as exc:
        main_circular()
    assert exc.value.code == 1


def test_count_loc_helpers(tmp_path):
    source = '"""module docstring"""\ndef foo():\n    """func docstring"""\n    # a comment\n    return 42\n'
    doc_lines = get_doc_count(source)
    assert len(doc_lines) > 0

    f = tmp_path / "sample.py"
    f.write_text(source)
    metrics = analyze_count(str(f))
    assert metrics["code"] == 2
    assert metrics["comment"] == 1
    assert metrics["docstring"] == 2

    # Test grouping
    data = [
        {"path": "a/b/c.py", "code": 10, "comment": 2, "docstring": 3, "total": 15},
        {"path": "a/b/d.py", "code": 5, "comment": 1, "docstring": 1, "total": 7},
    ]
    grouped = group_count(data, 2)
    assert len(grouped) == 1
    assert grouped[0]["path"] == "a/b"
    assert grouped[0]["code"] == 15


def test_count_loc_main(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("import sys\n")

    with patch("sys.argv", ["count_loc.py", str(tmp_path)]):
        main_count()
    captured = capsys.readouterr()
    assert "Codebase Analysis" in captured.out


def test_tree_loc_helpers(tmp_path, capsys):
    f = tmp_path / "sample.py"
    f.write_text("import sys\n# comment\n")
    c, m, d = analyze_tree(str(f))
    assert c == 1
    assert m == 1
    assert d == 0

    # Print tree test
    print_tree(str(tmp_path))
    captured = capsys.readouterr()
    assert "sample.py" in captured.out


def test_tree_loc_main(tmp_path, capsys):
    with patch("sys.argv", ["tree_loc.py", str(tmp_path)]):
        main_tree()
    captured = capsys.readouterr()
    assert "Structure" in captured.out


def test_trace_dependencies_visitor():
    source = """
import sys
from os import path
class A:
    def method(self):
        pass
def func():
    x = 10
"""
    visitor = ASTScopeVisitor()
    tree = ast.parse(source)
    visitor.generic_visit(tree)

    assert "A" in visitor.flat_defs
    assert "func" in visitor.flat_defs
    assert "x" in visitor.flat_defs
    assert "sys" in visitor.imports_graph
    assert "path" in visitor.imports_graph


def test_dependency_grapher_tracing(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    a_py = src_dir / "a.py"
    b_py = src_dir / "b.py"

    a_py.write_text("from src.b import TargetClass\nclass UserClass:\n    pass\n")
    b_py.write_text("class TargetClass:\n    pass\n")

    grapher = DependencyGrapher(str(tmp_path))
    grapher.jinja_env = MagicMock()
    mock_template = MagicMock()
    mock_template.render.return_value = "Mocked HTML Content"
    grapher.jinja_env.get_template.return_value = mock_template

    with patch("logic.src.utils.validation.trace_dependencies.Network") as mock_network_cls:
        mock_net = MagicMock()
        mock_net.show.side_effect = lambda f, notebook: Path(f).write_text("<html><body></body></html>")
        mock_network_cls.return_value = mock_net

        grapher.scan_project()
        assert len(grapher.all_files) == 2

        try:
            grapher.generate_graph(str(a_py), "TargetClass")
            assert mock_network_cls.called
            assert Path("dependency_graph.html").exists()
        finally:
            if Path("dependency_graph.html").exists():
                os.remove("dependency_graph.html")


def test_trace_dependencies_main(tmp_path):
    a_py = tmp_path / "a.py"
    a_py.write_text("class Target:\n    pass\n")

    with patch("sys.argv", ["trace_dependencies.py", str(tmp_path), str(a_py), "Target"]), \
         patch("logic.src.utils.validation.trace_dependencies.DependencyGrapher.generate_graph") as mock_gen:
        main_trace()
        mock_gen.assert_called_once_with(os.path.abspath(str(a_py)), "Target")


def test_visualize_module_graph_helpers(tmp_path):
    root = tmp_path
    filepath = root / "logic" / "src" / "utils" / "helper.py"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("class Helper:\n    pass\n")

    assert file_to_module(filepath, root) == "logic.src.utils.helper"

    m_map = collect_module_map(root, set())
    assert "logic.src.utils.helper" in m_map

    known = {"a.b.c", "d.e"}
    assert resolve_to_module("c", 1, "a.b.d", known) == "a.b.c"
    assert resolve_to_module("d.e", 0, "a.b.d", known) == "d.e"

    layer, color = get_layer("logic.src.utils.helper", [("logic", "Logic", "#123")])
    assert layer == "Logic"
    assert color == "#123"

    layer_other, color_other = get_layer("unknown.module", [("logic", "Logic", "#123")])
    assert layer_other == "Other"


def test_visualize_module_graph_build_and_violations(tmp_path):
    root = tmp_path
    a = root / "logic" / "a.py"
    b = root / "gui" / "b.py"
    a.parent.mkdir(parents=True, exist_ok=True)
    b.parent.mkdir(parents=True, exist_ok=True)

    a.write_text("import gui.b\n")
    b.write_text("pass\n")

    graph = build_graph(root, set())
    assert "logic.a" in graph
    assert "gui.b" in graph

    pkg_graph, node_to_pkg = condense_to_packages(graph, 1)
    assert "logic" in pkg_graph
    assert pkg_graph["logic"] == {"gui"}

    layers = [("logic", "Logic", "#1"), ("gui", "GUI", "#2")]
    forbidden = [("Logic", "GUI")]
    violations = find_violations(graph, layers, forbidden)
    assert len(violations) == 1
    assert violations[0] == ("logic.a", "gui.b")


def test_visualize_module_graph_html(tmp_path):
    graph = {"logic.a": {"gui.b"}}
    layers = [("logic", "Logic", "#1"), ("gui", "GUI", "#2")]
    violation_edges = {("logic.a", "gui.b")}
    output = tmp_path / "module_graph.html"

    with patch("logic.src.utils.validation.visualize_module_graph.Network") as mock_net_cls:
        mock_net = MagicMock()
        mock_net_cls.return_value = mock_net
        generate_html_visualize(graph, layers, violation_edges, output, depth=0)
        assert mock_net_cls.called


def test_visualize_module_graph_main(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    a = root / "a.py"
    a.write_text("pass\n")

    with patch("sys.argv", ["visualize_module_graph.py", str(root), "--no-html"]), \
         pytest.raises(SystemExit) as exc:
        main_visualize()
    assert exc.value.code == 0


# --- Tests for check_interface_compliance ---

def test_compliance_base_name():
    import ast
    assert base_name(ast.Name(id="MyBase")) == "MyBase"
    assert base_name(ast.Attribute(attr="MyAttr")) == "MyAttr"
    assert base_name(ast.Constant(value=42)) == ""


def test_compliance_has_decorator():
    import ast
    func1 = ast.FunctionDef(
        name="foo",
        decorator_list=[ast.Name(id="abstractmethod")],
        body=[]
    )
    func2 = ast.FunctionDef(
        name="bar",
        decorator_list=[ast.Attribute(attr="abstractmethod")],
        body=[]
    )
    func3 = ast.FunctionDef(
        name="baz",
        decorator_list=[],
        body=[]
    )
    assert has_decorator(func1, "abstractmethod") is True
    assert has_decorator(func2, "abstractmethod") is True
    assert has_decorator(func3, "abstractmethod") is False


def test_compliance_collect_python_files(tmp_path):
    root = tmp_path
    (root / "dir1").mkdir()
    (root / ".git").mkdir()
    (root / "dir1" / "a.py").write_text("pass")
    (root / "dir1" / "b.txt").write_text("pass")
    (root / ".git" / "c.py").write_text("pass")

    files = collect_python_files(root, set())
    assert len(files) == 1
    assert files[0].name == "a.py"


def test_compliance_flow(tmp_path):
    interface_py = tmp_path / "interface.py"
    interface_py.write_text('''
from abc import ABC, abstractmethod

class MyInterface(ABC):
    @abstractmethod
    def required_one(self):
        pass

    @abstractmethod
    def required_two(self):
        pass
''')

    concrete_py = tmp_path / "concrete.py"
    concrete_py.write_text('''
from interface import MyInterface

class CompliantClass(MyInterface):
    def required_one(self):
        return 1

    def required_two(self):
        return 2

class NonCompliantClass(MyInterface):
    def required_one(self):
        return 1
''')

    files = [interface_py, concrete_py]
    registry = build_registry(files)

    assert "MyInterface" in registry
    assert "CompliantClass" in registry
    assert "NonCompliantClass" in registry

    # Test get_required_abstract_methods
    reqs = get_required_abstract_methods("CompliantClass", registry)
    assert reqs == {"required_one", "required_two"}

    # Test get_implemented_methods
    impl_compliant = get_implemented_methods("CompliantClass", registry)
    assert "required_one" in impl_compliant
    assert "required_two" in impl_compliant

    impl_non_compliant = get_implemented_methods("NonCompliantClass", registry)
    assert "required_one" in impl_non_compliant
    assert "required_two" not in impl_non_compliant

    # Check compliance
    violations = check_compliance(registry)
    assert len(violations) == 1
    assert violations[0]["class"] == "NonCompliantClass"
    assert violations[0]["missing"] == ["required_two"]


@patch("sys.exit")
def test_compliance_main_success(mock_exit, tmp_path):
    mock_exit.side_effect = SystemExit
    interface_py = tmp_path / "interface.py"
    interface_py.write_text('''
from abc import ABC, abstractmethod
class MyInterface(ABC):
    @abstractmethod
    def run(self): pass
class Concrete(MyInterface):
    def run(self): pass
''')

    with patch("sys.argv", ["check_interface_compliance.py", str(tmp_path)]), \
         pytest.raises(SystemExit):
        main_compliance()
    mock_exit.assert_called_once_with(0)


@patch("sys.exit")
def test_compliance_main_failure(mock_exit, tmp_path):
    mock_exit.side_effect = SystemExit
    interface_py = tmp_path / "interface.py"
    interface_py.write_text('''
from abc import ABC, abstractmethod
class MyInterface(ABC):
    @abstractmethod
    def run(self): pass
class Concrete(MyInterface):
    pass
''')

    with patch("sys.argv", ["check_interface_compliance.py", str(tmp_path)]), \
         pytest.raises(SystemExit):
        main_compliance()
    mock_exit.assert_called_once_with(1)


@patch("sys.exit")
def test_compliance_main_non_existent_dir(mock_exit):
    mock_exit.side_effect = SystemExit
    with patch("sys.argv", ["check_interface_compliance.py", "non_existent_directory_abc"]), \
         pytest.raises(SystemExit):
         main_compliance()
    mock_exit.assert_called_once_with(1)


# --- Tests for check_embedded_languages.py ---

def test_embedded_languages_detect():
    from logic.src.utils.validation.check_embedded_languages import detect_language
    assert detect_language("") == ""
    assert detect_language("short") == ""
    assert detect_language("<html><body>Hello</body></html>") == "HTML"
    assert detect_language("const x = 10; console.log(x);") == "JavaScript"
    assert detect_language("body { background-color: #fff; }") == "CSS"
    assert detect_language("SELECT * FROM users WHERE id = 1;") == "SQL"


def test_embedded_languages_get_docstring_lines():
    import ast
    from logic.src.utils.validation.check_embedded_languages import get_docstring_lines
    code = '''
def my_func():
    """This is a docstring
    spanning multiple lines.
    """
    x = 1
'''
    tree = ast.parse(code)
    doc_lines = get_docstring_lines(tree)
    # The docstring in my_func is on lines 3, 4, 5
    assert 3 in doc_lines
    assert 4 in doc_lines
    assert 5 in doc_lines
    assert 6 not in doc_lines


def test_embedded_languages_visitor():
    import ast
    from logic.src.utils.validation.check_embedded_languages import EmbeddedCodeVisitor
    code = 'html_str = "<div>hello</div>"'
    tree = ast.parse(code)
    visitor = EmbeddedCodeVisitor(doc_lines=set())
    visitor.visit(tree)
    assert len(visitor.findings) == 1
    assert visitor.findings[0][1] == "HTML"

    # Test JoinedStr (f-string)
    code_f = 'html_str = f"const x = {var}; console.log(x);"'
    tree_f = ast.parse(code_f)
    visitor_f = EmbeddedCodeVisitor(doc_lines=set())
    visitor_f.visit(tree_f)
    assert len(visitor_f.findings) > 0
    assert any(f[1] == "JavaScript" for f in visitor_f.findings)


def test_embedded_languages_analyze_file(tmp_path):
    from logic.src.utils.validation.check_embedded_languages import analyze_file
    py_file = tmp_path / "test_embed.py"
    py_file.write_text('''
def run():
    js = "console.log(123);"
    html = "<html><body></body></html>"
''')
    findings = analyze_file(py_file)
    assert len(findings) == 2
    langs = [f[1] for f in findings]
    assert "JavaScript" in langs
    assert "HTML" in langs

    # Test invalid syntax file
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class 123Unbound:")
    assert analyze_file(bad_file) == []


@patch("sys.exit")
def test_embedded_languages_main(mock_exit, tmp_path):
    from logic.src.utils.validation.check_embedded_languages import main as main_embedded
    # Create clean file
    py_file = tmp_path / "clean.py"
    py_file.write_text("def my_func():\n    return 1\n")

    with patch("sys.argv", ["check_embedded_languages.py", str(tmp_path)]):
        main_embedded()

    # Create unclean file
    py_file_unclean = tmp_path / "unclean.py"
    py_file_unclean.write_text('sql = "SELECT * FROM my_table;"')

    with patch("sys.argv", ["check_embedded_languages.py", str(tmp_path)]):
        main_embedded()

    # Non-existent dir
    with patch("sys.argv", ["check_embedded_languages.py", "non_existent_directory_abc"]):
        main_embedded()


# --- Tests for check_multi_classes.py ---

def test_multi_classes_get_top_level_classes(tmp_path):
    from logic.src.utils.validation.check_multi_classes import get_top_level_classes
    py_file = tmp_path / "classes.py"
    py_file.write_text('''
class A:
    pass

class _B:
    pass

def func():
    class Nested:
        pass
''')
    classes = get_top_level_classes(py_file)
    assert classes == ["A", "_B"]

    # Test syntax error
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class 123Unbound:")
    assert get_top_level_classes(bad_file) == []


@patch("sys.exit")
def test_multi_classes_main(mock_exit, tmp_path):
    from logic.src.utils.validation.check_multi_classes import main as main_multi_classes
    mock_exit.side_effect = SystemExit

    # 1. Clean run (at most 1 class)
    dir_clean = tmp_path / "clean"
    dir_clean.mkdir()
    py_file = dir_clean / "clean.py"
    py_file.write_text("class A:\n    pass\n")

    with patch("sys.argv", ["check_multi_classes.py", str(dir_clean)]), \
         pytest.raises(SystemExit):
        main_multi_classes()
    mock_exit.assert_called_once_with(0)

    # 2. Violation run (multiple classes)
    dir_viol = tmp_path / "violation"
    dir_viol.mkdir()
    py_file_2 = dir_viol / "unclean.py"
    py_file_2.write_text("class A:\n    pass\nclass B:\n    pass\n")

    # Clear previous mock calls
    mock_exit.reset_mock()
    with patch("sys.argv", ["check_multi_classes.py", str(dir_viol)]), \
         pytest.raises(SystemExit):
        main_multi_classes()
    mock_exit.assert_called_once_with(1)

    # 3. Violation run with --ignore-private
    dir_priv = tmp_path / "private"
    dir_priv.mkdir()
    py_file_3 = dir_priv / "unclean_private.py"
    py_file_3.write_text("class A:\n    pass\nclass _B:\n    pass\n")

    mock_exit.reset_mock()
    with patch("sys.argv", ["check_multi_classes.py", str(dir_priv), "--ignore-private"]), \
         pytest.raises(SystemExit):
        main_multi_classes()
    mock_exit.assert_called_once_with(0)


# --- Tests for check_type_coverage.py ---

def test_type_coverage_analyze_function():
    import ast
    from logic.src.utils.validation.check_type_coverage import analyze_function
    # Full annotation
    node1 = ast.parse("def f(x: int) -> str: pass").body[0]
    assert analyze_function(node1) == (1, 1, True)

    # Partial annotation
    node2 = ast.parse("def f(x: int, y) -> str: pass").body[0]
    assert analyze_function(node2) == (1, 2, True)

    # None annotation
    node3 = ast.parse("def f(x, y): pass").body[0]
    assert analyze_function(node3) == (0, 2, False)

    # Skip self / cls
    node4 = ast.parse("def f(self, x: int) -> str: pass").body[0]
    assert analyze_function(node4) == (1, 1, True)

    # Complex args (vararg/kwarg)
    node5 = ast.parse("def f(*args: int, **kwargs: str) -> None: pass").body[0]
    assert analyze_function(node5) == (2, 2, True)


def test_type_coverage_analyze_file(tmp_path):
    from logic.src.utils.validation.check_type_coverage import analyze_file
    py_file = tmp_path / "cov.py"
    py_file.write_text('''
def f1(x: int) -> str:
    pass

def f2(x):
    pass

def f3():
    pass
''')
    stats = analyze_file(py_file)
    assert stats["funcs"] == 3
    assert stats["full"] == 1
    assert stats["none"] == 2
    assert stats["partial"] == 0

    # Test syntax error
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("class 123Unbound:")
    assert analyze_file(bad_file) == {"funcs": 0, "full": 0, "partial": 0, "none": 0}


def test_type_coverage_markup():
    from logic.src.utils.validation.check_type_coverage import _coverage_markup
    assert _coverage_markup(0, 0) == "[dim]—[/dim]"
    assert "green" in _coverage_markup(9, 10)
    assert "yellow" in _coverage_markup(7, 10)
    assert "red" in _coverage_markup(3, 10)


def test_type_coverage_main(tmp_path):
    from logic.src.utils.validation.check_type_coverage import main as main_type_coverage
    # Create test files
    py_file1 = tmp_path / "test_cov1.py"
    py_file1.write_text('''
def f1(x: int) -> str: pass
def f2(x: int) -> str: pass
''')

    py_file2 = tmp_path / "test_cov2.py"
    py_file2.write_text('''
def f1(x): pass
def f2(x): pass
''')

    with patch("sys.argv", ["check_type_coverage.py", str(tmp_path), "--limit", "2", "--min-funcs", "2"]):
        main_type_coverage()
