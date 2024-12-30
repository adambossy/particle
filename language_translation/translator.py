import ast
import pprint
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Union

import click
from graphviz import Digraph

# Map language names to file extensions
LANGUAGE_EXTENSIONS = {"python": ".py", "swift": ".swift"}


def get_function_key(func_name: str, namespace: str | None, file_path: str) -> str:
    module_name = get_module_name(file_path)
    tokens = []
    if module_name != "UNKNOWN":
        tokens.append(module_name)
    if namespace:
        tokens.append(namespace)
    tokens.append(func_name)
    return ".".join(tokens)


def get_module_name(file_path: str) -> str:
    return Path(file_path).stem


@dataclass
class TranslatorNode:
    """Base class for storing information about code entities like functions and classes"""

    name: str

    # FIXME (adam) Deprecate node in favor of scope
    node: ast.AST  # AST node that was used to create this entity
    scope: Union["Scope", None] = None

    lineno: int = -1  # Line where function starts
    end_lineno: int = -1  # Line where function ends
    class_deps: Set[str] = field(default_factory=set)
    var_deps: Set[str] = field(default_factory=set)
    source_code: str = ""  # Add source code field

    def module_name(self) -> str:
        """Return the module name for the node."""
        return get_module_name(self.file)

    def __str__(self) -> int:
        """Return the hash of the node, using its unique key."""
        return self.key()


@dataclass
class FunctionNode(TranslatorNode):
    """Store information about a function/method"""

    file: str = None  # Path to the file containing this entity
    namespace: str | None = None  # Full namespace path
    calls: Set["FunctionNode"] = field(
        default_factory=set
    )  # Set of fully qualified function names this function calls
    called_by: Set[str] = field(
        default_factory=set
    )  # Set of fully qualified function names that call this function

    def key(self) -> str:
        """Return a unique key for the node, combining namespace and name."""
        return get_function_key(self.name, self.namespace, self.file)

    def __hash__(self) -> int:
        """Return the hash of the node, using its unique key."""
        return hash(self.key())

    def is_test(self) -> bool:
        """Determine if a function is a test function based on its name."""
        return self.name.startswith("test_")


@dataclass
class CallNode(TranslatorNode):
    """Store information about a function call"""

    calling_node: TranslatorNode | None = None

    # attrs: Dict[str, FunctionNode] = field(default_factory=dict)
    def key(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class VarNode(TranslatorNode):
    type: str | None = None

    def key(self) -> str:
        return f"{self.scope.name}.{self.name}"


@dataclass
class ClassNode(TranslatorNode):
    """Store information about a class"""

    file: str = None  # Path to the file containing this entity

    # No additional fields needed for now, but can be extended in the future

    def key(self) -> str:
        """Return a unique key for the node, combining namespace and name."""
        return f"{self.module_name()}.{self.name}"

    def is_test(self) -> bool:
        """Determine if a class is a test class based on its name."""
        return self.name.startswith("Test")


class Scope:
    def __init__(self, name: str):
        self.name = name
        self.classes: dict[str, ClassNode] = {}
        self.functions: dict[str, FunctionNode] = {}
        self.vars: dict[str, VarNode] = {}
        self.parent: Scope | None = None
        self.children: list[Scope] = []
        self.imports: dict[str, str] = {}

    def add_child(self, child: "Scope"):
        self.children.append(child)
        child.parent = self

    def add_class(self, class_node: ClassNode):
        self.classes[class_node.name] = class_node

    def get_class(self, name: str) -> ClassNode | None:
        return self.classes.get(name)

    def add_var(self, var_node: VarNode):
        self.vars[var_node.name] = var_node

    def get_var(self, name: str) -> VarNode | None:
        return self.vars.get(name)

    def add_function(self, function_node: FunctionNode):
        self.functions[function_node.name] = function_node

    def get_function(self, name: str) -> FunctionNode | None:
        return self.functions.get(name)

    def get_local_symbols(self) -> list[str]:
        return (
            list(self.classes.keys())
            + list(self.functions.keys())
            + list(self.vars.keys())
            + (list(self.imports.keys()) if self.is_module() else [])
        )

    def is_module(self) -> bool:
        return self.parent is None

    def get_enclosing_symbols(self) -> set[TranslatorNode]:
        symbols = set()
        scope = self.parent
        while scope and not scope.is_module():
            symbols.update(scope.get_local_symbols())
            scope = scope.parent
        return symbols

    def get_global_symbols(self) -> set[TranslatorNode]:
        symbols = set()
        scope = self.parent
        if not scope:
            return set()
        while not scope.is_module():
            scope = scope.parent
        assert scope.is_module()
        if scope:
            symbols.update(scope.get_local_symbols())
        return symbols

    def get_local_node(self, symbol_name: str) -> TranslatorNode | None:
        if symbol_name in self.vars:
            return self.vars[symbol_name]
        if symbol_name in self.functions:
            return self.functions[symbol_name]
        if symbol_name in self.classes:
            return self.classes[symbol_name]
        return None

    def get_node(self, symbol_name: str) -> TranslatorNode | None:
        node = self.get_local_node(symbol_name)
        if node:
            return node
        return self.parent and self.parent.get_node(symbol_name)

    def __str__(self) -> str:
        return f"Scope: {self.name}"

    def __repr__(self) -> str:
        return f"Scope: {self.name}"


class CallGraphAnalyzer(ast.NodeVisitor):
    def __init__(
        self, language: str, project_path: str = None, files: list[str] = None
    ):
        self.language = language.lower()
        self.project_path = Path(project_path) if project_path else None
        self.files = [Path(f) for f in files] if files else None

        if self.language != "python":
            raise ValueError(
                "Unsupported language: Only Python is supported with ast.NodeVisitor"
            )

        # Initialize attributes for call graph
        self.functions: Dict[str, FunctionNode] = {}
        self.current_namespace: List[str] = []
        self.current_class = None
        self.current_function = None
        self.current_scope = None
        self.tree: ast.AST = None  # Store the current AST
        self.code = None
        self.current_file = None
        self.classes: Dict[str, ClassNode] = {}
        self.calls: Dict[str, CallNode] = {}
        self.vars: Dict[str, VarNode] = {}
        self.scopes: list[Scope] = []

    def analyze(self):
        """Analyze either the project directory or specific files."""
        if self.project_path:
            self.analyze_project()
        elif self.files:
            self.analyze_files()
        else:
            raise ValueError("Either project_path or files must be provided")

    def analyze_files(self):
        self._analyze_files(self.files)

    def _analyze_files(self, files: list[str]):
        """Analyze specific files and build call graph."""
        self.classes.clear()
        self.functions.clear()
        self.calls.clear()

        for file_path in files:
            print(f"\nAnalyzing file: {file_path}")
            tree = self.parse_file(str(file_path))
            if tree:
                self.visit(tree)

        self._resolve_imports()
        self._resolve_calls()

    def analyze_project(self):
        """Analyze all files in the project directory."""
        # Get the file extension for the current language
        extension = LANGUAGE_EXTENSIONS.get(self.language, f".{self.language}")
        self.files = [f for f in self.project_path.rglob(f"*{extension}")]
        return self._analyze_files(self.files)

    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file based on its name."""
        return file_path.name.startswith("test_") or file_path.name.endswith("_test.py")

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.current_scope.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self.current_scope.imports[alias.asname or alias.name] = (
                f"{module}.{alias.name}"
            )
        self.generic_visit(node)

    def _maybe_track_namespace(self, node: ast.FunctionDef | ast.ClassDef):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            symbol_name = node.name  # Directly access the name attribute

            # TODO (adam) May want to pack namespace handling in to the scope object
            self.current_namespace.append(symbol_name)

            scope = Scope(name=".".join(self.current_namespace))
            if self.current_scope:
                self.current_scope.add_child(scope)
            self.current_scope = scope

            # TODO (adam) May want to pack class and function tracking into the scope object
            if isinstance(node, ast.FunctionDef):
                self.current_function = symbol_name

            if isinstance(node, ast.ClassDef):
                self.current_class = symbol_name

    def _maybe_untrack_namespace(self, node: ast.FunctionDef | ast.ClassDef):
        # Legacy - pop namespace when leaving class or function
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):

            # TODO (adam) May want to pack namespace handling in to the scope object
            self.current_namespace.pop()

            self.current_scope = self.current_scope.parent

            # TODO (adam) May want to pack class and function tracking into the scope object
            if isinstance(node, ast.FunctionDef):
                self.current_function = None

            if isinstance(node, ast.ClassDef):
                self.current_class = None

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._create_var_node(
            node,
            [self._resolve_generic(node.target)],
            self._resolve_generic(node.annotation),
        )
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module):
        scope = Scope(name=get_module_name(self.current_file))
        self.scopes.append(scope)
        self.current_scope = scope
        self.generic_visit(node)

    def _flatten_nested_targets(self, targets: list) -> list[str]:
        """
        This is for handling syntax like:

        a, b, c = 1, 2, 3
        a, (b, c) = 1, (2, 3)
        """
        flattened = []
        for target in targets:
            if isinstance(target, (list, tuple, set)):
                flattened.extend(self._flatten_nested_targets(target))
            else:
                flattened.append(target)
        return flattened

    def visit_Assign(self, node: ast.Assign):
        target_names = [self._resolve_generic(t) for t in node.targets]
        flattened_target_names = self._flatten_nested_targets(target_names)
        self._create_var_node(node, flattened_target_names)
        self.generic_visit(node)

    def _create_var_node(
        self,
        node: ast.Assign | ast.AnnAssign,
        target_names: list[str],
        annotation: str | None = None,
    ) -> str:
        scope = ".".join(self.current_namespace)

        # TODO (adam) Should probably get rid of self.vars
        if scope not in self.vars:
            self.vars[scope] = []

        for target_name in target_names:
            var_node = VarNode(
                name=target_name,
                node=node,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                scope=self.current_scope,
                type=annotation,
            )
            self.current_scope.add_var(var_node)
            # TODO (adam) Should probably get rid of self.vars
            self.vars[scope].append(var_node)

    def _resolve_generic(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            return self._resolve_call(node)
        if isinstance(node, ast.Attribute):
            return self._resolve_attribute(node)
        elif isinstance(node, ast.Name):
            return self._resolve_name(node)
        elif isinstance(node, ast.BoolOp):
            # node.func could be ast.Or or ast.And
            raise NotImplementedError(f"Unsupported node type: {type(node)}")
        elif isinstance(node, ast.Subscript):
            return f"{self._resolve_generic(node.value)}[{self._resolve_generic(node.slice)}]"
        elif isinstance(node, ast.Constant):
            # node.kind = 'int' or 'str' or 'float' or None
            # Returns the constant value - not sure we need it
            return node.value
        elif isinstance(node, ast.List | ast.Tuple | ast.Set):
            return [self._resolve_generic(e) for e in node.elts]
        elif isinstance(node, ast.Dict):
            return [self._resolve_generic(e) for e in node.elts]  # Needs testing
        elif isinstance(node, ast.JoinedStr):
            return "".join(
                [self._resolve_generic(e) for e in node.values]
            )  # Needs testing
        elif isinstance(node, ast.FormattedValue):
            return self._resolve_generic(node.value)  # Needs testing

        import pdb

        pdb.set_trace()

    def _resolve_attribute(self, node: ast.Attribute) -> str:
        name_chain = []
        # current = node.func
        current = node

        # Walk “backwards” while we have Attribute nodes.
        while isinstance(current, ast.Attribute):
            name_chain.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            name_chain.append(self._resolve_name(current))

        # name_chain is backwards, e.g. ["entries", "cmudict", "corpus", "nltk"]
        name_chain.reverse()

        return ".".join(name_chain)

    def _resolve_name(self, node: ast.Name) -> str:
        return node.id

    def _resolve_assign(self, node: ast.Assign):
        pass

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process a function definition node."""
        self._maybe_track_namespace(node)

        func_name = node.name
        # FIXME (adam) This is a hack to get the namespace to work for now
        namespace = ".".join(self.current_namespace[:-1])
        function_node = self._create_function_node(
            node, func_name, namespace=namespace, file=self.current_file
        )
        function_node.lineno = node.lineno
        function_node.end_lineno = node.end_lineno

        self.generic_visit(node)
        self._maybe_untrack_namespace(node)

    def _try_resolve_call_with_function_info(
        self, func_name: str, namespace: str | None
    ) -> FunctionNode | None:
        partial_key = (namespace or "") + "." + func_name
        for function_key in self.functions.keys():
            if function_key.endswith(partial_key):
                return self.functions[function_key]
        return None

    def get_all_args(self, node: ast.FunctionDef) -> list[VarNode]:
        # node.args example:
        # {
        #     'posonlyargs': [],
        #     'args': [<ast.arg object at 0x1034f7760>],
        #     'vararg': None,
        #     'kwonlyargs': [],
        #     'kw_defaults': [],
        #     'kwarg': None,
        #     'defaults': []
        # }
        #
        # node.args.args[0] example:
        # {
        #     'arg': 'self',
        #     'annotation': None,
        #     'type_comment': None,
        #     'lineno': 8,
        #     'col_offset': 26,
        #     'end_lineno': 8,
        #     'end_col_offset': 30
        # }

        scope = ".".join(self.current_namespace)
        vars = []

        for arg in node.args.args:
            vars.append(
                VarNode(
                    name=arg.arg,
                    node=arg,
                    lineno=arg.lineno,
                    end_lineno=arg.end_lineno,
                    scope=self.current_scope,
                    type=arg.annotation,
                )
            )

        if node.args.kwarg:
            vars.append(
                VarNode(
                    name=node.args.kwarg.arg,
                    node=node.args.kwarg,
                    lineno=node.args.kwarg.lineno,
                    end_lineno=node.args.kwarg.end_lineno,
                    scope=self.current_scope,
                    type=None,
                )
            )

        if node.args.vararg:
            vars.append(
                VarNode(
                    name=node.args.vararg.arg,
                    node=node.args.vararg,
                    lineno=node.args.vararg.lineno,
                    end_lineno=node.args.vararg.end_lineno,
                    scope=scope,
                    type=None,
                )
            )

        return vars

    def _create_function_node(
        self,
        node: ast.FunctionDef,
        func_name: str,
        namespace: str = None,
        file: str = "UNKNOWN",
    ) -> FunctionNode:
        source_code = self._get_source_code(node, self.code)
        function_key = get_function_key(func_name, namespace, file)

        function_node = FunctionNode(
            name=func_name,
            namespace=namespace,
            calls=set(),
            called_by=set(),
            lineno=node.lineno,
            end_lineno=node.end_lineno,
            node=node,
            file=file,
            source_code=source_code,  # Store the source code
            scope=self.current_scope,
        )

        parent_scope = self.current_scope.parent
        parent_scope.add_function(function_node)

        self.current_scope.add_function(function_node)

        arg_vars = self.get_all_args(node)
        for var in arg_vars:
            self.current_scope.add_var(var)

        # TODO (adam) Should probably get rid of self.functions
        self.functions[function_key] = function_node

        return function_node

    def visit_ClassDef(self, node: ast.ClassDef):
        # Process class definitions
        self._maybe_track_namespace(node)

        class_name = node.name
        class_info = ClassNode(
            name=class_name,
            node=node,
            file=self.current_file,
            lineno=node.lineno,
            end_lineno=node.end_lineno,
            scope=self.current_scope,
        )
        # TODO (adam) Should probably get rid of self.classes
        self.classes[class_info.key()] = class_info

        parent_scope = self.current_scope.parent
        parent_scope.add_class(class_info)

        self.current_scope.add_class(class_info)

        # TODO (adam) May want to pack class tracking into the scope object
        self.current_class = class_name

        self.generic_visit(node)
        self._maybe_untrack_namespace(node)

    def visit_Call(self, node: ast.Call):
        current_module = get_module_name(self.current_file)
        caller_key = ".".join([current_module] + self.current_namespace)
        callee_node = self._resolve_call(node)
        if callee_node:
            if caller_key not in self.calls:
                self.calls[caller_key] = []
            # NOTE (adam) This adds dupes, which is fine for now
            self.calls[caller_key].append(callee_node)

        self.generic_visit(node)

    def _resolve_call(self, node: ast.Call) -> CallNode:
        method_name = self._resolve_generic(node.func)
        # if isinstance(method_name, CallNode):
        #     import pdb

        #     pdb.set_trace()

        if self.current_function:
            calling_node = self.current_function
        elif self.current_class:
            calling_node = self.current_class
        else:
            # Assume none means we're at the module level
            calling_node = None

        return CallNode(
            node=node,
            name=method_name,
            scope=self.current_scope,
            calling_node=calling_node,
        )

    def _find_node(self, attribute: str, scope: Scope) -> TranslatorNode | None:
        if attribute in scope.get_local_symbols():
            return scope.get_node(attribute)
        elif attribute in scope.get_enclosing_symbols():
            return scope.get_node(attribute)
        elif attribute in scope.get_global_symbols():
            return scope.get_node(attribute)
        return None

    def _resolve_imports(self):
        print("Resolving imports")

        for scope in self.scopes:
            print(f"\nScope: {scope.name}")

            for import_name, full_import_name in scope.imports.items():
                print(f"Import: {import_name} -> {full_import_name}")

                other_scopes = [s for s in self.scopes if s != scope]
                for other_scope in other_scopes:

                    # We resolve the import_name instead of the full_import_name here,
                    # which is a less precise match and could run into ambiguity issues
                    # but we'll deal with those later
                    matched_node = self._resolve_attribute_chain(
                        import_name, other_scope
                    )
                    if matched_node:
                        if isinstance(matched_node, FunctionNode):
                            scope.add_function(matched_node)
                        elif isinstance(matched_node, ClassNode):
                            scope.add_class(matched_node)
                        elif isinstance(matched_node, VarNode):
                            scope.add_var(matched_node)

    def _resolve_attribute_chain(
        self, attribute_chain: str, scope: Scope
    ) -> TranslatorNode:
        attributes = attribute_chain.split(".")
        for attribute in attributes:

            # May want to treat "self" as a special case
            # May want to treat "None" as a special case
            # Discern between "terminal" attributes and attributes in a chain

            matched_node = self._find_node(attribute, scope)
            if matched_node:
                if isinstance(matched_node, VarNode) and matched_node.type:
                    type_node = self._find_node(matched_node.type, matched_node.scope)
                    if type_node:
                        scope = type_node.scope
        return matched_node

    def _resolve_calls(self):
        function_nodes_by_name = defaultdict(list)
        for f in self.functions.values():
            function_nodes_by_name[f.name].append(f)

        class_nodes_by_name = defaultdict(list)
        for c in self.classes.values():
            class_nodes_by_name[c.name].append(c)

        print("\nCalls:")
        for caller, calls in self.calls.items():
            for call in calls:
                # FIXME (adam) I'm not sure yet what to do with nested calls,
                # so we'll just skip them for now
                if isinstance(call.name, CallNode):
                    continue
                func_name = call.name.split(".")[-1]
                matches = []

                matching_functions = function_nodes_by_name[func_name]
                matches.extend(matching_functions)

                matching_classes = class_nodes_by_name[func_name]
                matches.extend(matching_classes)

                if len(matches) > 0:

                    # LEGB - Local, Enclosing, Global, Built-in
                    scope = call.scope

                    print("\n--------------------------------")
                    print(f"{len(matches)} matches for {func_name}")

                    print(f"\nCaller: {caller}")
                    print(f"\nCall: {call.name}")
                    print(f"\nMatches:")
                    for match in matches:
                        print(f"  Name: {match.name}")
                        print(f"  File: {match.file}")

                    node = self._resolve_attribute_chain(call.name, scope)

                    # Add deps with real, resolved nodes, including var_deps and class_deps eventually
                    print(f"NODE---------------> {node}")
                    print(f"CALLING NODE-------> {call.calling_node}")

                    if not call.calling_node:
                        print(f"Function {call.name} called at the module level")

                    if node and isinstance(node, FunctionNode) and call.calling_node:
                        calling_node = self._resolve_attribute_chain(
                            call.calling_node, scope
                        )
                        if isinstance(calling_node, FunctionNode):
                            calling_node.calls.add(node)
                            node.called_by.add(calling_node)
                        elif isinstance(calling_node, ClassNode):
                            calling_node.calls.add(node)
                            node.called_by.add(calling_node)

                elif not matches:
                    print(f"Unmatched call {call.name}")

    def print_scope_tree(self, scope: Scope, indent: int = 0):
        print(f"\n{indent * '  '}Scope: {scope.name}")
        print(f"{indent * '  '}  Symbols: {scope.get_local_symbols()}")
        print(f"{indent * '  '}  Is Module? {scope.is_module()}")
        for child in scope.children:
            self.print_scope_tree(child, indent + 1)

    def parse_file(self, file_path: str) -> ast.AST:
        """Parse a single file and return its AST."""
        self.current_file = file_path  # Set current file
        with open(file_path, "rb") as f:
            self.code = f.read().decode("utf-8")
        try:
            self.tree: ast.AST = ast.parse(self.code)  # Store the tree
        except SyntaxError as e:
            # Catching this for an issue with 'dropdown_html' in paloma-story-generation/backend/common/views.py:
            #
            # SyntaxError: f-string expression part cannot include a backslash
            print(f"Error parsing file {file_path}: {e}")
            return None
        return self.tree

    def print_ast(
        self,
        node: ast.AST,
        level: int = 0,
        output_file: str = None,
        output_lines: list = None,
    ):
        """Recursively print the AST in a readable format and save to log file."""
        # Initialize output_lines list on first call
        if output_lines is None:
            output_lines = ["Abstract Syntax Tree:"]
            log_path = self._get_log_path(output_file, "ast")
            is_root_call = True
        else:
            is_root_call = False

        node_type = str(node.type)
        indent = self._get_indent(level)
        line = f"{indent}{node_type}"

        # Add to output lines and print to stdout
        output_lines.append(line)

        # Recursively process children
        for child in node.children:
            self.print_ast(child, level + 1, output_file, output_lines)

        # Write to log file if this is the root call
        if is_root_call:
            log_path = self._get_log_path(output_file, "ast")
            with open(log_path, "w") as f:
                f.write("\n".join(output_lines))
            print(f"\nAST log saved to: {log_path}")

    def _get_indent(self, level: int) -> str:
        """Generate the appropriate indentation for the current level."""
        return "  " * level

    def _get_log_path(
        self, output_file: str = None, log_type: str = "call_graph"
    ) -> Path:
        """Generate the log file path with timestamp.

        Args:
            output_file: Base name for the output directory
            log_type: Type of log file ("call_graph" or "ast")
        """
        output_file = self._output_file_slug(output_file)
        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%I:%M_%p"
        )  # e.g. 2024-03-20_02:30_PM
        log_dir = Path("logs") / f"{output_file}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{log_type}.log"

    def print_call_graph(self, output_file: str = None):
        """Print the generated call graph to both stdout and a log file."""
        log_path = self._get_log_path(output_file, "call_graph")

        # Create a list to store the output lines
        output_lines = ["Call Graph Analysis:"]
        for full_name, info in self.functions.items():
            # Include the file name in the output
            output_lines.append(f"\nFunction: {full_name}")
            output_lines.append(f"  File: {info.file}")
            if info.calls:
                output_lines.append("  Calls:")
                for call in sorted(info.calls, key=lambda x: x.key()):
                    output_lines.append(f"    → {call.key()}")
            if info.called_by:
                output_lines.append("  Called by:")
                for caller in sorted(info.called_by, key=lambda x: x.key()):
                    output_lines.append(f"    ← {caller.key()}")

        # Write to log file
        with open(log_path, "w") as f:
            f.write("\n".join(output_lines))

        print(f"\nCall graph log saved to: {log_path}")

    def get_leaf_nodes(self) -> List[FunctionNode]:
        """Return all FunctionNode objects that don't call any other functions."""
        return [node for node in self.functions.values() if not node.calls]

    def get_nodes_at_level(self, level: int) -> List[FunctionNode]:
        """Return all FunctionNode objects at a specific level in the call tree.
        Level 0 represents leaf nodes (functions that don't call others).
        Level 1 represents functions that only call leaf nodes.
        Level 2 represents functions that call level 1 nodes, and so on.
        """
        if level < 0:
            return []

        # For level 0, return leaf nodes
        if level == 0:
            return self.get_leaf_nodes()

        all_prev_level_nodes = set()
        for level in range(level):
            all_prev_level_nodes.update(
                {node.key() for node in self.get_nodes_at_level(level)}
            )

        # For other levels, find nodes that only call nodes from previous levels
        result = []
        for func in self.functions.values():
            # Skip if already found in previous levels
            if any(func.key() in nodes for nodes in [all_prev_level_nodes]):
                continue

            # Check if all calls are to previous level nodes
            if func.calls and all(
                call.key() in all_prev_level_nodes for call in func.calls
            ):
                result.append(func)

        return result

    def _output_file_slug(self, output_file: str = None) -> str:
        # Determine the output file name based on project_path or files
        if not output_file:
            if self.project_path:
                output_file = self.project_path.name
            elif self.files:
                file_names = [f.stem[:20] for f in self.files]
                output_file = "_".join(file_names)

        return output_file

    def visualize_graph(self, output_file: str = None, view: bool = True):
        """Create a visual representation of the call graph using graphviz.

        Args:
            output_file: Name of the output file (without extension)
            view: Whether to automatically open the generated graph
        """
        output_file = self._output_file_slug(output_file)

        # Create a new directed graph
        dot = Digraph(comment="Call Graph")
        dot.attr(rankdir="TB")  # Top to bottom layout

        # Track processed nodes to avoid duplicates
        processed: Set[str] = set()

        # Helper function to add nodes and edges recursively
        def add_node_and_edges(namespace: str, processed: Set[str]):
            stack = [namespace]

            while stack:
                current_namespace = stack.pop()
                if current_namespace in processed:
                    continue

                func_node = self.functions.get(current_namespace)
                if not func_node:
                    continue

                # Add the current node
                node_label = f"name={func_node.name}\nnamespace={func_node.namespace}\nfile={func_node.file}"
                dot.node(current_namespace, node_label)
                processed.add(current_namespace)

                # Add edges for each function call
                for called_func in func_node.calls:
                    # Add the called function node if it hasn't been processed
                    if called_func.key() not in processed:
                        called_node = self.functions.get(called_func.key())
                        if called_node:
                            called_label = f"name={called_node.name}\nnamespace={called_node.namespace}\nfile={called_node.file}"
                            dot.node(called_func.key(), called_label)

                    # Add the edge
                    dot.edge(current_namespace, called_func.key())

                    # Add the called function to the stack for further processing
                    stack.append(called_func.key())

                processed.add(current_namespace)

        # Start with root nodes (functions that aren't called by others)
        root_nodes = {
            name for name, node in self.functions.items() if not node.called_by
        }

        # Process each root node
        for root in root_nodes:
            add_node_and_edges(root, processed)

        # Set graph attributes for better visualization
        dot.attr("node", shape="box", style="rounded")
        dot.attr("edge", arrowsize="0.5")

        # Save the graph
        dot.render(output_file, view=view, cleanup=True)
        print(f"Graph saved as {output_file}.pdf")

    def _get_source_code(self, node: ast.FunctionDef, source: str) -> str:
        """Extract source code for a given AST node."""
        lines = source.splitlines()
        start_line = node.lineno - 1
        end_line = node.end_lineno
        return "\n".join(lines[start_line:end_line])


class Translator:

    def __init__(self, analyzer: CallGraphAnalyzer):
        self.analyzer = analyzer

    def _setup_project(self):
        self.project_path: Path = self.analyzer.project_path.with_name(
            self.analyzer.project_path.name + "_go"
        )
        self.files = self.analyzer.files
        self.project_path.mkdir(parents=True, exist_ok=True)

    def _setup_files(self, files: list[Path]):
        for file in files:
            # For each file, we'll create a new file in the project_path
            # with the same name but with a .go extension
            relative_path = file.relative_to(self.analyzer.project_path)
            new_file_path = self.project_path / relative_path.with_suffix(".go")
            print(f"Creating {new_file_path}")
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not new_file_path.exists():
                with open(new_file_path, "w") as new_file:
                    new_file.write("")
        # TODO (adam) Init git repo, setup go.mod, etc.

    def _find_nodes_with_exclusive_callers(
        self,
    ) -> list[tuple[FunctionNode, list[FunctionNode]]]:
        nodes_and_exclusive_callers = []
        for node in self.analyzer.get_leaf_nodes():
            if node.is_test():
                continue
            # Callers that exclusively call this function and no others. In the future, we can get more sophisticated
            # and get callers that only call functions at the same level as this one, but we're starting simple
            exclusive_callers = [
                caller
                for caller in node.called_by
                if len(caller.calls) == 1 and caller.is_test()
            ]
            if exclusive_callers:
                print(f"Translating {node.name}")
                print(f"  File: {node.file}")
                print(f"  Scope: {node.scope} module_level? {node.scope.is_module()}")
                nodes_and_exclusive_callers.append((node, exclusive_callers))
            for caller in exclusive_callers:
                print(f"  Exclusive caller: {caller.name}")
        return nodes_and_exclusive_callers

    def _translate_tree(self, node: FunctionNode, exclusive_callers: list[FunctionNode]):
        pass

    def translate(self) -> str:
        print(f"Found {len(self.analyzer.functions)} functions")
        nodes_and_exclusive_callers = self._find_nodes_with_exclusive_callers()
        for node, exclusive_callers in nodes_and_exclusive_callers:
            self._setup_project()
            files = [Path(f.file) for f in exclusive_callers] + [Path(node.file)]
            self._setup_files(files)
            self._translate_tree(node, exclusive_callers)


@click.command()
@click.option(
    "--project-path",
    type=click.Path(exists=True),
    help="Path to the project directory to analyze",
)
@click.option(
    "--files",
    type=click.Path(exists=True),
    multiple=True,
    help="One or more files to analyze",
)
@click.option(
    "--language",
    type=click.Choice(["python", "swift"], case_sensitive=False),
    default="python",
    help="Programming language to analyze",
)
def main(project_path, files, language):
    """Analyze Python or Swift code and build a call graph."""
    if not project_path and not files:
        raise click.UsageError("Either --project-path or --files must be provided")
    if project_path and files:
        raise click.UsageError("Cannot use both --project-path and --files together")

    analyzer = CallGraphAnalyzer(
        language=language,
        project_path=project_path if project_path else None,
        files=files if files else None,
    )
    analyzer.analyze()

    translator = Translator(analyzer)
    translator.translate()

    # analyzer.print_call_graph()

    # print("Generating graph...")
    # analyzer.visualize_graph()

    print("Done.")


if __name__ == "__main__":
    main()
