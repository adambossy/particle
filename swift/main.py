from typing import Any

from tree_sitter import Language, Parser, Query

SWIFT_LANGUAGE_LIBRARY_PATH = "tree-sitter-swift/"

Language.build_library("build/swift.so", [SWIFT_LANGUAGE_LIBRARY_PATH])

SWIFT_LANGUAGE = Language("build/swift.so", "swift")


class SwiftFile:

    parser = Parser()
    parser.set_language(SWIFT_LANGUAGE)

    def __init__(self, path):
        self.path = path

        with open(self.path) as f:
            code = f.read().encode("utf-8")

        self.tree = self.parser.parse(code)
        self.root_node = self.tree.root_node

    def get_function_declarations(self):
        query_string = """
        (function_declaration
            name: (identifier) @function_name
            signature: (function_signature
                input: (parameter_clause) @parameters
                output: (return_clause) @return_type))
"""
        query = SWIFT_LANGUAGE.query(query_string)
        captures = query.captures(self.root_node)

        # nodes = []
        # for capture in captures:
        #     node, _ = capture
        #     import pdb; pdb.set_trace()
        #     nodes.append(node)

        # return [
        #     SwiftFunction(node.text.decode("utf-8"), node.start_point, node.end_point)
        #     for node in nodes
        # ]

        functions = []
        for capture in captures:
            function_node, _ = capture
            function_name = function_node.child_by_field_name("identifier").text.decode(
                "utf-8"
            )
            parameter_nodes = function_node.child_by_field_name("parameters").children
            parameters = []
            for parameter_node in parameter_nodes:
                parameter_name = parameter_node.child_by_field_name(
                    "identifier"
                ).text.decode("utf-8")
                parameter_type = parameter_node.child_by_field_name(
                    "type_annotation"
                ).text.decode("utf-8")
                parameters.append((parameter_name, parameter_type))
            return_type_node = function_node.child_by_field_name("return")
            return_type = return_type_node.child_by_field_name(
                "type_annotation"
            ).text.decode("utf-8")
            functions.append((function_name, parameters, return_type))
        return functions

    def __str__(self):
        return f"Swift file at {self.path} with functions {self.functions}"


# class SwiftClass:
#     def __init__(self) -> None:


class SwiftFunction:
    def __init__(self, name, start_point, end_point):
        self.name = name
        self.start_point = start_point
        self.end_point = end_point

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        return (
            f"Function {self.name} at position {self.start_point} to {self.end_point}"
        )


if __name__ == "__main__":
    swift_file = SwiftFile(
        "../repos/todoList-iOS/todoList/ViewControllers/DailyTasksViewController.swift"
    )
    print(swift_file.get_function_declarations())
