from tree_sitter import Language, Parser, Query

SWIFT_LANGUAGE_LIBRARY_PATH = "tree-sitter-swift/"

Language.build_library("build/swift.so", [SWIFT_LANGUAGE_LIBRARY_PATH])

SWIFT_LANGUAGE = Language("build/swift.so", "swift")

parser = Parser()
parser.set_language(SWIFT_LANGUAGE)

code = b"""
func add(a: Int, b: Int) -> Int {
    return a + b
}
"""

tree = parser.parse(code)

root_node = tree.root_node
print(root_node)

query_string = """
(function_declaration) @function
"""

query = SWIFT_LANGUAGE.query(query_string)

captures = query.captures(root_node)

for capture in captures:
    node = capture[0]
    print(f"Function found: {node.text.decode('utf-8')} at position {node.start_point}")
