from sqlglot import exp
from probsql.constructs import REGISTRY

def find_constructs(ast):
    """Return every PSQL_* call in the query tree, as a list of nodes.

    Names are matched case-insensitively. A call with the wrong number of
    arguments raises ValueError.
    """
    found = []
    for node in ast.find_all(exp.Anonymous):
        name = node.name.upper()
        if name not in REGISTRY:
            continue
        expected = REGISTRY[name].n_args
        got = len(node.expressions)
        if got != expected:
            raise ValueError(f"{name} takes {expected} arguments, got {got}")
        found.append(node)
    return found