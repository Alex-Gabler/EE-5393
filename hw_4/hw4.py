from itertools import product


PERMS = {
    "A": (1, 4, 3, 5, 2),
    "B": (1, 4, 5, 2, 3),
    "C": (1, 3, 4, 2, 5),
    "D": (1, 2, 4, 5, 3),
    "E": (1, 4, 2, 3, 5),
}


def cycle_to_map(cycle):
    p = list(range(5))
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        p[a - 1] = b - 1
    return tuple(p)


def compose(p, q):
    return tuple(q[i] for i in p)


def inverse(p):
    inv = [0] * len(p)
    for i, x in enumerate(p):
        inv[x] = i
    return tuple(inv)


ID = tuple(range(5))
MAPS = {"*": ID}
for name, cycle in PERMS.items():
    m = cycle_to_map(cycle)
    MAPS[name] = m
    MAPS[name + "'"] = inverse(m)


def parse_product(expr):
    if expr == "*":
        return ["*"]
    out = []
    i = 0
    while i < len(expr):
        if expr[i] in "ABCDE":
            token = expr[i]
            if i + 1 < len(expr) and expr[i + 1] == "'":
                token += "'"
                i += 1
            out.append(token)
        i += 1
    return out


def perm_expr(expr):
    result = ID
    for token in parse_product(expr):
        result = compose(result, MAPS[token])
    return result


def inv_expr(expr):
    if expr == "*":
        return "*"
    tokens = parse_product(expr)
    inv_tokens = []
    for token in reversed(tokens):
        inv_tokens.append(token[:-1] if token.endswith("'") else token + "'")
    return "".join(inv_tokens)


def mul_expr(left, right):
    if left == "*":
        return right
    if right == "*":
        return left
    return left + right


FACTORS = {
    "A": ("C", "B"),
    "B": ("C", "D"),
    "C": ("D", "E"),
    "D": ("E", "B"),
    "E": ("D", "A"),
    "A'": ("B", "C"),
    "B'": ("D", "C"),
    "C'": ("E", "D"),
    "D'": ("B", "E"),
    "E'": ("A", "D"),
}


class Var:
    def __init__(self, name):
        self.name = name


class Not:
    def __init__(self, arg):
        self.arg = arg


class And:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Nand:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Or:
    def __init__(self, left, right):
        self.left = left
        self.right = right


def rows_for(formula, target):
    if isinstance(formula, Var):
        return [(formula.name, target, "*")]

    if isinstance(formula, Not):
        if isinstance(formula.arg, Var):
            return [(formula.arg.name, "*", target)]
        return rows_for(formula.arg, inv_expr(target)) + [("*", target, target)]

    if isinstance(formula, Or):
        return rows_for(Nand(Not(formula.left), Not(formula.right)), target)

    if isinstance(formula, And):
        q, r = FACTORS[target]
        return (
            rows_for(formula.left, q)
            + rows_for(formula.right, r)
            + rows_for(formula.left, inv_expr(q))
            + rows_for(formula.right, inv_expr(r))
        )

    if isinstance(formula, Nand):
        return rows_for(And(formula.left, formula.right), inv_expr(target)) + [
            ("*", target, target)
        ]

    raise TypeError(formula)


def simulate(rows, inputs):
    state = ID
    for cond, if_one, if_zero in rows:
        value = 1 if cond == "*" else inputs[cond]
        state = compose(state, perm_expr(if_one if value else if_zero))
    return state


def bool_eval(name, x):
    x1, x2, x3 = x["x1"], x["x2"], x["x3"]
    return {
        "f1": x1 and (x2 or x3),
        "f2": x2 or (x1 and x3),
        "f3": x3 and (x1 or x2),
        "f4": x1 or (x2 and x3),
        "f5": x2 and (x1 or x3),
        "f6": x3 or (x1 and x2),
    }[name]


X1, X2, X3 = Var("x1"), Var("x2"), Var("x3")
FORMULAS = {
    "f1": And(X1, Or(X2, X3)),
    "f2": Or(X2, And(X1, X3)),
    "f3": And(X3, Or(X1, X2)),
    "f4": Or(X1, And(X2, X3)),
    "f5": And(X2, Or(X1, X3)),
    "f6": Or(X3, And(X1, X2)),
}


def format_row(row):
    cond, if_one, if_zero = row
    return f"{{{cond}: {if_one}, {if_zero}}}"


if __name__ == "__main__":
    for name, formula in FORMULAS.items():
        rows = rows_for(formula, "A")
        for bits in product([0, 1], repeat=3):
            inputs = dict(zip(["x1", "x2", "x3"], bits))
            result = simulate(rows, inputs)
            expected = bool_eval(name, inputs)
            assert (result != ID) == bool(expected), (name, inputs, result)

        print(f"{name}: {len(rows)} rows")
        for row in rows:
            print(format_row(row))
        print()
