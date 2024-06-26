from sympy import Or, simplify


class Expression:
    def __init__(self, expr) -> None:
        self.expr = expr

    def simplify(self):
        terms = self.expr.args
        n_terms = len(terms)

        simplified_terms = []

        # Simplify the first and last terms
        first_term = terms[0]
        last_term = terms[-1]
        simplified_terms.append(simplify(first_term | last_term))

        # Simplify the remaining pairs of terms
        for i in range(1, n_terms // 2):
            term1 = terms[i]
            term2 = terms[n_terms - 1 - i]
            res = simplify(term1 | term2)
            # Check if res.args has an Or element inside
            has_or = any(isinstance(arg, Or) for arg in res.args)
            if has_or:
                res = term1 | term2
            simplified_terms.append(res)

        # If there is an odd number of terms, include the middle term
        if n_terms % 2 != 0:
            middle_term = terms[n_terms // 2]
            simplified_terms.append(middle_term)

        # Combine the simplified terms using Or
        simplified_expr = Or(*simplified_terms)

        return Expression(simplified_expr)
