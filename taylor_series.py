import sympy as sp
from functions import power, multiply, addition


def taylor_polynomial(function, a, order):
    x = sp.Symbol('x')
    f = sp.sympify(function)
    serie = 0
    for i in range(order + 1):
        deriv = sp.diff(f, x, i).subs(x, a)
        serie += deriv * (x - a) ** i / sp.factorial(i)
    return sp.expand(serie)


def evaluate_taylor_on_interval(poly, interval):
    x = sp.Symbol('x')
    poly = sp.expand(poly)
    terms = sp.Add.make_args(poly)

    result = (0.0, 0.0)

    for term in terms:
        coeff, pow_part = term.as_coeff_exponent(x)
        c = float(coeff)
        n = int(pow_part)

        term_int = power(interval, n)
        term_int = multiply((c, c), term_int)
        result = addition(result, term_int)

    return result
