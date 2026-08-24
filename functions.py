from math import *


# Binary Operations ---------------------------------------------------


def addition(interval1, interval2):
    [x1,y1] = interval1
    [x2,y2] = interval2
    return (x1 + x2, y1 + y2)


def minus(interval1, interval2):
    [x1,y1] = interval1
    [x2,y2] = interval2
    return (x1 - y2, y1 - x2)


def multiply(interval1, interval2):
    [x1,y1] = interval1
    [x2,y2] = interval2
    products = (x1 * x2, x1 * y2, y1 * x2, y1 * y2)
    return [min(products), max(products)]


def divide(interval1, interval2):
    [x1, y1] = interval1
    [x2, y2] = interval2
    if x2 <= 0 <= y2:
        raise ValueError("Divisão por intervalo contendo zero")
    divisions = (x1/x2, x1/y2, y1/x2, y1/y2)
    return [min(divisions), max(divisions)]


# Unary Operations ---------------------------------------------------


def exponential(interval1):
    [x1, y1] = interval1
    return [exp(x1), exp(y1)]


def logarithm(interval1):
    [x1, y1] = interval1
    if (y1 <= 0 or x1 <= 0):
        raise ValueError("Log indefinido para intervalo ≤ 0")
    return [log(x1), log(y1)]


def power(interval1, num):
    [x1, y1] = interval1
    if num == 0:
        return [1, 1]  # x^0 = 1 sempre
    if (num % 2 == 0):
        if (x1 > 0):
            return [pow(x1, num), pow(y1, num)]
        elif (y1 < 0):
            return [pow(y1, num), pow(x1, num)]
        elif (x1 <= 0 <= y1):
            return [0, max(pow(x1, num), pow(y1, num))]
    else:
        return [pow(x1, num), pow(y1, num)]


def root(interval1, num):
    [x1, y1] = interval1
    if x1 < 0:
        raise ValueError("Raiz indefinida para intervalo negativo")
    return [x1**(1/num), y1**(1/num)]


# Comparação ---------------------------------------------------------


def compare(interval1, interval2):
    [x1, y1] = interval1
    [x2, y2] = interval2
    if (x1 >= x2 and y1 <= y2):
        return 1
    else:
        return 0
