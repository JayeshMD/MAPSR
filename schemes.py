def Rk4(fun, t, x, h):
    k1 = fun(t, x)
    k2 = fun(t + h / 2, x + h * k1 / 2)
    k3 = fun(t + h / 2, x + h * k2 / 2)
    k4 = fun(t + h, x + h * k3)
    xn = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xn
