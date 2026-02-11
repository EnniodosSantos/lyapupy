import decimal as dc
import math
import matplotlib.pyplot as plt
import numpy as np
import random

# Configuração interna
_CTX = dc.getcontext()
# --------------------------------------------------------------------------
# Private Definitions
# --------------------------------------------------------------------------

# Constants (Internal use only)
_r_logistic = 4
_r_tent = dc.Decimal('2')
_b_asymm = dc.Decimal('0.4')
_pi = _CTX.create_decimal_from_float(math.pi)
_ln2 = _CTX.ln(dc.Decimal(2))

# --- Map Equations --------------------------------------------------------

# Ulam ---------------------------------------------------------------------

_ulam_map = lambda x: 1 - 2 * x**2
_ulam_df = lambda x: -4 * x
_ulam_lyap = _ln2

# Logistic------------------------------------------------------------------

_logistic_map = lambda x: _r_logistic * x * (1 - x)
_logistic_df = lambda x: _r_logistic * (1 - 2 * x)
_logistic_lyap = _ln2

# Bernoulli-----------------------------------------------------------------

_bernoulli_map = lambda x: (2 * x) % dc.Decimal(1)
_bernoulli_df = lambda x: dc.Decimal(2)
_bernoulli_lyap = _ln2

# Gauss----------------------------------------------------------------------

def _gauss_map(x):
    if x == 0: return dc.Decimal(0)
    inv_x = 1 / x
    return inv_x - inv_x.to_integral_value(rounding=dc.ROUND_FLOOR)

def _gauss_df(x):
    if x == 0: return dc.Decimal(0)
    return -1 / (x**2)

_gauss_lyap = (_pi**2) / (dc.Decimal(6) * _ln2)

# Tent-------------------------------------------------------------------------

_tent_map = lambda x: _r_tent * min(x, 1 - x)
_tent_df = lambda x: _r_tent if x < 0.5 else -_r_tent
_tent_lyap = _r_tent.ln()

# Asymmetric Tent---------------------------------------------------------------
_asymmetric_tent_map = lambda x: (x / _b_asymm) if x < _b_asymm else ((1 - x) / (1 - _b_asymm))
_asymmetric_tent_df = lambda x: (1 / _b_asymm) if x < _b_asymm else (-1 / (1 - _b_asymm))
_asymmetric_tent_lyap = -(_b_asymm * _b_asymm.ln()) - ((1 - _b_asymm) * (1 - _b_asymm).ln())

# Chebyshev --------------------------------------------------------------------
_chebyshev_map = lambda x: 2 * x**2 - 1
_chebyshev_df = lambda x: 4 * x
_chebyshev_lyap = _ln2

# Cusp--------------------------------------------------------------------------
'''
_cusp_map = lambda x: 1 - 2 * (abs(x).sqrt())
_cusp_df = lambda x: (-1 / x.abs().sqrt()) if x >= 0 else (1 / x.abs().sqrt())
_cusp_lyap = dc.Decimal('0.5') * dc.Decimal(2).ln()
'''

# --- Master Dictionary --------------------------------------------------------

_chaotic_maps = {
    "Gauss": {"f": _gauss_map, "df": _gauss_df, "l": _gauss_lyap,
              "domain": {"min": (0),"max": (1)}},

    "Logistic": {"f": _logistic_map, "df": _logistic_df, "l": _logistic_lyap,
                 "domain": {"min": (0),"max": (1)}},

    "Bernoulli": {"f": _bernoulli_map, "df": _bernoulli_df, "l": _bernoulli_lyap,
                  "domain": {"min": (0),"max": (1)}},

    "Ulam":  {"f": _ulam_map,  "df": _ulam_df,  "l": _ulam_lyap,
              "domain": {"min": (-1),"max": (1)}},

    "Tent": {"f": _tent_map,  "df": _tent_df,  "l": _tent_lyap,
             "domain": {"min": (0),"max": (1)}},

    "Asymmetric Tent": {"f": _asymmetric_tent_map, "df": _asymmetric_tent_df, "l": _asymmetric_tent_lyap,
                        "domain": {"min": (0),"max": (1)}},

    "Chebyshev": {"f": _chebyshev_map, "df": _chebyshev_df, "l": _chebyshev_lyap,
                  "domain": {"min": (-1),"max": (1)}},

    # "Cusp": {"f": _cusp_map, "df": _cusp_df, "l": _cusp_lyap}
}

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def initial_condition(map_name):
   if map_name not in _chaotic_maps:
        raise ValueError(f"Map not found. Options: {available_maps()}")
   dom = _chaotic_maps[map_name]["domain"]
   a = dom["min"]
   b = dom["max"]
   value = random.uniform(a, b)
   return value

def lyapunov_convergence(map_name, x0, steps, trans, prec=50, dec=False, plot=False):

    if map_name not in _chaotic_maps: raise ValueError(f"Map not found. Options: {available_maps()}")
    dc.getcontext().prec = prec
    x = dc.Decimal(str(x0))

    f  = _chaotic_maps[map_name]["f"]
    df = _chaotic_maps[map_name]["df"]

    soma = dc.Decimal(0)
    lambda_evolution = []

    # Transient
    for _ in range(trans):
        x = f(x)

    # Lyapunov convergence
    for i in range(1, steps + 1):
        x = f(x)
        soma += dc.getcontext().ln(abs(df(x)))
        lambda_evolution.append(soma / dc.Decimal(i))

    # Plot (optional)
    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lambda_evolution, lw=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Lyapunov exponent")
        ax.set_title(f"Lyapunov convergence — {map_name}")
        ax.grid(True)
        plt.show()

    # Return
    if dec:
        return lambda_evolution
    else:
        return np.array([float(v) for v in lambda_evolution])

def available_maps(): return list(_chaotic_maps.keys())


def map_time_serie(map_name, x0, steps, trans, prec=50, dec=False, plot=False):

    if map_name not in _chaotic_maps: raise ValueError(f"Map not found. Options: {available_maps()}")

    dc.getcontext().prec = prec
    f, x = _chaotic_maps[map_name]["f"], dc.Decimal(str(x0))

    for _ in range(trans): x = f(x)

    data = [x := f(x) for _ in range(steps)]

    if plot: _plot(data, map_name)
    return data if dec else np.array(data, dtype=float)

def _plot(data, name):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(data, lw=1.5)
        ax.set_xlabel("n")
        ax.set_ylabel("$x_n$")
        ax.set_title(f"{name} - Time Serie")
        ax.grid(True)
        plt.show()

def lyapunov_estimated(map_name, x0, steps, trans, prec=50, dec=False):

    if map_name not in _chaotic_maps: raise ValueError(f"Map not found. Options: {available_maps()}")

    dc.getcontext().prec = prec
    x0 = dc.Decimal(str(x0))

    f  = _chaotic_maps[map_name]["f"]
    df = _chaotic_maps[map_name]["df"]

    x = x0
    soma = dc.Decimal(0)

    # Transiente
    for _ in range(trans):
        x = f(x)

    # Soma logarítmica
    for _ in range(steps):
        x = f(x)
        soma += dc.getcontext().ln(abs(df(x)))

    lambda_est = soma / dc.Decimal(steps)

    if dec:
        return lambda_est
    else:
        return float(lambda_est)


def lyapunov_estimated(map_name, x0, steps, trans, prec=50, dec=False):
    """
    Estimates the Lyapunov Exponent.
    """
    if map_name not in _chaotic_maps:
        raise ValueError(f"Map '{map_name}' not found.")

    dc.getcontext().prec = prec
    x0 = dc.Decimal(str(x0))

    f  = _chaotic_maps[map_name]["f"]
    df = _chaotic_maps[map_name]["df"]

    x = x0
    soma = dc.Decimal(0)

    for _ in range(trans):
        x = f(x)

    for _ in range(steps):
        x = f(x)
        deriv = df(x)
        # Proteção extra contra log(0)
        if deriv == 0:
            soma += dc.Decimal("-1e5")
        else:
            soma += dc.getcontext().ln(abs(deriv))

    lambda_est = soma / dc.Decimal(steps)

    return lambda_est if dec else float(lambda_est)

def theoretical_lyapunov(map_name, dec=False):
    """Returns the theoretical Lyapunov exponent for the map."""
    if map_name not in _chaotic_maps:
        raise ValueError(f"Map '{map_name}' not found.")

    lam = _chaotic_maps[map_name]["l"]
    return lam if dec else float(lam)

def lyapunov_summary(map_name, x0, steps, trans, prec=50, dec=False, plot=False):

    if map_name not in _chaotic_maps:
        raise ValueError(f"Map '{map_name}' not found.")

    series = map_time_serie(map_name, x0, steps, trans, prec, dec, plot)
    estimated = lyapunov_estimated(map_name, x0, steps, trans, prec, dec)
    theoretical = theoretical_lyapunov(map_name, dec)

    erro = abs((estimated - theoretical) / theoretical) * 100
    erro = erro if dec else float(erro)

    return {
        "map": map_name,
        "theoretical": theoretical,
        "estimated": estimated,
        "error": erro,
        "time_serie": series

    }

