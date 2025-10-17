import math
from dataclasses import dataclass

# Math behind the model
def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d1_d2(S, K, T, r, sigma, q):
    if any(v <= 0 for v in (S, K, T, sigma)):
        raise ValueError("S, K, T, sigma must be > 0.")
    vsqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrtT
    d2 = d1 - vsqrtT
    return d1, d2

def bs_price(S, K, T, r, sigma, q, option):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option == "call":
        return S * disc_q * _Phi(d1) - K * disc_r * _Phi(d2)
    elif option == "put":
        return K * disc_r * _Phi(-d2) - S * disc_q * _Phi(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'")

@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

def bs_greeks(S, K, T, r, sigma, q, option):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf = _phi(d1)
    if option == "call":
        delta = disc_q * _Phi(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T))
                 - r * K * disc_r * _Phi(d2)
                 + q * S * disc_q * _Phi(d1))
        rho = K * T * disc_r * _Phi(d2)
    else:  # put
        delta = disc_q * (_Phi(d1) - 1.0)
        theta = (-S * disc_q * pdf * sigma / (2 * math.sqrt(T))
                 + r * K * disc_r * _Phi(-d2)
                 - q * S * disc_q * _Phi(-d1))
        rho = -K * T * disc_r * _Phi(-d2)
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)  # per 1.0 change in sigma
    return Greeks(delta, gamma, vega, theta, rho)

def implied_vol(target_price, S, K, T, r, q, option,
                seed=0.2, tol=1e-8, max_iter=100):
    sigma = max(1e-6, float(seed))
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, q, option)
        diff = price - target_price
        if abs(diff) < tol:
            return max(1e-12, sigma)
        vega = bs_greeks(S, K, T, r, sigma, q, option).vega
        if vega < 1e-12:
            break
        sigma = min(5.0, max(1e-6, sigma - diff / vega))
    # bisection fallback
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        pmid = bs_price(S, K, T, r, mid, q, option)
        if abs(pmid - target_price) < tol:
            return mid
        if pmid > target_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

# Input error case
def _f(prompt, default=None, cast=float):
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not s and default is not None:
            return cast(default)
        try:
            return cast(s)
        except Exception:
            print("Invalid. Try again.")

def _option():
    while True:
        o = input("Option type (call/put): ").strip().lower()
        if o in ("call", "put"):
            return o
        print("Type 'call' or 'put'.")

def main():
    print("Black–Scholes Options Pricing Model")
    S = _f("Spot S")
    K = _f("Strike K")
    T = _f("Time to expiry T in years")
    r = _f("Risk-free rate r (in decimals)")
    q = _f("Dividend yield q (0 if none)")
    sigma = _f("Volatility sigma (in decimals)")
    option = _option()

    price = bs_price(S, K, T, r, sigma, q, option)
    g = bs_greeks(S, K, T, r, sigma, q, option)

    print("\n--- Results ---")
    print(f"Price: {price:.6f}")
    print(f"Delta: {g.delta:.6f}")
    print(f"Gamma: {g.gamma:.6f}")
    print(f"Vega : {g.vega:.6f}")
    print(f"Theta: {g.theta:.6f}")
    print(f"Rho  : {g.rho:.6f}")

    # Optional IV solve
    ans = input("\nCompute implied vol from a market price? (y/n): ").strip().lower()
    if ans == "y":
        target = _f("Market option price")
        iv = implied_vol(target, S, K, T, r, q, option, seed=sigma)
        print(f"Implied vol: {iv:.6f}")

if __name__ == "__main__":
    main()
