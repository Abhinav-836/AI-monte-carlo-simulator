"""
Advanced Portfolio Optimization with Multiple Strategies
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

warnings.filterwarnings("ignore")


class PortfolioOptimizer:
    """`returns` and `cov_matrix` MUST be in the same frequency (both daily or both annual)."""

    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray,
                 asset_names: Optional[List[str]] = None):
        self.returns = np.asarray(returns, float)
        self.cov_matrix = np.asarray(cov_matrix, float)
        self.n_assets = len(self.returns)
        self.asset_names = list(asset_names) if asset_names is not None else None
        self.bounds = [(0.0, 1.0)] * self.n_assets
        self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        self._opts = {"ftol": 1e-10, "maxiter": 500}

    # ---------------- helpers ----------------
    def _vol(self, w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ self.cov_matrix @ w, 0.0)))

    def _objective(self, method: str, rf: float) -> Callable[[np.ndarray], float]:
        cov, mu, n = self.cov_matrix, self.returns, self.n_assets
        asset_vol = np.sqrt(np.diag(cov))

        if method == "sharpe":
            return lambda w: -((w @ mu - rf) / (self._vol(w) + 1e-8))
        if method == "min_vol":
            return lambda w: float(w @ cov @ w)
        if method == "max_return":
            return lambda w: -float(w @ mu)
        if method == "risk_parity":
            def obj(w):
                var = w @ cov @ w + 1e-12
                rc = w * (cov @ w) / var            # fractional contributions (sum to 1)
                return float(np.sum((rc - 1.0 / n) ** 2))   # well scaled -> SLSQP converges
            return obj
        if method == "max_diversification":
            return lambda w: -float((w @ asset_vol) / (self._vol(w) + 1e-8))
        raise ValueError(f"Unknown method: {method}")

    def _clean(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(w, float), 0.0, 1.0)
        s = w.sum()
        return w / s if s > 1e-12 else np.ones(self.n_assets) / self.n_assets

    def _solve(self, method: str, rf: float = 0.02, risk_tolerance: Optional[float] = None) -> Dict:
        obj = self._objective(method, rf)
        cons = list(self.constraints)
        if method == "max_return" and risk_tolerance is not None:
            cons.append({"type": "ineq", "fun": lambda x: risk_tolerance - self._vol(x)})

        res = minimize(obj, np.ones(self.n_assets) / self.n_assets, method="SLSQP",
                       bounds=self.bounds, constraints=cons, options=self._opts)
        if res.success and np.all(np.isfinite(res.x)):
            w = self._clean(res.x)
        else:
            w = self._fallback(obj, method, risk_tolerance)
        return self._calculate_portfolio_stats(w, rf)

    def _fallback(self, obj: Callable, method: str, risk_tolerance: Optional[float]) -> np.ndarray:
        """Differential evolution with weight normalisation (DE can't take dict constraints)."""
        def penalised(x):
            w = self._clean(x)
            val = obj(w)
            if method == "max_return" and risk_tolerance is not None:
                val += 1e3 * max(0.0, self._vol(w) - risk_tolerance)
            return val

        res = differential_evolution(penalised, bounds=self.bounds, maxiter=300, popsize=15,
                                     seed=0, tol=1e-7, polish=False)
        return self._clean(res.x) if np.all(np.isfinite(res.x)) else np.ones(self.n_assets) / self.n_assets

    # ---------------- strategies ----------------
    def optimize_max_sharpe(self, risk_free_rate: float = 0.02) -> Dict:
        return self._solve("sharpe", risk_free_rate)

    def optimize_min_volatility(self) -> Dict:
        return self._solve("min_vol")

    def optimize_max_return(self, risk_tolerance: float = 1.0) -> Dict:
        return self._solve("max_return", risk_tolerance=risk_tolerance)

    def optimize_risk_parity(self) -> Dict:
        return self._solve("risk_parity")

    def optimize_max_diversification(self) -> Dict:
        return self._solve("max_diversification")

    def optimize_black_litterman(self, views: Dict[str, Tuple[float, float]], tau: float = 0.05,
                                 risk_free_rate: float = 0.02) -> Dict:
        """views: {asset: (expected_return, view_variance)}; needs asset_names."""
        if self.asset_names is None:
            raise ValueError("Pass asset_names to PortfolioOptimizer to use Black-Litterman views")

        w_mkt = np.ones(self.n_assets) / self.n_assets
        pi = 2.5 * self.cov_matrix @ w_mkt  # implied equilibrium returns

        k = len(views)
        P, Q, Omega = np.zeros((k, self.n_assets)), np.zeros(k), np.zeros((k, k))
        for i, (asset, (q, unc)) in enumerate(views.items()):
            P[i, self.asset_names.index(asset)] = 1.0
            Q[i] = q
            Omega[i, i] = max(float(unc), 1e-8)

        tau_cov = tau * self.cov_matrix
        posterior = pi + tau_cov @ P.T @ np.linalg.solve(P @ tau_cov @ P.T + Omega, Q - P @ pi)

        original = self.returns
        self.returns = posterior
        try:
            return self.optimize_max_sharpe(risk_free_rate)
        finally:
            self.returns = original

    def generate_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        min_vol = self.optimize_min_volatility()
        r_lo = float(min_vol["return"])
        r_hi = float(np.max(self.returns))
        if r_hi <= r_lo:
            return pd.DataFrame([{"return": r_lo, "volatility": min_vol["volatility"],
                                  "weights": np.array(min_vol["weights"])}])

        rows = []
        for target in np.linspace(r_lo, r_hi, n_points):
            cons = list(self.constraints) + [
                {"type": "eq", "fun": lambda x, t=target: float(x @ self.returns) - t}]
            res = minimize(lambda x: float(x @ self.cov_matrix @ x),
                           np.ones(self.n_assets) / self.n_assets, method="SLSQP",
                           bounds=self.bounds, constraints=cons, options=self._opts)
            if res.success:
                w = self._clean(res.x)
                rows.append({"return": float(w @ self.returns), "volatility": self._vol(w), "weights": w})
        return pd.DataFrame(rows)

    def _calculate_portfolio_stats(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
        w = np.asarray(weights, float)
        ret = float(w @ self.returns)
        vol = self._vol(w)
        weighted_vol = float(w @ np.sqrt(np.diag(self.cov_matrix)))
        return {
            "weights": w.tolist(),
            "return": ret,
            "volatility": vol,
            "sharpe": float((ret - risk_free_rate) / (vol + 1e-8)),
            "diversification_ratio": float(weighted_vol / (vol + 1e-8)),
            # rough parametric heuristic (not a simulated drawdown), based on PORTFOLIO volatility
            "max_drawdown": float(-0.5 * 2.33 * vol),
        }

    @staticmethod
    def optimize_portfolio(returns: np.ndarray, cov_matrix: np.ndarray, method: str = "max_sharpe",
                           risk_free_rate: float = 0.02, asset_names: Optional[List[str]] = None,
                           **kwargs) -> Dict:
        opt = PortfolioOptimizer(returns, cov_matrix, asset_names)
        if method == "max_sharpe":
            return opt.optimize_max_sharpe(risk_free_rate, **kwargs)
        methods = {
            "min_vol": opt.optimize_min_volatility,
            "max_return": opt.optimize_max_return,
            "risk_parity": opt.optimize_risk_parity,
            "max_diversification": opt.optimize_max_diversification,
        }
        if method not in methods:
            raise ValueError(f"Unknown method: {method}")
        return methods[method](**kwargs)


class FactorModel:
    def __init__(self, returns: np.ndarray, factor_returns: np.ndarray):
        self.returns = np.asarray(returns, float)
        fr = np.asarray(factor_returns, float)
        self.factor_returns = fr.reshape(-1, 1) if fr.ndim == 1 else fr
        self.n_assets = self.returns.shape[1]
        self.n_factors = self.factor_returns.shape[1]
        self.betas = self.alpha = self.residuals = self.r_squared = None

    def fit(self):
        X = np.column_stack([np.ones(len(self.factor_returns)), self.factor_returns])
        self.betas = np.zeros((self.n_assets, self.n_factors))
        self.alpha = np.zeros(self.n_assets)
        self.r_squared = np.zeros(self.n_assets)
        self.residuals = np.zeros(self.returns.shape)

        for i in range(self.n_assets):
            y = self.returns[:, i]
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            self.alpha[i], self.betas[i] = coef[0], coef[1:]
            self.residuals[:, i] = y - X @ coef
            ss_tot = np.sum((y - y.mean()) ** 2)
            self.r_squared[i] = 1 - np.sum(self.residuals[:, i] ** 2) / (ss_tot + 1e-12)

    def get_factor_exposures(self) -> Dict:
        if self.betas is None:
            self.fit()
        return {"betas": self.betas, "alpha": self.alpha, "r_squared": self.r_squared}

    def calculate_factor_risk(self, weights: np.ndarray) -> Dict:
        if self.betas is None:
            self.fit()
        w = np.asarray(weights, float)
        factor_cov = np.atleast_2d(np.cov(self.factor_returns, rowvar=False))  # 1-factor safe
        port_betas = w @ self.betas
        factor_risk = float(port_betas @ factor_cov @ port_betas)
        idio_risk = float(np.sum(w ** 2 * np.var(self.residuals, axis=0, ddof=1)))
        total = factor_risk + idio_risk
        return {
            "total_risk": total, "factor_risk": factor_risk, "idio_risk": idio_risk,
            "factor_risk_pct": factor_risk / (total + 1e-12),
            "idio_risk_pct": idio_risk / (total + 1e-12),
        }