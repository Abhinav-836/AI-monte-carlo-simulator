"""
Advanced Portfolio Optimization with Multiple Strategies
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer with multiple optimization strategies
    """
    
    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray):
        """
        Initialize optimizer
        
        Args:
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix of asset returns
        """
        self.returns = np.asarray(returns)
        self.cov_matrix = np.asarray(cov_matrix)
        self.n_assets = len(returns)
        
        # Constraints and bounds
        self.bounds = [(0, 1)] * self.n_assets
        self.constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
    
    def optimize_max_sharpe(self, risk_free_rate: float = 0.02) -> Dict:
        """Optimize for maximum Sharpe ratio"""
        
        def objective(weights):
            portfolio_return = np.sum(weights * self.returns)
            portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
            sharpe = (portfolio_return - risk_free_rate) / (portfolio_volatility + 1e-8)
            return -sharpe  # Minimize negative Sharpe
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints
        )
        
        if result.success:
            weights = result.x
            return self._calculate_portfolio_stats(weights, risk_free_rate)
        else:
            return self._fallback_optimization('sharpe', risk_free_rate)
    
    def optimize_min_volatility(self) -> Dict:
        """Optimize for minimum volatility"""
        
        def objective(weights):
            return np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints
        )
        
        if result.success:
            weights = result.x
            return self._calculate_portfolio_stats(weights)
        else:
            return self._fallback_optimization('min_vol')
    
    def optimize_max_return(self, risk_tolerance: float = 1.0) -> Dict:
        """Optimize for maximum return with risk constraint"""
        
        def objective(weights):
            return -np.sum(weights * self.returns)
        
        # Add risk constraint
        constraints = self.constraints.copy()
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: risk_tolerance - np.sqrt(x.T @ self.cov_matrix @ x)
        })
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            method='SLSQP',
            bounds=self.bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            return self._calculate_portfolio_stats(weights)
        else:
            return self._fallback_optimization('max_return')
    
    def optimize_risk_parity(self) -> Dict:
        """Optimize for risk parity (equal risk contribution)"""
        
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            marginal_contrib = (self.cov_matrix @ weights) / (portfolio_vol + 1e-8)
            risk_contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target = portfolio_vol / self.n_assets
            return np.sum((risk_contrib - target) ** 2)
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints
        )
        
        if result.success:
            weights = result.x
            return self._calculate_portfolio_stats(weights)
        else:
            return self._fallback_optimization('risk_parity')
    
    def optimize_max_diversification(self) -> Dict:
        """Optimize for maximum diversification ratio"""
        
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            weighted_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix)))
            diversification_ratio = weighted_vol / (portfolio_vol + 1e-8)
            return -diversification_ratio
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            method='SLSQP',
            bounds=self.bounds,
            constraints=self.constraints
        )
        
        if result.success:
            weights = result.x
            return self._calculate_portfolio_stats(weights)
        else:
            return self._fallback_optimization('max_diversification')
    
    def optimize_black_litterman(
        self,
        views: Dict[str, Tuple[float, float]],
        tau: float = 0.05
    ) -> Dict:
        """
        Black-Litterman optimization with views
        
        Args:
            views: Dict of views {asset: (expected_return, uncertainty)}
            tau: Confidence in prior
        """
        # Market capitalization weights (implied)
        market_weights = np.ones(self.n_assets) / self.n_assets
        
        # Implied excess returns
        lam = 2.5  # Risk aversion
        pi = lam * self.cov_matrix @ market_weights
        
        # Views matrix
        P = np.zeros((len(views), self.n_assets))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (asset, (view_return, view_uncertainty)) in enumerate(views.items()):
            asset_idx = list(self.returns.keys()).index(asset) if isinstance(self.returns, dict) else i
            P[i, asset_idx] = 1
            Q[i] = view_return
            Omega[i, i] = view_uncertainty
        
        # Black-Litterman posterior
        tau_inv = tau
        pi_inv = np.linalg.pinv(tau * self.cov_matrix)
        P_inv = np.linalg.inv(P @ (tau * self.cov_matrix) @ P.T + Omega)
        
        # Posterior returns
        posterior_returns = pi + (tau * self.cov_matrix) @ P.T @ P_inv @ (Q - P @ pi)
        
        # Optimize with posterior returns
        original_returns = self.returns
        self.returns = posterior_returns
        
        result = self.optimize_max_sharpe()
        
        # Restore original returns
        self.returns = original_returns
        
        return result
    
    def generate_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Generate efficient frontier"""
        results = []
        
        # Get min and max returns
        min_return = np.min(self.returns)
        max_return = np.max(self.returns)
        
        target_returns = np.linspace(min_return, max_return * 0.8, n_points)
        
        for target in target_returns:
            # Constraint for target return
            constraints = self.constraints.copy()
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * self.returns) - target
            })
            
            result = minimize(
                lambda x: np.sqrt(x.T @ self.cov_matrix @ x),
                x0=np.ones(self.n_assets) / self.n_assets,
                method='SLSQP',
                bounds=self.bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return = np.sum(weights * self.returns)
                portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
                results.append({
                    'return': portfolio_return,
                    'volatility': portfolio_vol,
                    'weights': weights
                })
        
        return pd.DataFrame(results)
    
    def _calculate_portfolio_stats(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
        """Calculate portfolio statistics"""
        portfolio_return = np.sum(weights * self.returns)
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / (portfolio_vol + 1e-8)
        
        # Diversification ratio
        weighted_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = weighted_vol / (portfolio_vol + 1e-8)
        
        # Maximum drawdown (estimated)
        max_dd = -0.5 * np.max(weights * np.sqrt(np.diag(self.cov_matrix))) * 2.33
        
        return {
            'weights': weights.tolist(),
            'return': float(portfolio_return),
            'volatility': float(portfolio_vol),
            'sharpe': float(sharpe),
            'diversification_ratio': float(diversification_ratio),
            'max_drawdown': float(max_dd)
        }
    
    def _fallback_optimization(self, method: str, risk_free_rate: float = 0.02) -> Dict:
        """Fallback optimization using differential evolution"""
        
        if method == 'sharpe':
            objective = lambda x: -((np.sum(x * self.returns) - risk_free_rate) / (np.sqrt(x.T @ self.cov_matrix @ x) + 1e-8))
        elif method == 'min_vol':
            objective = lambda x: np.sqrt(x.T @ self.cov_matrix @ x)
        elif method == 'max_return':
            objective = lambda x: -np.sum(x * self.returns)
        else:
            objective = lambda x: np.sum((x - 1/self.n_assets) ** 2)
        
        result = differential_evolution(
            objective,
            bounds=[(0, 1)] * self.n_assets,
            constraints=self.constraints,
            maxiter=1000,
            popsize=15
        )
        
        if result.success:
            weights = result.x
            return self._calculate_portfolio_stats(weights, risk_free_rate)
        else:
            # Equal weight as fallback
            weights = np.ones(self.n_assets) / self.n_assets
            return self._calculate_portfolio_stats(weights, risk_free_rate)
    
    @staticmethod
    def optimize_portfolio(
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        method: str = 'max_sharpe',
        risk_free_rate: float = 0.02,
        **kwargs
    ) -> Dict:
        """
        Static method for quick portfolio optimization
        
        Args:
            returns: Expected returns
            cov_matrix: Covariance matrix
            method: 'max_sharpe', 'min_vol', 'max_return', 'risk_parity', 'max_diversification'
            risk_free_rate: Risk-free rate
            **kwargs: Additional arguments for specific methods
        """
        optimizer = PortfolioOptimizer(returns, cov_matrix)
        
        methods = {
            'max_sharpe': optimizer.optimize_max_sharpe,
            'min_vol': optimizer.optimize_min_volatility,
            'max_return': optimizer.optimize_max_return,
            'risk_parity': optimizer.optimize_risk_parity,
            'max_diversification': optimizer.optimize_max_diversification
        }
        
        if method in methods:
            return methods[method](**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


class FactorModel:
    """
    Factor model for risk decomposition
    """
    
    def __init__(self, returns: np.ndarray, factor_returns: np.ndarray):
        """
        Initialize factor model
        
        Args:
            returns: Asset returns (n_periods, n_assets)
            factor_returns: Factor returns (n_periods, n_factors)
        """
        self.returns = returns
        self.factor_returns = factor_returns
        self.n_assets = returns.shape[1]
        self.n_factors = factor_returns.shape[1]
        self.betas = None
        self.alpha = None
        self.residuals = None
        self.r_squared = None
    
    def fit(self):
        """Fit the factor model"""
        # Calculate betas using OLS
        self.betas = np.zeros((self.n_assets, self.n_factors))
        self.alpha = np.zeros(self.n_assets)
        self.r_squared = np.zeros(self.n_assets)
        self.residuals = np.zeros(self.returns.shape)
        
        for i in range(self.n_assets):
            X = self.factor_returns
            y = self.returns[:, i]
            
            # Add constant
            X_const = np.column_stack([np.ones(len(X)), X])
            
            # OLS
            beta_hat = np.linalg.pinv(X_const.T @ X_const) @ X_const.T @ y
            
            self.alpha[i] = beta_hat[0]
            self.betas[i] = beta_hat[1:]
            
            # Residuals
            y_pred = X_const @ beta_hat
            self.residuals[:, i] = y - y_pred
            
            # R-squared
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum(self.residuals[:, i] ** 2)
            self.r_squared[i] = 1 - ss_residual / (ss_total + 1e-8)
    
    def get_factor_exposures(self) -> Dict:
        """Get factor exposures"""
        return {
            'betas': self.betas,
            'alpha': self.alpha,
            'r_squared': self.r_squared
        }
    
    def calculate_factor_risk(self, weights: np.ndarray) -> Dict:
        """Calculate factor and idiosyncratic risk"""
        if self.betas is None:
            self.fit()
        
        # Factor covariance
        factor_cov = np.cov(self.factor_returns.T)
        
        # Factor risk
        portfolio_betas = weights @ self.betas
        factor_risk = portfolio_betas @ factor_cov @ portfolio_betas
        
        # Idiosyncratic risk
        residual_variance = np.var(self.residuals, axis=0)
        idio_risk = np.sum(weights ** 2 * residual_variance)
        
        # Total risk
        total_risk = factor_risk + idio_risk
        
        return {
            'total_risk': total_risk,
            'factor_risk': factor_risk,
            'idio_risk': idio_risk,
            'factor_risk_pct': factor_risk / (total_risk + 1e-8),
            'idio_risk_pct': idio_risk / (total_risk + 1e-8)
        }