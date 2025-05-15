import numpy as np
from scipy.stats import norm

class BlackScholes:
    """Black-Scholes option pricing model."""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter for Black-Scholes formula."""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter for Black-Scholes formula."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate call option price using Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return max(0, S - K)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate put option price using Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return max(0, K - S)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class MonteCarlo:
    """Monte Carlo simulation for option pricing."""
    
    @staticmethod
    def generate_paths(S, T, r, sigma, num_paths, steps):
        """Generate price paths using geometric Brownian motion."""
        dt = T / steps
        paths = np.zeros((num_paths, steps + 1))
        paths[:, 0] = S
        
        for i in range(1, steps + 1):
            z = np.random.standard_normal(num_paths)
            paths[:, i] = paths[:, i-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            
        return paths
    
    @staticmethod
    def call_price(S, K, T, r, sigma, num_paths=10000, steps=252):
        """Calculate call option price using Monte Carlo simulation."""
        paths = MonteCarlo.generate_paths(S, T, r, sigma, num_paths, steps)
        payoffs = np.maximum(paths[:, -1] - K, 0)
        return np.exp(-r * T) * np.mean(payoffs)
    
    @staticmethod
    def put_price(S, K, T, r, sigma, num_paths=10000, steps=252):
        """Calculate put option price using Monte Carlo simulation."""
        paths = MonteCarlo.generate_paths(S, T, r, sigma, num_paths, steps)
        payoffs = np.maximum(K - paths[:, -1], 0)
        return np.exp(-r * T) * np.mean(payoffs)
    
    @staticmethod
    def generate_summary(S, K, T, r, sigma, num_paths=10000, steps=252):
        """Generate a summary of Monte Carlo simulation results."""
        paths = MonteCarlo.generate_paths(S, T, r, sigma, num_paths, steps)
        call_payoffs = np.maximum(paths[:, -1] - K, 0)
        put_payoffs = np.maximum(K - paths[:, -1], 0)
        
        call_price = np.exp(-r * T) * np.mean(call_payoffs)
        put_price = np.exp(-r * T) * np.mean(put_payoffs)
        
        call_std_err = np.std(call_payoffs) / np.sqrt(num_paths)
        put_std_err = np.std(put_payoffs) / np.sqrt(num_paths)
        
        # Sample paths for visualization
        sample_indices = np.random.choice(num_paths, min(10, num_paths), replace=False)
        sample_paths = paths[sample_indices]
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'call_std_err': call_std_err,
            'put_std_err': put_std_err,
            'sample_paths': sample_paths,
            'times': np.linspace(0, T, steps + 1)
        }