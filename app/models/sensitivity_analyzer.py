import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Add this JSON encoder class to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class SensitivityAnalyzer:
    """
    Creates advanced visualizations showing how option prices and Greeks
    change with varying parameters.
    """
    
    @staticmethod
    def create_volatility_surface(S, K_range, T_range, r, sigma, option_type, pricing_method):
        """
        Create a 3D volatility surface showing how option prices change
        with different strikes and maturities.
        
        Args:
            S: Stock price
            K_range: Range of strike prices
            T_range: Range of maturities
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            pricing_method: Function to calculate option price
            
        Returns:
            JSON string for Plotly 3D surface
        """
        # Create meshgrid for strike and maturity
        K_mesh, T_mesh = np.meshgrid(K_range, T_range)
        
        # Calculate option prices for each combination
        Z = np.zeros_like(K_mesh)
        for i in range(len(T_range)):
            for j in range(len(K_range)):
                K = K_range[j]
                T = T_range[i]
                
                # Calculate option price
                try:
                    if option_type == 'call':
                        price = pricing_method.call_price(S, K, T, r, sigma)
                    else:
                        price = pricing_method.put_price(S, K, T, r, sigma)
                    Z[i, j] = price
                except Exception as e:
                    print(f"Error calculating price for K={K}, T={T}: {e}")
                    Z[i, j] = np.nan
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=K_mesh, 
            y=T_mesh, 
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title="Option Price ($)",
                titleside="right"
            )
        )])
        
        fig.update_layout(
            title=f"{option_type.capitalize()} Option Price Surface",
            scene=dict(
                xaxis_title="Strike Price ($)",
                yaxis_title="Time to Maturity (years)",
                zaxis_title="Option Price ($)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
            ),
            width=800,
            height=600,
            margin=dict(l=30, r=30, b=30, t=50)
        )
        
        # Add reference point for current parameters
        current_K_idx = np.argmin(np.abs(np.array(K_range) - K_range[len(K_range)//2]))
        current_T_idx = np.argmin(np.abs(np.array(T_range) - T_range[len(T_range)//2]))
        
        fig.add_trace(go.Scatter3d(
            x=[K_range[current_K_idx]],
            y=[T_range[current_T_idx]],
            z=[Z[current_T_idx, current_K_idx]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle'
            ),
            name="Reference Point"
        ))
        
        # Convert to JSON using NumpyEncoder to handle NumPy types
        return json.dumps(fig.to_dict(), cls=NumpyEncoder)
    
    @staticmethod
    def create_greeks_heatmap(S_range, sigma_range, K, T, r, option_type, greeks_calculator):
        """
        Create a heatmap showing how Delta and Gamma change with different
        stock prices and volatilities.
        
        Args:
            S_range: Range of stock prices
            sigma_range: Range of volatilities
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            greeks_calculator: Function to calculate Greeks
            
        Returns:
            JSON string for Plotly subplots with heatmaps
        """
        # Initialize arrays for Greeks
        S_mesh, sigma_mesh = np.meshgrid(S_range, sigma_range)
        delta_values = np.zeros_like(S_mesh)
        gamma_values = np.zeros_like(S_mesh)
        vega_values = np.zeros_like(S_mesh)
        theta_values = np.zeros_like(S_mesh)
        
        # Calculate Greeks for each combination
        for i in range(len(sigma_range)):
            for j in range(len(S_range)):
                S = S_range[j]
                sigma = sigma_range[i]
                
                try:
                    # Calculate Greeks
                    greeks = greeks_calculator.black_scholes_greeks(S, K, T, r, sigma, option_type)
                    
                    # Store values
                    delta_values[i, j] = greeks['delta']
                    gamma_values[i, j] = greeks['gamma']
                    vega_values[i, j] = greeks['vega_percent']
                    theta_values[i, j] = greeks['theta_daily']
                except Exception as e:
                    print(f"Error calculating Greeks for S={S}, sigma={sigma}: {e}")
                    delta_values[i, j] = np.nan
                    gamma_values[i, j] = np.nan
                    vega_values[i, j] = np.nan
                    theta_values[i, j] = np.nan
        
        # Create subplots with more spacing and better layout
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=("Delta", "Gamma", "Vega (1% move)", "Theta (daily)"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]],
            horizontal_spacing=0.15,  # Add more spacing between columns
            vertical_spacing=0.15     # Add more spacing between rows
        )
        
        # Add heatmaps with improved colorbar settings
        fig.add_trace(
            go.Heatmap(
                z=delta_values,
                x=S_range,
                y=sigma_range,
                colorscale='RdBu',
                colorbar=dict(
                    title="Delta", 
                    x=0.46, 
                    y=0.8,
                    len=0.4,        # Make it shorter
                    thickness=15,   # Make it thinner
                    title_side="right",
                    title_font=dict(size=12)
                ),
                zmin=-1 if option_type == 'put' else 0,
                zmax=0 if option_type == 'put' else 1,
                hovertemplate="Stock: $%{x}<br>Vol: %{y}<br>Delta: %{z:.4f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=gamma_values,
                x=S_range,
                y=sigma_range,
                colorscale='Viridis',
                colorbar=dict(
                    title="Gamma", 
                    x=0.96, 
                    y=0.8,
                    len=0.4,
                    thickness=15,
                    title_side="right",
                    title_font=dict(size=12)
                ),
                hovertemplate="Stock: $%{x}<br>Vol: %{y}<br>Gamma: %{z:.4f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=vega_values,
                x=S_range,
                y=sigma_range,
                colorscale='Plasma',
                colorbar=dict(
                    title="Vega", 
                    x=0.46, 
                    y=0.2,
                    len=0.4,
                    thickness=15,
                    title_side="right",
                    title_font=dict(size=12)
                ),
                hovertemplate="Stock: $%{x}<br>Vol: %{y}<br>Vega: %{z:.4f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=theta_values,
                x=S_range,
                y=sigma_range,
                colorscale='Cividis',
                colorbar=dict(
                    title="Theta", 
                    x=0.96, 
                    y=0.2,
                    len=0.4,
                    thickness=15,
                    title_side="right",
                    title_font=dict(size=12)
                ),
                hovertemplate="Stock: $%{x}<br>Vol: %{y}<br>Theta: %{z:.4f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Add marker for current parameters with clearer styling
        current_S_idx = np.argmin(np.abs(np.array(S_range) - S_range[len(S_range)//2]))
        current_sigma_idx = np.argmin(np.abs(np.array(sigma_range) - sigma_range[len(sigma_range)//2]))
        
        for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_trace(
                go.Scatter(
                    x=[S_range[current_S_idx]],
                    y=[sigma_range[current_sigma_idx]],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='red',
                        symbol='cross',
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False,
                    hovertemplate="Current Parameters<extra></extra>"
                ),
                row=i, col=j
            )
        
        # Update layout with better spacing and axis labels
        fig.update_layout(
            title=f"Greeks Sensitivity for {option_type.capitalize()} Option (K=${K}, T={T}yr)",
            width=950,    # Fixed width to prevent responsiveness issues
            height=850,   # Taller to accommodate spacing
            margin=dict(l=120, r=120, t=100, b=100),  # Add more margin
            font=dict(family="Arial, sans-serif", size=12),
        )
        
        # Update x-axis titles with better positioning
        fig.update_xaxes(title="Stock Price ($)", title_standoff=20, row=1, col=1)
        fig.update_xaxes(title="Stock Price ($)", title_standoff=20, row=1, col=2)
        fig.update_xaxes(title="Stock Price ($)", title_standoff=20, row=2, col=1)
        fig.update_xaxes(title="Stock Price ($)", title_standoff=20, row=2, col=2)
        
        # Update y-axis titles with better positioning
        fig.update_yaxes(title="Volatility (%)", title_standoff=20, row=1, col=1)
        fig.update_yaxes(title="Volatility (%)", title_standoff=20, row=1, col=2)
        fig.update_yaxes(title="Volatility (%)", title_standoff=20, row=2, col=1)
        fig.update_yaxes(title="Volatility (%)", title_standoff=20, row=2, col=2)
        
        # Convert to JSON using NumpyEncoder to handle NumPy types
        return json.dumps(fig.to_dict(), cls=NumpyEncoder)