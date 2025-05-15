from flask import render_template, request, jsonify
from app import app
from app.models.classical_models import BlackScholes, MonteCarlo
from app.models.quantum_models import QuantumAmplitudeEstimation
from app.models.advanced_quantum_models import AdvancedQuantumOptionPricing
from app.models.performance_analyzer import PerformanceAnalyzer
from app.models.sensitivity_analyzer import SensitivityAnalyzer
from app.models.quantum_greeks import QuantumGreeks
import numpy as np
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
from app.models.quantum_risk_management import QuantumRiskManagement
from app.models.validation import MarketValidation
import time
from datetime import datetime, timedelta

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

# Add a new route for Indian markets
@app.route('/indian_markets')
def indian_markets():
    """Render the Indian markets page."""
    return render_template('indian_markets.html')

# Fix Plotly colorbar properties error by correcting the create_volatility_surface method
def fix_plotly_colorbar(fig):
    """Fix the Plotly colorbar property issues."""
    if hasattr(fig, 'data') and len(fig.data) > 0:
        for trace in fig.data:
            if hasattr(trace, 'colorbar') and hasattr(trace.colorbar, 'titleside'):
                # Change 'titleside' to 'title' or remove it
                del trace.colorbar.titleside
                # Add proper title if needed
                trace.colorbar.title = trace.colorbar.get('title', 'Value')
    return fig

# Update your existing calculate route to include the new functionality
@app.route('/calculate', methods=['POST'])
def calculate():
    """Calculate option prices using different models."""
    # Get form data
    data = request.form
    
    S = float(data.get('stock_price', 100))
    K = float(data.get('strike_price', 110))
    T = float(data.get('time_to_maturity', 1))
    r = float(data.get('risk_free_rate', 0.05))
    sigma = float(data.get('volatility', 0.2))
    option_type = data.get('option_type', 'call')
    num_qubits = int(data.get('num_qubits', 5))
    market = data.get('market', 'global')  # New parameter for market selection
    ticker = data.get('ticker', '')  # Get ticker if provided
    
    # Calculate Black-Scholes price
    if option_type == 'call':
        bs_price = BlackScholes.call_price(S, K, T, r, sigma)
    else:
        bs_price = BlackScholes.put_price(S, K, T, r, sigma)
    
    # Calculate Monte Carlo price
    mc_result = MonteCarlo.generate_summary(S, K, T, r, sigma)
    mc_price = mc_result['call_price'] if option_type == 'call' else mc_result['put_price']
    
    # Calculate Quantum price (primary attempt)
    try:
        if option_type == 'call':
            # Include resource analysis here:
            quantum_result = QuantumAmplitudeEstimation.simulate_and_visualize(
                S, K, T, r, sigma, num_qubits, include_resource_analysis=True
            )
        else:
            quantum_result = QuantumAmplitudeEstimation.simulate_and_visualize_put(
                S, K, T, r, sigma, num_qubits, include_resource_analysis=True
            )
        
        quantum_price = quantum_result.get('price', 0)
        quantum_error = quantum_result.get('error', None)
        quantum_plot = quantum_result.get('plot_image', None)
        circuit_diagram = quantum_result.get('circuit_diagram', None)
        model_image = None
    except Exception as e:
        # If the primary call fails, try the advanced model as a fallback
        try:
            quantum_result = AdvancedQuantumOptionPricing.simulate_and_visualize_option(
                S, K, T, r, sigma, option_type, num_qubits, include_resource_analysis=True
            )
            quantum_price = quantum_result.get('price', 0)
            quantum_error = quantum_result.get('error', None)
            quantum_plot = quantum_result.get('plot_image', None)
            circuit_diagram = quantum_result.get('circuit_diagram', None)
            model_image = quantum_result.get('model_image', None)
        except Exception as e2:
            quantum_price = 0
            quantum_error = f"Original: {str(e)}, Advanced: {str(e2)}"
            quantum_plot = None
            circuit_diagram = None
            model_image = None

    # Calculate Greeks using the quantum approach if we have a valid quantum price
    greeks_data = None
    greeks_plots = None
    if not quantum_error:
        try:
            # Calculate Greeks with enhanced higher-order Greeks
            greeks_data = QuantumGreeks.calculate_all_greeks(
                S, K, T, r, sigma, option_type, num_qubits
            )
            
            # Create Delta plot
            delta_fig = go.Figure()
            # Use a range of stock prices around the current price (example logic)
            price_range = np.linspace(S * 0.8, S * 1.2, 11)
            delta_values = []
            bs_delta_values = []
            
            # For simplicity, we'll create a single-point delta plot at current S
            delta_values = [greeks_data['delta']]
            bs_delta_values = [greeks_data['bs_greeks']['delta']]
            
            delta_fig.add_trace(go.Scatter(
                x=[S],
                y=delta_values,
                mode='markers',
                name='Quantum Delta',
                marker=dict(size=12)
            ))
            delta_fig.add_trace(go.Scatter(
                x=[S],
                y=bs_delta_values,
                mode='markers',
                name='Black-Scholes Delta',
                marker=dict(size=12, symbol='diamond')
            ))
            delta_fig.update_layout(
                title='Delta: Sensitivity to Underlying Price',
                xaxis_title='Stock Price',
                yaxis_title='Delta',
                legend=dict(x=0, y=1, traceorder='normal')
            )
            
            greeks_plots = {
                'delta': json.dumps(delta_fig, cls=plotly.utils.PlotlyJSONEncoder)
            }
            
        except Exception as e:
            print(f"Greeks calculation failed: {e}")
            greeks_data = None
            greeks_plots = None
            
    # Add performance analysis with hardware-realistic simulation
    performance_data = None
    try:
        # Run performance comparison with different precision levels
        precision_levels = [100, 1000, 5000, 10000]
        performance_data = PerformanceAnalyzer.benchmark_methods(
            S, K, T, r, sigma, option_type, precision_levels,
            BlackScholes, MonteCarlo, QuantumAmplitudeEstimation,
            hardware_realistic=True  # Enable hardware-realistic simulation
        )
    except Exception as e:
        print(f"Performance analysis failed: {e}")
        performance_data = None
        
    # Add sensitivity analysis
    sensitivity_data = None
    try:
        # Create volatility surface
        K_range = np.linspace(K * 0.8, K * 1.2, 10)
        T_range = np.linspace(max(0.1, T * 0.5), T * 1.5, 10)
        volatility_surface = SensitivityAnalyzer.create_volatility_surface(
            S, K_range, T_range, r, sigma, option_type, BlackScholes
        )
        
        # Fix Plotly titleside issue in volatility surface
        if 'figure' in volatility_surface:
            volatility_surface['figure'] = fix_plotly_colorbar(volatility_surface['figure'])
        
        # Create Greeks heatmap
        S_range = np.linspace(S * 0.8, S * 1.2, 10)
        sigma_range = np.linspace(max(0.05, sigma * 0.5), sigma * 1.5, 10)
        greeks_heatmap = SensitivityAnalyzer.create_greeks_heatmap(
            S_range, sigma_range, K, T, r, option_type, QuantumGreeks
        )
        
        # Fix Plotly titleside issue in Greeks heatmap
        if 'figure' in greeks_heatmap:
            greeks_heatmap['figure'] = fix_plotly_colorbar(greeks_heatmap['figure'])
        
        sensitivity_data = {
            'volatility_surface': volatility_surface,
            'greeks_heatmap': greeks_heatmap
        }
    except Exception as e:
        print(f"Sensitivity analysis failed: {e}")
        sensitivity_data = None
    
    # Create MC simulation path plot
    fig = go.Figure()
    for i in range(mc_result['sample_paths'].shape[0]):
        fig.add_trace(go.Scatter(
            x=mc_result['times'],
            y=mc_result['sample_paths'][i],
            mode='lines',
            name=f'Path {i+1}'
        ))
    fig.add_trace(go.Scatter(
        x=[0, T],
        y=[K, K],  # Horizontal line at strike price
        mode='lines',
        name='Strike Price',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title='Monte Carlo Simulation Paths',
        xaxis_title='Time (years)',
        yaxis_title='Stock Price',
        legend_title='Paths'
    )
    mc_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Add volatility smile plot if advanced visualization is requested
    try:
        smile_fig = go.Figure()
        # Generate implied volatility smile
        strikes = np.linspace(0.8 * S, 1.2 * S, 9)
        implied_vols = []
        
        for strike in strikes:
            # Calculate skew effect - higher implied vol for lower strikes (put skew)
            skew_factor = np.exp(-0.5 * ((strike / S - 1) / 0.1)**2)
            implied_vol = sigma * (1 + 0.2 * skew_factor)
            implied_vols.append(implied_vol)
        
        smile_fig.add_trace(go.Scatter(
            x=strikes,
            y=implied_vols,
            mode='lines+markers',
            name='Implied Volatility'
        ))
        smile_fig.add_trace(go.Scatter(
            x=[S, S],
            y=[0, max(implied_vols) * 1.1],
            mode='lines',
            name='Current Stock Price',
            line=dict(color='green', dash='dash')
        ))
        smile_fig.update_layout(
            title='Implied Volatility Smile',
            xaxis_title='Strike Price',
            yaxis_title='Implied Volatility',
            legend=dict(x=0, y=1, traceorder='normal')
        )
        smile_plot = json.dumps(smile_fig, cls=plotly.utils.PlotlyJSONEncoder)
    except:
        smile_plot = None
        
    # NEW: Add theoretical speedup analysis
    try:
        speedup_analysis = QuantumAmplitudeEstimation.theoretical_speedup_analysis(0.01)  # 1% precision
    except Exception as e:
        print(f"Speedup analysis failed: {e}")
        speedup_analysis = None
        
    # NEW: Add risk management analysis
    var_analysis = None
    es_analysis = None
    try:
        # Generate sample returns for demonstration
        returns = np.random.normal(-0.001, 0.02, 252)  # 1 year of daily returns
        var_analysis = QuantumRiskManagement.calculate_var(returns)
        es_analysis = QuantumRiskManagement.calculate_expected_shortfall(returns)
    except Exception as e:
        print(f"Risk management analysis failed: {e}")
        
    # NEW: Add market validation with support for Indian markets
    market_validation = None
    try:
        if ticker:
            # Use the provided ticker for market validation
            # For Indian markets, adjust ticker selection
            if market == 'india':
                if not any(ind_index in ticker.lower() for ind_index in ['nifty', 'banknifty', 'finnifty']):
                    # Add .NS suffix if not already present for Indian stocks
                    if not ticker.endswith('.NS'):
                        ticker = f"{ticker}.NS"
                        
                market_validation = MarketValidation.validate_against_market(
                    tickers=[ticker], 
                    num_options=2,
                    market='india'
                )
            else:
                market_validation = MarketValidation.validate_against_market(
                    tickers=[ticker], 
                    num_options=2,
                    market='global'
                )
        else:
            # Default tickers based on market
            if market == 'india':
                market_validation = MarketValidation.validate_against_market(
                    tickers=['NIFTY', 'BANKNIFTY'], 
                    num_options=2,
                    market='india'
                )
            else:
                market_validation = MarketValidation.validate_against_market(
                    tickers=['AAPL', 'MSFT'], 
                    num_options=2,
                    market='global'
                )
    except Exception as e:
        print(f"Market validation failed: {e}")
        market_validation = None
    
    # Return the results to the template, ensuring we pass quantum_result
    return render_template(
        'results.html',
        bs_price=bs_price,
        mc_price=mc_price,
        quantum_price=quantum_price,
        quantum_error=quantum_error,
        quantum_plot=quantum_plot,
        circuit_diagram=circuit_diagram,
        model_image=model_image,
        mc_plot=mc_plot,
        smile_plot=smile_plot,
        greeks=greeks_data,
        greeks_plots=greeks_plots,
        performance_data=performance_data,
        sensitivity_data=sensitivity_data,
        quantum_result=quantum_result,  # Pass the quantum results
        # NEW: Pass the enhanced features
        speedup_analysis=speedup_analysis,
        var_analysis=var_analysis,
        es_analysis=es_analysis,
        market_validation=market_validation,
        # Extract higher-order Greeks if available
        higher_order_greeks={
            'speed': greeks_data.get('speed', None) if greeks_data else None,
            'zomma': greeks_data.get('zomma', None) if greeks_data else None,
            'color': greeks_data.get('color', None) if greeks_data else None,
            'vanna': greeks_data.get('vanna', None) if greeks_data else None, 
            'ultima': greeks_data.get('ultima', None) if greeks_data else None
        },
        uncertainty_metrics=greeks_data.get('uncertainty_metric', {}) if greeks_data else {},
        params={
            'S': S,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'option_type': option_type,
            'num_qubits': num_qubits,
            'market': market,
            'ticker': ticker
        }
    )

@app.route('/api/calculate', methods=['POST'])
def api_calculate():
    """API endpoint for calculating option prices."""
    data = request.json
    
    S = float(data.get('stock_price', 100))
    K = float(data.get('strike_price', 110))
    T = float(data.get('time_to_maturity', 1))
    r = float(data.get('risk_free_rate', 0.05))
    sigma = float(data.get('volatility', 0.2))
    option_type = data.get('option_type', 'call')
    num_qubits = int(data.get('num_qubits', 5))
    market = data.get('market', 'global')
    ticker = data.get('ticker', '')
    
    # Calculate Black-Scholes price
    if option_type == 'call':
        bs_price = BlackScholes.call_price(S, K, T, r, sigma)
    else:
        bs_price = BlackScholes.put_price(S, K, T, r, sigma)
    
    # Calculate Monte Carlo price
    mc_result = MonteCarlo.generate_summary(S, K, T, r, sigma)
    mc_price = mc_result['call_price'] if option_type == 'call' else mc_result['put_price']
    
    # Calculate Quantum price using the original implementation
    try:
        if option_type == 'call':
            quantum_result = QuantumAmplitudeEstimation.european_call_price(S, K, T, r, sigma, num_qubits)
        else:
            quantum_result = QuantumAmplitudeEstimation.european_put_price(S, K, T, r, sigma, num_qubits)
            
        quantum_price = quantum_result.get('price', 0)
        quantum_error = quantum_result.get('error', None)
        confidence_interval = 0  # Original implementation doesn't provide this
    except Exception as e:
        # Try advanced model as fallback
        try:
            quantum_result = AdvancedQuantumOptionPricing.european_option_price(
                S, K, T, r, sigma, option_type, num_qubits
            )
            quantum_price = quantum_result.get('price', 0)
            quantum_error = quantum_result.get('error', None)
            confidence_interval = quantum_result.get('confidence_interval', 0)
        except Exception as e2:
            quantum_price = 0
            quantum_error = f"Original: {str(e)}, Advanced: {str(e2)}"
            confidence_interval = 0
    
    # Get market data if provided ticker and market
    market_data = None
    if ticker:
        try:
            # Fetch live market data for the ticker
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            stock_data = MarketValidation.fetch_market_data(ticker, start_date, end_date, market=market)
            
            if not stock_data.empty:
                # Get current market price
                current_price = float(stock_data['Close'].iloc[-1])
                
                # Try to get option chain data
                expiry_dates, option_chains = MarketValidation.fetch_option_chain(ticker, market=market)
                
                if expiry_dates and option_chains:
                    # For demo purposes, use the first expiry date
                    expiry = expiry_dates[0]
                    chain = option_chains[expiry]
                    
                    # Extract relevant options (closest to our K)
                    options = chain.calls if option_type.lower() == 'call' else chain.puts
                    if not options.empty:
                        # Find option closest to our strike price
                        strikes_diff = abs(options['strike'] - K)
                        closest_idx = strikes_diff.idxmin()
                        closest_option = options.loc[closest_idx]
                        
                        market_data = {
                            'current_price': current_price,
                            'option_price': float(closest_option['lastPrice']) if 'lastPrice' in closest_option else None,
                            'implied_vol': float(closest_option['impliedVolatility']) if 'impliedVolatility' in closest_option else None,
                            'expiry_date': expiry,
                            'strike': float(closest_option['strike']) if 'strike' in closest_option else K
                        }
        except Exception as e:
            print(f"Error fetching market data: {e}")
    
    response = {
        'black_scholes': float(bs_price),
        'monte_carlo': float(mc_price),
        'quantum': float(quantum_price),
        'quantum_error': quantum_error,
        'confidence_interval': confidence_interval
    }
    
    if market_data:
        response['market_data'] = market_data
    
    return jsonify(response)

# Add a new route for fetching Indian market data
@app.route('/api/indian_options', methods=['GET'])
def get_indian_options():
    """API endpoint for getting Indian option chains."""
    symbol = request.args.get('symbol', 'NIFTY')
    
    try:
        # Get option chain data
        expiry_dates, option_chains = MarketValidation.fetch_option_chain(symbol, market='india')
        
        if not expiry_dates:
            return jsonify({"error": f"No options data available for {symbol}"}), 404
        
        # For simplicity, return only the first expiry date
        expiry = expiry_dates[0]
        chain = option_chains[expiry]
        
        # Get current stock/index price
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        stock_data = MarketValidation.fetch_market_data(symbol, start_date, end_date, market='india')
        
        current_price = float(stock_data['Close'].iloc[-1]) if not stock_data.empty else 0
        
        # Format data for response
        calls = []
        puts = []
        
        if hasattr(chain, 'calls') and not chain.calls.empty:
            for idx, option in chain.calls.iterrows():
                calls.append({
                    'strike': float(option['strike']),
                    'price': float(option['lastPrice']) if 'lastPrice' in option else 0,
                    'iv': float(option['impliedVolatility']) if 'impliedVolatility' in option else 0,
                    'volume': int(option['volume']) if 'volume' in option else 0,
                    'openInterest': int(option['openInterest']) if 'openInterest' in option else 0,
                })
        
        if hasattr(chain, 'puts') and not chain.puts.empty:
            for idx, option in chain.puts.iterrows():
                puts.append({
                    'strike': float(option['strike']),
                    'price': float(option['lastPrice']) if 'lastPrice' in option else 0,
                    'iv': float(option['impliedVolatility']) if 'impliedVolatility' in option else 0,
                    'volume': int(option['volume']) if 'volume' in option else 0,
                    'openInterest': int(option['openInterest']) if 'openInterest' in option else 0,
                })
        
        return jsonify({
            'symbol': symbol,
            'current_price': current_price,
            'expiry_date': expiry,
            'calls': calls,
            'puts': puts
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a new route for comparing models against market prices
@app.route('/compare_market', methods=['POST'])
def compare_market():
    """Compare option pricing models against real market prices."""
    data = request.form
    
    ticker = data.get('ticker', 'NIFTY')
    market = data.get('market', 'india')
    num_options = int(data.get('num_options', 5))
    
    try:
        # Validate models against market data
        results = MarketValidation.validate_against_market(
            tickers=[ticker],
            num_options=num_options,
            market=market
        )
        
        if results.empty:
            return jsonify({"error": "No market data available for comparison"}), 404
        
        # Format results for display
        formatted_results = []
        for _, row in results.iterrows():
            formatted_results.append({
                'ticker': row['ticker'],
                'expiry': row['expiry'],
                'strike': float(row['strike']),
                'type': row['type'],
                'market_price': float(row['market_price']),
                'stock_price': float(row['stock_price']) if 'stock_price' in row else 0,
                'time_to_expiry': float(row['time_to_expiry']) if 'time_to_expiry' in row else 0,
                'implied_volatility': float(row['implied_volatility']) if 'implied_volatility' in row else 0,
                'bs_price': float(row['bs_price']) if 'bs_price' in row else 0,
                'mc_price': float(row['mc_price']) if 'mc_price' in row else 0,
                'quantum_price': float(row['quantum_price']) if 'quantum_price' in row else 0,
                'advanced_quantum_price': float(row['advanced_quantum_price']) if 'advanced_quantum_price' in row else 0,
                'bs_error': float(row['bs_error']) if 'bs_error' in row else 0,
                'mc_error': float(row['mc_error']) if 'mc_error' in row else 0,
                'quantum_error': float(row['quantum_error']) if 'quantum_error' in row else 0,
                'advanced_quantum_error': float(row['advanced_quantum_error']) if 'advanced_quantum_error' in row else 0
            })
        
        return jsonify({
            'results': formatted_results,
            'ticker': ticker,
            'market': market
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500