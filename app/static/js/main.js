document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip);
    });
    
    // Form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Basic validation
            const stockPrice = parseFloat(document.getElementById('stock_price').value);
            const strikePrice = parseFloat(document.getElementById('strike_price').value);
            const timeToMaturity = parseFloat(document.getElementById('time_to_maturity').value);
            const riskFreeRate = parseFloat(document.getElementById('risk_free_rate').value);
            const volatility = parseFloat(document.getElementById('volatility').value);
            
            let isValid = true;
            let errorMessage = '';
            
            if (stockPrice <= 0) {
                errorMessage = 'Stock price must be positive';
                isValid = false;
            } else if (strikePrice <= 0) {
                errorMessage = 'Strike price must be positive';
                isValid = false;
            } else if (timeToMaturity <= 0) {
                errorMessage = 'Time to maturity must be positive';
                isValid = false;
            } else if (riskFreeRate < 0 || riskFreeRate > 1) {
                errorMessage = 'Risk-free rate must be between 0 and 1';
                isValid = false;
            } else if (volatility <= 0 || volatility > 2) {
                errorMessage = 'Volatility must be positive and not exceed 2';
                isValid = false;
            }
            
            if (!isValid) {
                e.preventDefault();
                alert('Validation Error: ' + errorMessage);
            } else {
                // Show loading indicator
                const submitBtn = document.querySelector('button[type="submit"]');
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Calculating...';
                submitBtn.disabled = true;
            }
        });
    }
    
    // Responsive charts
    function resizeCharts() {
        // Resize all Plotly charts
        const plotlyCharts = ['mc-plot', 'greeks-heatmap', 'volatility-surface', 'smile-plot', 'delta-chart'];
        
        plotlyCharts.forEach(chartId => {
            const chartElem = document.getElementById(chartId);
            if (chartElem && window.Plotly) {
                try {
                    Plotly.Plots.resize(chartElem);
                } catch (error) {
                    console.warn(`Failed to resize ${chartId}:`, error);
                }
            }
        });
    }
    
    // Execute on resize with debounce
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(resizeCharts, 250);
    });
    
    // Initialize tab events for better plot rendering
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            // When a tab is shown, trigger resize to properly render plots
            setTimeout(resizeCharts, 50);
        });
    });
    
    // Special handling for Greeks heatmap
    const heatmapTab = document.getElementById('heatmap-tab');
    if (heatmapTab) {
        heatmapTab.addEventListener('shown.bs.tab', function() {
            const greeksHeatmap = document.getElementById('greeks-heatmap');
            
            if (greeksHeatmap && window.greeksHeatmapData) {
                // Wait a moment for the tab to fully display
                setTimeout(function() {
                    try {
                        // Update layout for better display
                        if (window.greeksHeatmapData.layout) {
                            // Adjust layout properties for better appearance
                            const updatedLayout = Object.assign({}, window.greeksHeatmapData.layout, {
                                height: 800,
                                font: { 
                                    family: 'Arial, sans-serif',
                                    size: 12 
                                },
                                margin: { 
                                    l: 80, 
                                    r: 80, 
                                    t: 80, 
                                    b: 80 
                                }
                            });
                            
                            // Create a new plot with improved layout
                            Plotly.react('greeks-heatmap', window.greeksHeatmapData.data, updatedLayout, {responsive: true});
                        }
                    } catch (e) {
                        console.error("Error enhancing Greeks heatmap:", e);
                    }
                }, 100);
            }
        });
    }
    
    // Store Greeks heatmap data in a global variable
    if (typeof greeksHeatmapData !== 'undefined') {
        window.greeksHeatmapData = greeksHeatmapData;
    }
    
    // Execute on initial load
    setTimeout(resizeCharts, 100);
    
    // Fix footer positioning
    function adjustFooter() {
        const body = document.body;
        const html = document.documentElement;
        const documentHeight = Math.max(
            body.scrollHeight, body.offsetHeight,
            html.clientHeight, html.scrollHeight, html.offsetHeight
        );
        const viewportHeight = window.innerHeight;
        
        const footer = document.querySelector('footer');
        if (footer) {
            if (documentHeight <= viewportHeight) {
                footer.style.position = 'fixed';
                footer.style.bottom = '0';
                footer.style.width = '100%';
            } else {
                footer.style.position = 'static';
            }
        }
    }
    
    // Call on load and resize
    adjustFooter();
    window.addEventListener('resize', adjustFooter);
});

// Add a function to initialize sensitivity plots
function initializeSensitivityPlots() {
    if (window.sensitivityData) {
        try {
            // For volatility surface
            if (window.sensitivityData.volatility_surface) {
                const volSurfaceData = JSON.parse(window.sensitivityData.volatility_surface);
                Plotly.newPlot('volatility-surface', volSurfaceData.data, volSurfaceData.layout, {responsive: true});
            }
            
            // For Greeks heatmap
            if (window.sensitivityData.greeks_heatmap) {
                const greeksHeatmapData = JSON.parse(window.sensitivityData.greeks_heatmap);
                
                // Store data for tab switch enhancement
                window.greeksHeatmapData = greeksHeatmapData;
                
                // Enhance layout
                if (greeksHeatmapData.layout) {
                    greeksHeatmapData.layout.height = 800;
                    greeksHeatmapData.layout.width = null; // Let it be responsive
                    
                    // Improve font and margins
                    greeksHeatmapData.layout.font = {
                        family: 'Arial, sans-serif',
                        size: 12
                    };
                    
                    greeksHeatmapData.layout.margin = {
                        l: 80,
                        r: 80,
                        t: 80,
                        b: 80
                    };
                    
                    // Improve subplot titles
                    if (greeksHeatmapData.layout.annotations) {
                        greeksHeatmapData.layout.annotations.forEach(ann => {
                            ann.font = {
                                size: 14,
                                weight: 'bold'
                            };
                        });
                    }
                }
                
                // Enhance data
                if (greeksHeatmapData.data) {
                    greeksHeatmapData.data.forEach(trace => {
                        if (trace.colorbar) {
                            trace.colorbar.len = 0.8;
                            trace.colorbar.thickness = 15;
                            trace.colorbar.title.font = {
                                size: 12
                            };
                        }
                    });
                }
                
                Plotly.newPlot('greeks-heatmap', greeksHeatmapData.data, greeksHeatmapData.layout, {responsive: true});
            }
        } catch (e) {
            console.error("Error initializing sensitivity plots:", e);
        }
    }
}

// Call this after page load if sensitivity data is present
window.addEventListener('load', function() {
    setTimeout(initializeSensitivityPlots, 500);
});