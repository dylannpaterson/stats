import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import webbrowser

# --- CATPPUCCIN MOCHA PALETTE ---
CATPPUCCIN_MOCHA = {
    'base': '#1e1e2e',       # Background
    'mantle': '#181825',     # Secondary Background/UI elements
    'text': '#cdd6f4',       # Primary Text
    'blue': '#89b4fa',       # Primary Trace (Chain Trace)
    'lavender': '#b4befe',   # Secondary Trace (ECDF)
    'green': '#a6e3a1',      # MAP Estimate
    'mauve': '#cba6f7',      # Median/Table Header
    'red': '#f38ba8',        # Observed Value
    'surface0': '#313244',   # Table Cell/Grid line
    'colorway': ['#89b4fa', '#b4befe', '#a6e3a1', '#f38ba8', '#fab387', '#f9e2af']
}

# --- CONFIGURATION ---
OUTPUT_DIR = "docs"
OUTPUT_FILENAME = "index.html"
PLOT_SUBDIR = "_plots"

def _create_step_data(x, y):
    """Helper to create step-plot coordinates."""
    x_step = np.repeat(x, 2)[1:]
    y_step = np.repeat(y, 2)[:-1]
    return x_step, y_step

# --- THIS FUNCTION IS NOW THE ONLY SOURCE OF TRUTH FOR TITLES ---
def generate_report_figures(category_code, sa2_code, data, labels_map):
    """
    Generates BOTH Plotly figures, ensuring their titles are correct and identical.
    This function is now self-contained and robust.
    """
    # Look up the human-readable names INSIDE this function.
    category_label = labels_map['category'].get(category_code, category_code)
    sa2_label = labels_map['sa2'].get(sa2_code, sa2_code)

    # Define the single, correct title text. This is the only title we will use.
    clean_title = f"SA2: <b>{sa2_label}</b><br>Category: {category_label}"

    # Extract data for plotting
    summary_stats = data['summary_stats']
    trace_data = data['trace']
    ecdf_x = np.array(data['ecdf_x'])
    ecdf_y = np.array(data['ecdf_y'])
    obs_value = summary_stats['Observed Value']
    median_val = summary_stats['Posterior Median']
    map_val = summary_stats['MAP Estimate']
    
    # --- Figure 1: The Table (MOCHA STYLED) ---
    # This figure will now ONLY be used for its data table. The title will be suppressed.
    table_fig = go.Figure(data=[go.Table(
        # Apply Mocha colors to header and cells
        header=dict(values=['Statistic', 'Value'], fill_color=CATPPUCCIN_MOCHA['mauve'], align='left', font=dict(size=14, color=CATPPUCCIN_MOCHA['base'])),
        cells=dict(values=[list(summary_stats.keys()), list(summary_stats.values())], fill_color=CATPPUCCIN_MOCHA['surface0'], align='left', font=dict(size=12, color=CATPPUCCIN_MOCHA['text']), height=30)
    )])
    # Apply Mocha theme to the table figure layout
    table_fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=CATPPUCCIN_MOCHA['base'],
        paper_bgcolor=CATPPUCCIN_MOCHA['base'],
        font=dict(color=CATPPUCCIN_MOCHA['text'])
    )

    # --- Figure 2: The Plots (MOCHA STYLED) ---
    # We create the subplots WITHOUT any titles.
    plots_fig = make_subplots(rows=1, cols=2)

    # Add traces with new Mocha colors
    plots_fig.add_trace(go.Scatter(y=trace_data, mode='lines', name='Chain Trace', line=dict(color=CATPPUCCIN_MOCHA['blue'])), row=1, col=1)

    if len(ecdf_x) > 1:
        x_step, y_step = _create_step_data(ecdf_x, ecdf_y)
        plots_fig.add_trace(go.Scatter(x=x_step, y=y_step, mode='lines', name='ECDF', line=dict(color=CATPPUCCIN_MOCHA['lavender'])), row=1, col=2)
    else:
        plots_fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode='markers', name='ECDF (Single Point)', marker=dict(color=CATPPUCCIN_MOCHA['lavender'], size=15, symbol='diamond')), row=1, col=2)

    # Add Horizontal lines with new Mocha colors
    plots_fig.add_hline(y=median_val, line_dash="dash", line_color=CATPPUCCIN_MOCHA['mauve'], name="Median", legendgroup="median", showlegend=True, row=1, col=1)
    plots_fig.add_hline(y=map_val, line_dash="dot", line_color=CATPPUCCIN_MOCHA['green'], name="MAP", legendgroup="map", showlegend=True, row=1, col=1)
    plots_fig.add_trace(go.Scatter(x=[median_val, median_val], y=[0, 1], mode='lines', line_dash="dash", line_color=CATPPUCCIN_MOCHA['mauve'], name="Median", legendgroup="median", showlegend=False), row=1, col=2)
    plots_fig.add_trace(go.Scatter(x=[map_val, map_val], y=[0, 1], mode='lines', line_dash="dot", line_color=CATPPUCCIN_MOCHA['green'], name="MAP", legendgroup="map", showlegend=False), row=1, col=2)
    
    if obs_value != "Suppressed":
        plots_fig.add_trace(go.Scatter(x=[obs_value, obs_value], y=[0, 1], mode='lines', line_dash="solid", line_color=CATPPUCCIN_MOCHA['red'], name="Observed", legendgroup="observed", showlegend=True), row=1, col=2)
        obs_prob_index = np.searchsorted(ecdf_x, obs_value, side='right')
        obs_prob = ecdf_y[obs_prob_index - 1] if obs_prob_index > 0 else 0
        plots_fig.add_trace(go.Scatter(x=[obs_value], y=[obs_prob], mode='markers', marker=dict(color=CATPPUCCIN_MOCHA['red'], size=12, symbol='star'), name='Observed Value', legendgroup="observed", showlegend=False), row=1, col=2)

    # --- APPLY MOCHA THEME TO LAYOUT ---
    plots_fig.update_layout(
        title_text=clean_title, # Set the main, overarching title for the plots.
        title_font_size=18,
        height=450, 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        # MOCHA THEME APPLICATION
        plot_bgcolor=CATPPUCCIN_MOCHA['base'],
        paper_bgcolor=CATPPUCCIN_MOCHA['base'],
        font=dict(color=CATPPUCCIN_MOCHA['text']),
        colorway=CATPPUCCIN_MOCHA['colorway'],
        # Manually add the subplot titles as annotations, which cannot be overridden.
        annotations=[
            dict(text="Down-Sampled Trace Plot", x=0.225, xref="paper", y=1.0, yref="paper", showarrow=False, font=dict(size=16)),
            dict(text="Empirical CDF", x=0.775, xref="paper", y=1.0, yref="paper", showarrow=False, font=dict(size=16))
        ]
    )
    # Update axes titles and grid lines
    plots_fig.update_yaxes(title_text="Estimated Count", row=1, col=1, gridcolor=CATPPUCCIN_MOCHA['surface0'])
    plots_fig.update_xaxes(title_text="Iteration (Down-sampled)", row=1, col=1, gridcolor=CATPPUCCIN_MOCHA['surface0'])
    plots_fig.update_yaxes(title_text="Cumulative Probability", range=[-0.05, 1.05], row=1, col=2, gridcolor=CATPPUCCIN_MOCHA['surface0'])
    plots_fig.update_xaxes(title_text="Estimated Count", row=1, col=2, gridcolor=CATPPUCCIN_MOCHA['surface0'])
    
    return table_fig, plots_fig

def main():
    plots_dir = os.path.join(OUTPUT_DIR, PLOT_SUBDIR)
    os.makedirs(plots_dir, exist_ok=True)
    
    with open('dashboard_data.json', 'r') as f:
        dashboard_data = json.load(f)
    
    # --- All hard-coded labels are now in a single, clear structure ---
    labels_map = {
        'category': {
            "hh_1102_bed_03": "Couple only with two usual residents & Three bedrooms"
        },
        'sa2': {
            "100301": "Inlets Far North District",
            "109800": "Mangawhai Rural",
            "126601": "Takapuna West"
        }
    }

    # --- HTML STRUCTURE AND MOCHA STYLING ---
    index_html_parts = [
        '<html><head><title>Bayesian Model Diagnostics</title>',
        # Updated inline CSS to use Mocha colors
        f"""<style>
        body {{ font-family: sans-serif; line-height: 1.6; color: {CATPPUCCIN_MOCHA['text']}; background-color: {CATPPUCCIN_MOCHA['base']}; }} 
        iframe {{ border: none; width: 100%; }} 
        .plot-container {{ margin-bottom: 50px; border-top: 2px solid {CATPPUCCIN_MOCHA['surface0']}; padding-top: 20px; }} 
        .intro-text {{ max-width: 800px; margin: 20px auto; padding: 10px; background-color: {CATPPUCCIN_MOCHA['mantle']}; border: 1px solid {CATPPUCCIN_MOCHA['surface0']}; border-radius: 5px;}} 
        h1, h2 {{border-bottom: 1px solid {CATPPUCCIN_MOCHA['surface0']}; padding-bottom: 10px; color: {CATPPUCCIN_MOCHA['text']};}}
        </style>""",
        '</head><body>\n',
        '<h1>Bayesian Model Diagnostic Report</h1>\n'
    ]
    
    explanation_html = """
    <div class="intro-text">
        <p>This report provides diagnostic visualizations for the Bayesian hierarchical model. The following plots are presented for a selection of Statistical Area 2 (SA2) units to assess model performance and convergence.</p>
        <h3>Understanding the Plots</h3>
        <ul>
            <li><strong>Down-Sampled Trace Plot:</strong> This plot shows the sequence of estimated values from the Gibbs sampler for each iteration. A healthy trace plot should resemble "white noise" with no discernible trends, indicating that the sampler has converged on a stable posterior distribution.</li>
            <li><strong>Empirical CDF (ECDF):</strong> This plot represents the model's complete belief about the true value. For any count on the x-axis, the y-axis shows the model's calculated probability that the true value is less than or equal to that count. The median and Maximum A Posteriori (MAP) estimates are shown as lines, and the originally observed value is marked with a star.</li>
        </ul>
        <h3>Selection of Examples</h3>
        <p>The SA2s included in this report were chosen to illustrate a range of model behaviors and serve as exemplars for cases with varying levels of uncertainty. The selected areas are: <b>Inlets Far North District</b>, <b>Mangawhai Rural</b>, and <b>Takapuna West</b>.</p>
    </div>
    """
    index_html_parts.append(explanation_html)

    for category_code, sa2_data in dashboard_data.items():
        category_label = labels_map['category'].get(category_code, category_code)
        
        print(f"Processing Category: {category_label}...")
        index_html_parts.append(f'<h2>Category: {category_label}</h2>\n')

        for sa2_code, data_packet in sa2_data.items():
            sa2_label = labels_map['sa2'].get(sa2_code, sa2_code)
            print(f"  - Generating sandboxed report for SA2: {sa2_label}")
            
            # The function call is now simple and clean
            table_fig, plots_fig = generate_report_figures(category_code, sa2_code, data_packet, labels_map)
            
            table_filename = f"{category_code}_{sa2_code}_table.html"
            plots_filename = f"{category_code}_{sa2_code}_plots.html"
            table_filepath = os.path.join(plots_dir, table_filename)
            plots_filepath = os.path.join(plots_dir, plots_filename)
            
            table_fig.write_html(table_filepath, full_html=True, include_plotlyjs='cdn')
            plots_fig.write_html(plots_filepath, full_html=True, include_plotlyjs='cdn')
            
            relative_table_path = f"./{PLOT_SUBDIR}/{table_filename}"
            relative_plots_path = f"./{PLOT_SUBDIR}/{plots_filename}"
            
            index_html_parts.append(f'<div class="plot-container">\n')
            index_html_parts.append(f'  <iframe src="{relative_table_path}" height="300"></iframe>\n')
            index_html_parts.append(f'  <iframe src="{relative_plots_path}" height="500"></iframe>\n')
            index_html_parts.append('</div>\n')

    index_html_parts.append('</body></html>\n')
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, 'w') as f:
        f.write("".join(index_html_parts))
    
    print(f"\nHTML report generation complete. Opening '{output_path}'...")
    webbrowser.open('file://' + os.path.realpath(output_path))


if __name__ == '__main__':
    main()