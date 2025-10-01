# simple_plot.py
# Final version: Saves the plot to an HTML file to avoid terminal errors.

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

def _create_step_data(x, y):
    """A helper function to manually create coordinates for a step plot."""
    x_step = np.repeat(x, 2)[1:]
    y_step = np.repeat(y, 2)[:-1]
    return x_step, y_step

# --- 1. Load the data ---
try:
    with open('dashboard_data.json', 'r') as f:
        dashboard_data = json.load(f)
except FileNotFoundError:
    print("Error: `dashboard_data.json` not found. Place this script in the same folder.")
    exit()

# --- 2. CHOOSE WHICH PLOT TO SEE ---
# --- Edit the line below to view a different SA2 ---
sa2_to_plot = '100301' # Options are '100301', '109800', '126601'
# ---------------------------------------------------

category = "hh_1102_bed_03"
print(f"Generating plot for SA2: {sa2_to_plot}...")

try:
    data = dashboard_data[category][sa2_to_plot]
except KeyError:
    print(f"Error: SA2 code '{sa2_to_plot}' not found in the data file.")
    exit()

# --- 3. Generate the plot figure ---
summary_stats = data['summary_stats']
trace_data = data['trace']
ecdf_x = np.array(data['ecdf_x'])
ecdf_y = np.array(data['ecdf_y'])
median_val = summary_stats['Posterior Median']
map_val = summary_stats['MAP Estimate']
obs_value = summary_stats['Observed Value']

plots_fig = make_subplots(rows=1, cols=2, subplot_titles=('Down-Sampled Trace Plot', 'Empirical CDF'))
plots_fig.add_trace(go.Scatter(y=trace_data, mode='lines', name='Chain Trace', line=dict(color='cornflowerblue')), row=1, col=1)

if len(ecdf_x) > 1:
    x_step, y_step = _create_step_data(ecdf_x, ecdf_y)
    plots_fig.add_trace(go.Scatter(x=x_step, y=y_step, mode='lines', name='ECDF', line=dict(color='darkorange')), row=1, col=2)
else:
    plots_fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode='markers', name='ECDF (Single Point)', marker=dict(color='darkorange', size=15, symbol='diamond')), row=1, col=2)

plots_fig.add_hline(y=median_val, line_dash="dash", line_color="purple", name="Median", legendgroup="median", showlegend=True, row=1, col=1)
plots_fig.add_hline(y=map_val, line_dash="dot", line_color="green", name="MAP", legendgroup="map", showlegend=True, row=1, col=1)
plots_fig.add_trace(go.Scatter(x=[median_val, median_val], y=[0, 1], mode='lines', line_dash="dash", line_color="purple", name="Median", legendgroup="median", showlegend=False), row=1, col=2)
plots_fig.add_trace(go.Scatter(x=[map_val, map_val], y=[0, 1], mode='lines', line_dash="dot", line_color="green", name="MAP", legendgroup="map", showlegend=False), row=1, col=2)
if obs_value != "Suppressed":
    plots_fig.add_trace(go.Scatter(x=[obs_value, obs_value], y=[0, 1], mode='lines', line_dash="solid", line_color="red", name="Observed", legendgroup="observed", showlegend=True), row=1, col=2)
    obs_prob_index = np.searchsorted(ecdf_x, obs_value, side='right')
    obs_prob = ecdf_y[obs_prob_index - 1] if obs_prob_index > 0 else 0
    plots_fig.add_trace(go.Scatter(x=[obs_value], y=[obs_prob], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Observed Value', legendgroup="observed", showlegend=False), row=1, col=2)

plots_fig.update_layout(height=450, showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
plots_fig.update_yaxes(title_text="Estimated Count", row=1, col=1)
plots_fig.update_xaxes(title_text="Iteration (Down-sampled)", row=1, col=1)
plots_fig.update_yaxes(title_text="Cumulative Probability", range=[-0.05, 1.05], row=1, col=2)
plots_fig.update_xaxes(title_text="Estimated Count", row=1, col=2)

# --- 4. Save the plot to a file and open it ---
output_filename = 'interactive_plot.html'
plots_fig.write_html(output_filename)

print(f"Plot saved to '{output_filename}'. Opening in browser...")
webbrowser.open('file://' + os.path.realpath(output_filename))