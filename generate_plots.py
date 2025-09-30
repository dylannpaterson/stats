import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- CONFIGURATION ---
CURATED_SA2S = {
    'hh_1102_bed_03': [100301, 109800, 126601]
}
TRACE_DOWNSAMPLE_FACTOR = 10
OUTPUT_DIR = "docs"
OUTPUT_FILENAME = "index.html"

def generate_report_figures(category, sa2_code, data):
    """
    Generates the Plotly Figure objects for a single SA2's diagnostic plots.
    """
    summary_stats = data['summary_stats']
    trace_data = data['trace']
    ecdf_x = np.array(data['ecdf_x'])
    ecdf_y = np.array(data['ecdf_y'])
    
    obs_value = summary_stats['Observed Value']
    median_val = summary_stats['Posterior Median']
    map_val = summary_stats['MAP Estimate']
    
    # --- Figure 1: The Summary Table ---
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['Statistic', 'Value'], fill_color='paleturquoise', align='left', font=dict(size=14)),
        cells=dict(values=[list(summary_stats.keys()), list(summary_stats.values())], fill_color='lavender', align='left', font=dict(size=12), height=30)
    )])
    table_fig.update_layout(
        title_text=f"Diagnostic Suite for SA2: <b>{sa2_code}</b> (Category: {category})",
        title_font_size=20,
        margin=dict(l=10, r=10, t=80, b=10)
    )

    # --- Figure 2: The Diagnostic Plots ---
    plots_fig = make_subplots(rows=1, cols=2, subplot_titles=('Down-Sampled Trace Plot', 'Empirical CDF'))

    # --- Add Main Traces ---
    plots_fig.add_trace(go.Scatter(y=trace_data, mode='lines', name='Chain Trace', line=dict(color='cornflowerblue')), row=1, col=1)
    plots_fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode='lines', name='ECDF', line_shape='hv', line=dict(color='darkorange')), row=1, col=2)
    
    # --- Add Helper Lines as Shapes (more robust than add_hline/vline) ---
    plots_fig.add_shape(type="line", y0=median_val, y1=median_val, x0=0, x1=1, xref="paper", yref="y1", line=dict(color="purple", dash="dash"))
    plots_fig.add_shape(type="line", y0=map_val, y1=map_val, x0=0, x1=1, xref="paper", yref="y1", line=dict(color="green", dash="dot"))
    plots_fig.add_shape(type="line", x0=median_val, x1=median_val, y0=0, y1=1, xref="x2", yref="paper", line=dict(color="purple", dash="dash"))
    plots_fig.add_shape(type="line", x0=map_val, x1=map_val, y0=0, y1=1, xref="x2", yref="paper", line=dict(color="green", dash="dot"))

    if obs_value != "Suppressed":
        plots_fig.add_shape(type="line", x0=obs_value, x1=obs_value, y0=0, y1=1, xref="x2", yref="paper", line=dict(color="red", dash="solid"))
        obs_prob_index = np.where(ecdf_x == obs_value)[0]
        if len(obs_prob_index) > 0:
            obs_prob = ecdf_y[obs_prob_index[0]]
            plots_fig.add_trace(go.Scatter(x=[obs_value], y=[obs_prob], mode='markers',
                marker=dict(color='red', size=12, symbol='star', line=dict(width=1, color='DarkSlateGrey')),
                name='Observed Value Marker'), row=1, col=2)

    # --- Add Invisible Traces for a Clean Legend ---
    plots_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Median', line=dict(color='purple', dash='dash')))
    plots_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='MAP', line=dict(color='green', dash='dot')))
    if obs_value != "Suppressed":
        plots_fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Observed', line=dict(color='red', dash='solid')))

    # --- Layout Updates ---
    plots_fig.update_layout(height=450, showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    plots_fig.update_yaxes(title_text="Estimated Count", row=1, col=1)
    plots_fig.update_xaxes(title_text="Iteration (Down-sampled)", row=1, col=1)
    plots_fig.update_yaxes(title_text="Cumulative Probability", range=[-0.05, 1.05], row=1, col=2)
    plots_fig.update_xaxes(title_text="Estimated Count", row=1, col=2)
    
    return table_fig, plots_fig

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    with open(output_path, 'w') as f:
        f.write('<html><head><title>Bayesian Model Diagnostics</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>\n')
        
        for category, sa2_codes in CURATED_SA2S.items():
            print(f"Processing category: {category}...")
            try:
                summary_path = f'data/outputs/bayesian_summaries/sa2_summary_{category}.csv'
                samples_path = f'data/outputs/bayesian_samples/sa2_samples_{category}.npy'
                summary_df = pd.read_csv(summary_path)
                posterior_samples = np.load(samples_path)
            except FileNotFoundError as e:
                print(f"  Warning: Could not find files for {category}. Skipping. Details: {e}")
                continue

            for sa2_code in sa2_codes:
                print(f"  - Generating report for SA2: {sa2_code}")
                try:
                    sa2_row = summary_df.loc[summary_df['sa2_code'] == sa2_code]
                    sa2_index = sa2_row.index[0]
                except IndexError:
                    print(f"    Warning: SA2 code {sa2_code} not found. Skipping.")
                    continue

                sa2_chain = posterior_samples[:, sa2_index]
                obs_value = sa2_row['OBS_VALUE'].iloc[0]
                x_ecdf, counts = np.unique(sa2_chain, return_counts=True)
                y_ecdf = np.cumsum(counts) / sa2_chain.size

                data_for_plot = {
                    'summary_stats': {
                        'Observed Value': int(obs_value) if pd.notna(obs_value) else "Suppressed",
                        'Posterior Mean': sa2_row['estimated_count_mean'].iloc[0],
                        'Posterior Median': np.median(sa2_chain),
                        'MAP Estimate': sa2_row['estimated_count_map'].iloc[0],
                        '95% CI Lower': sa2_row['ci_95_lower'].iloc[0],
                        '95% CI Upper': sa2_row['ci_95_upper'].iloc[0]
                    },
                    'trace': sa2_chain[::TRACE_DOWNSAMPLE_FACTOR].tolist(),
                    'ecdf_x': x_ecdf.tolist(),
                    'ecdf_y': y_ecdf.tolist()
                }
                
                table_fig, plots_fig = generate_report_figures(category, sa2_code, data_for_plot)
                
                # Append figures to the single HTML file, wrapped in a div for spacing
                f.write('<div style="margin-bottom: 50px;">\n')
                f.write(table_fig.to_html(full_html=False, include_plotlyjs=False))
                f.write(plots_fig.to_html(full_html=False, include_plotlyjs=False))
                f.write('</div>\n<hr>\n')
        
        f.write('</body></html>\n')
    
    print(f"\nHTML report generation complete. File saved to: {output_path}")

if __name__ == '__main__':
    main()