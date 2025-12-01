import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Optional



def create_2d_visualization(analyzer, weights: Optional[np.ndarray] = None,
                           show_exchanges: bool = True,
                           item_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create 2D visualization in original space.

    Shows items as points, scoring function as a ray, and optionally
    ordering exchange lines where items swap positions.

    Args:
        analyzer: RankingStabilityAnalyzer instance
        weights: Optional weight vector to display as scoring function ray
        show_exchanges: Whether to show ordering exchange lines
        item_names: Optional list of item names for labels

    Returns:
        Plotly figure object
    """
    if analyzer.n_attrs != 2:
        return None

    fig = go.Figure()

    # Plot items as points
    labels = [item_names[i] if item_names else f"Item {i}" for i in range(analyzer.n_items)]

    fig.add_trace(go.Scatter(
        x=analyzer.data[:, 0],
        y=analyzer.data[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='blue', symbol='circle'),
        text=labels,
        textposition="top center",
        name="Items",
        hovertemplate='<b>%{text}</b><br>x1: %{x:.3f}<br>y1: %{y:.3f}<extra></extra>'
    ))

    # Add scoring function ray if weights provided
    if weights is not None:
        weights = weights / np.linalg.norm(weights)
        max_val = max(analyzer.data.max(), 1.0)
        fig.add_trace(go.Scatter(
            x=[0, weights[0] * max_val],
            y=[0, weights[1] * max_val],
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Scoring Function<br>w=[{weights[0]:.2f}, {weights[1]:.2f}]'
        ))

        # Add projections
        scores = analyzer.compute_score(weights)
        for i, (point, score) in enumerate(zip(analyzer.data, scores)):
            proj = score * weights
            fig.add_trace(go.Scatter(
                x=[point[0], proj[0]],
                y=[point[1], proj[1]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add ordering exchanges if requested
    if show_exchanges:
        for i in range(analyzer.n_items):
            for j in range(i + 1, analyzer.n_items):
                theta = analyzer.compute_ordering_exchange_2d(i, j)
                if theta is not None:
                    max_val = max(analyzer.data.max(), 1.0)
                    exchange_vec = np.array([np.cos(theta), np.sin(theta)])
                    fig.add_trace(go.Scatter(
                        x=[0, exchange_vec[0] * max_val],
                        y=[0, exchange_vec[1] * max_val],
                        mode='lines',
                        line=dict(color='lightgray', width=1, dash='dash'),
                        showlegend=False,
                        hovertemplate=f'Exchange: {labels[i]} ↔ {labels[j]}<br>θ={np.degrees(theta):.1f}°<extra></extra>'
                    ))

    fig.update_layout(
        title="Original Space: Items and Scoring Functions",
        xaxis_title=analyzer.attribute_names[0],
        yaxis_title=analyzer.attribute_names[1],
        hovermode='closest',
        width=700,
        height=600,
        showlegend=True
    )

    return fig
