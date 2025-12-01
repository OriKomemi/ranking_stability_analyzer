import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from typing import List, Tuple, Optional
from visualization import create_2d_visualization

from analyzer import RankingStabilityAnalyzer


def render_2d_visualizations(analyzer, attribute_names, item_names):
    """Render 2D visualizations (original and dual space)."""
    container = st.container()

    with container:
        st.markdown("##### Original Space")

        # Weight input
        w1 = st.slider("Weight for " + attribute_names[0], 0.0, 1.0, 0.5, 0.01, key="viz_w1")
        w2 = st.slider("Weight for " + attribute_names[1], 0.0, 1.0, 0.5, 0.01, key="viz_w2")
        weights = np.array([w1, w2])

        show_exchanges = st.checkbox("Show ordering exchanges", value=True)

        fig_original = create_2d_visualization(analyzer, weights, show_exchanges, item_names)
        st.plotly_chart(fig_original, use_container_width=True)

        # Show ranking
        ranking = analyzer.get_ranking(weights)
        st.markdown("**Ranking for current weights:**")
        st.info(ranking)

def main():
    """Main Streamlit application."""

    # Dataset selection
    data = np.array([
        [0.63, 0.71],  # Candidate 1
        [0.83, 0.65],  # Candidate 2
        [0.58, 0.78],  # Candidate 3
        [0.70, 0.68],  # Candidate 4
        [0.53, 0.82]   # Candidate 5
    ])
    attribute_names = ["Aptitude", "Experience"]
    item_names = [f"Candidate {i+1}" for i in range(5)]

    if data is None:
        st.info("No dataset found.")
        return

    # Create analyzer
    analyzer = RankingStabilityAnalyzer(data, attribute_names)

    st.header("2D Ranking Stability Analysis Demo")
    visualization = st.container()

    with visualization:
        render_2d_visualizations(analyzer, attribute_names, item_names)




if __name__ == "__main__":
    main()