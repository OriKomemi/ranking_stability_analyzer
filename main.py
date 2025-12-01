import numpy as np
from analyzer import RankingStabilityAnalyzer
from visualization import create_2d_visualization

def demo_2d_analysis():
    """Demonstrate 2D stability analysis with the example from the paper."""
    print("=" * 70)
    print("DEMO 1: 2D Stability Analysis (Candidate Ranking)")
    print("=" * 70)

    # Data from Figure 1 of the paper
    data = np.array([
        [0.63, 0.71],  # Candidate 1
        [0.83, 0.65],  # Candidate 2
        [0.58, 0.78],  # Candidate 3
        [0.70, 0.68],  # Candidate 4
        [0.53, 0.82]   # Candidate 5
    ])

    attribute_names = ["Aptitude", "Experience"]
    item_names = [f"Candidate {i+1}" for i in range(5)]

    analyzer = RankingStabilityAnalyzer(data, attribute_names)

    print(f"\nDataset: {len(data)} candidates, {data.shape[1]} attributes")
    print(f"Attributes: {', '.join(attribute_names)}")

    # Test different weight combinations
    weight_scenarios = [
        ("Equal weights", np.array([0.5, 0.5])),
        ("Aptitude focused", np.array([0.7, 0.3])),
        ("Experience focused", np.array([0.3, 0.7])),
    ]

    print("\n" + "-" * 70)
    print("Stability Verification for Different Weight Scenarios")
    print("-" * 70)

    for scenario_name, weights in weight_scenarios:
        weights = weights / np.linalg.norm(weights)  # Normalize
        ranking = analyzer.get_ranking(weights)
        stability, (theta_min, theta_max) = analyzer.verify_stability_2d(ranking)

        print(f"\n{scenario_name}: w = [{weights[0]:.2f}, {weights[1]:.2f}]")
        print(f"  Ranking: {' > '.join([item_names[i] for i in ranking])}")
        print(f"  Stability: {stability:.6f} ({stability*100:.3f}%)")
        print(f"  Angle range: [{np.degrees(theta_min):.2f}°, {np.degrees(theta_max):.2f}°]")

        if stability > 0.1:
            print(f"  ✅ HIGHLY STABLE")
        elif stability > 0.01:
            print(f"  ⚠️  MODERATELY STABLE")
        else:
            print(f"  ❌ UNSTABLE")

    # Enumerate stable rankings
    print("\n" + "-" * 70)
    print("Top 5 Most Stable Rankings")
    print("-" * 70)

    stable_rankings = analyzer.enumerate_stable_rankings_2d(n_top=5)

    for idx, (ranking, stability, (theta_min, theta_max)) in enumerate(stable_rankings):
        print(f"\n#{idx+1} - Stability: {stability:.6f} ({stability*100:.3f}%)")
        print(f"  Ranking: {' > '.join([item_names[i] for i in ranking])}")
        print(f"  Angle range: [{np.degrees(theta_min):.2f}°, {np.degrees(theta_max):.2f}°]")
        print(f"  Angular width: {np.degrees(theta_max - theta_min):.2f}°")
    # Create visualization for one of the scenarios
    scenario_name, weights = weight_scenarios[0]
    weights = weights / np.linalg.norm(weights)  # Normalize
    fig = create_2d_visualization(analyzer, weights=weights,
                                  show_exchanges=True,
                                  item_names=item_names)
    fig.show()

def main():
    demo_2d_analysis()

if __name__ == "__main__":
    main()