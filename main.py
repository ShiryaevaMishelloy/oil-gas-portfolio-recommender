from __future__ import annotations

from pathlib import Path
from data_collector import MarketDataCollector
from portfolio_model import OilGasPortfolioRecommender
from visualizer import MarketVisualizer


def run_pipeline(
    collect_fresh_data: bool = False,
    db_path: Path | str = "database.db",
    csv_path: Path | str = "main_dataset.csv",
    figures_dir: Path | str = "figures",
) -> None:
    """Запускает сбор данных (при необходимости), визуализацию и модель."""
    if collect_fresh_data:
        collector = MarketDataCollector(output_file=csv_path)
        dataset = collector.build_dataset()
        collector.print_report_tables(dataset)
    else:
        dataset = MarketVisualizer.load_data_source(
            db_path=db_path, csv_fallback=csv_path
        )

    visualizer = MarketVisualizer(df=dataset, output_dir=figures_dir)
    visualizer.generate_all()

    recommender = OilGasPortfolioRecommender(
        df=dataset, output_dir=figures_dir
    ).fit()
    recommender.evaluate_stages()
    recommender.evaluate_individual_companies()

    print("\nИтоговые рекомендации по компаниям на последнюю дату:")
    print(recommender.get_latest_recommendations().to_string(index=False))
    recommender.plot_diagnostics()


if __name__ == "__main__":
    run_pipeline(collect_fresh_data=False)