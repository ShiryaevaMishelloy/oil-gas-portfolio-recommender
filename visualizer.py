from __future__ import annotations

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MarketVisualizer:
    """Построение аналитических диаграмм по данным нефтегазового сектора."""

    DEFAULT_COLORS = {
        "Покупать (Buy)": "#55A868",
        "Держать (Hold)": "#8FAADC",
        "Продавать (Sell)": "#C44E52",
    }

    RECOMMENDATION_ORDER = [
        "Покупать (Buy)",
        "Держать (Hold)",
        "Продавать (Sell)",
    ]

    COMPANY_ORDER = ["GAZP", "LKOH", "NVTK", "ROSN", "SIBN", "TATN"]

    COMPANY_NAMES = {
        "GAZP": "Газпром",
        "LKOH": "ЛУКОЙЛ",
        "NVTK": "НОВАТЭК",
        "ROSN": "Роснефть",
        "SIBN": "Газпром нефть",
        "TATN": "Татнефть",
    }

    SQL_JOIN_QUERY = """
    SELECT
        q.quote_id AS record_id,
        q.tradedate AS TRADEDATE,
        q.secid AS SECID,
        q.open_price AS OPEN,
        q.high_price AS HIGH,
        q.low_price AS LOW,
        q.close_price AS CLOSE,
        q.volume AS VOLUME,
        q.value_rub AS VALUE,
        c.company_name AS COMPANY_NAME,
        c.subsector AS SUBSECTOR,
        m.brent_usd AS BRENT_USD,
        m.usd_rub AS USD_RUB,
        m.cny_rub AS CNY_RUB,
        m.brent_rub AS BRENT_RUB,
        q.daily_return_pct AS DAILY_RETURN_PCT,
        q.recommendation AS RECOMMENDATION
    FROM stock_quotes q
    JOIN companies c ON q.secid = c.secid
    JOIN macro_indicators m ON q.tradedate = m.tradedate
    ORDER BY q.secid, q.tradedate
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Path | str | None = None,
        dpi: int = 300,
    ) -> None:
        """Подготавливает датафрейм и настройки визуализации."""
        self.df = df.copy()
        self.df["TRADEDATE"] = pd.to_datetime(self.df["TRADEDATE"])
        self.df["YEAR"] = self.df["TRADEDATE"].dt.year
        self.output_dir = (
            Path(output_dir) if output_dir is not None else Path(".")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False

    @classmethod
    def load_data_source(
        cls,
        db_path: Path | str = "database.db",
        csv_fallback: Path | str = "main_dataset.csv",
    ) -> pd.DataFrame:
        """Загружает данные из SQLite (.db), а при отсутствии БД — из CSV."""
        db_file = Path(db_path)
        csv_file = Path(csv_fallback)

        if db_file.exists():
            try:
                with sqlite3.connect(db_file) as conn:
                    return pd.read_sql_query(cls.SQL_JOIN_QUERY, conn)
            except sqlite3.Error as exc:
                print(
                    f"Не удалось прочитать базу {db_file} ({exc}). "
                    f"Переключаюсь на {csv_file}..."
                )

        if csv_file.exists():
            print(f"База {db_file} не найдена. Загружаю данные из {csv_file}...")
            return pd.read_csv(csv_file)

        raise FileNotFoundError(
            f"Не найден ни файл базы данных '{db_file}', ни резервный CSV '{csv_file}'."
        )

    @classmethod
    def from_sources(
        cls,
        db_path: Path | str = "database.db",
        csv_fallback: Path | str = "main_dataset.csv",
        output_dir: Path | str | None = None,
        dpi: int = 300,
    ) -> "MarketVisualizer":
        """Создаёт визуализатор с загрузкой из .db и резервом на .csv."""
        dataframe = cls.load_data_source(db_path=db_path, csv_fallback=csv_fallback)
        return cls(df=dataframe, output_dir=output_dir, dpi=dpi)

    def _save_path(self, filename: str) -> Path:
        """Возвращает путь для сохранения графика."""
        return self.output_dir / filename

    def plot_recommendations_pie(self) -> Path:
        """Строит круговую диаграмму общей структуры рекомендаций."""
        recommendations = (
            self.df["RECOMMENDATION"]
            .value_counts()
            .reindex(self.RECOMMENDATION_ORDER, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            recommendations.values,
            labels=recommendations.index,
            colors=[self.DEFAULT_COLORS[x] for x in self.RECOMMENDATION_ORDER],
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.72,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        ax.set_title(
            "Распределение инвестиционных рекомендаций за весь период",
            fontsize=13,
            pad=15,
        )
        plt.tight_layout()

        out = self._save_path("fig2_1_recommendations.png")
        plt.savefig(out, dpi=self.dpi)
        plt.show()
        return out

    def plot_company_recommendations(self) -> Path:
        """Строит накопительную горизонтальную диаграмму сигналов по компаниям."""
        company_recommendations = (
            pd.crosstab(
                self.df["SECID"], self.df["RECOMMENDATION"], normalize="index"
            )
            * 100
        ).reindex(
            index=self.COMPANY_ORDER,
            columns=self.RECOMMENDATION_ORDER,
            fill_value=0,
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        left = np.zeros(len(company_recommendations))

        for recommendation in self.RECOMMENDATION_ORDER:
            values = company_recommendations[recommendation].values
            ax.barh(
                company_recommendations.index,
                values,
                left=left,
                color=self.DEFAULT_COLORS[recommendation],
                label=recommendation,
                height=0.65,
                edgecolor="white",
            )
            for i, value in enumerate(values):
                if value >= 5:
                    ax.text(
                        left[i] + value / 2,
                        i,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=(
                            "white"
                            if recommendation != "Держать (Hold)"
                            else "#263238"
                        ),
                        fontweight="bold",
                    )
            left += values

        buy_values = company_recommendations["Покупать (Buy)"]
        for i, value in enumerate(buy_values):
            ax.text(
                103,
                i,
                f"Buy: {value:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                color=self.DEFAULT_COLORS["Покупать (Buy)"],
                fontweight="bold",
            )

        ax.set_xlim(0, 119)
        ax.set_yticks(range(len(self.COMPANY_ORDER)))
        ax.set_yticklabels([self.COMPANY_NAMES[x] for x in self.COMPANY_ORDER])
        ax.set_xlabel("Доля рекомендаций, %")
        ax.set_ylabel("Компания")
        ax.set_title(
            "Структура инвестиционных рекомендаций по компаниям за весь период",
            fontsize=13,
            pad=15,
        )
        ax.legend(
            title="Тип рекомендации",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=3,
            frameon=False,
        )
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
        plt.tight_layout()

        out = self._save_path("fig2_2_company_recommendations.png")
        plt.savefig(out, dpi=self.dpi)
        plt.show()
        return out

    def plot_company_recommendations_by_year(self) -> Path:
        """Строит сетку 3x2 с годовой динамикой сигналов по эмитентам."""
        years = sorted(self.df["YEAR"].unique())
        yearly_data = (
            self.df.groupby(["SECID", "YEAR", "RECOMMENDATION"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=self.RECOMMENDATION_ORDER, fill_value=0)
        )
        yearly_percent = (
            yearly_data.div(yearly_data.sum(axis=1), axis=0) * 100
        )

        fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharey=True)
        axes = axes.flatten()

        for i, ticker in enumerate(self.COMPANY_ORDER):
            ax = axes[i]
            company_years = yearly_percent.loc[
                yearly_percent.index.get_level_values("SECID") == ticker
            ].reindex(
                pd.MultiIndex.from_product([[ticker], years], names=["SECID", "YEAR"]),
                fill_value=0,
            )

            bottom = np.zeros(len(years))
            for recommendation in self.RECOMMENDATION_ORDER:
                values = company_years[recommendation].values
                ax.bar(
                    range(len(years)),
                    values,
                    bottom=bottom,
                    color=self.DEFAULT_COLORS[recommendation],
                    edgecolor="white",
                    width=0.7,
                    label=recommendation,
                )
                for j, value in enumerate(values):
                    if value >= 10:
                        ax.text(
                            j,
                            bottom[j] + value / 2,
                            f"{value:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=(
                                "white"
                                if recommendation != "Держать (Hold)"
                                else "#263238"
                            ),
                        )
                bottom += values

            ax.set_title(self.COMPANY_NAMES[ticker], fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(
                [f"{year}*" if year == max(years) else str(year) for year in years],
                rotation=0,
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Доля рекомендаций, %")
            ax.grid(axis="y", alpha=0.2)

        fig.suptitle(
            "Динамика структуры инвестиционных рекомендаций по компаниям",
            fontsize=15,
            y=1.01,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Тип рекомендации",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
        )
        fig.text(
            0.5,
            -0.055,
            "* 2026 год — неполный период наблюдения",
            ha="center",
            fontsize=9,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])

        out = self._save_path("fig2_3_company_recommendations_by_year.png")
        plt.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.show()
        return out

    def plot_volatility(self) -> Path:
        """Строит столбчатую диаграмму стандартного отклонения доходности."""
        volatility = (
            self.df.groupby("SECID")["DAILY_RETURN_PCT"]
            .std()
            .reindex(self.COMPANY_ORDER)
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        gradient = plt.cm.PuRd(np.linspace(0.25, 0.85, len(volatility)))
        bars = ax.bar(
            [self.COMPANY_NAMES[x] for x in volatility.index],
            volatility.values,
            color=gradient,
            edgecolor="white",
            width=0.65,
        )

        for bar, value in zip(bars, volatility.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(
            "Сравнение волатильности акций нефтегазовых компаний",
            fontsize=13,
            pad=15,
        )
        ax.set_xlabel("Компания")
        ax.set_ylabel("Стандартное отклонение, %")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        out = self._save_path("fig2_4_volatility.png")
        plt.savefig(out, dpi=self.dpi)
        plt.show()
        return out

    def plot_brent_rub_dynamics(self) -> Path:
        """Строит линейный график динамики рублёвой цены нефти Brent."""
        macro = self.df.drop_duplicates("TRADEDATE").sort_values("TRADEDATE")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(macro["TRADEDATE"], macro["BRENT_RUB"], linewidth=1.5)
        ax.set_title(
            "Динамика рублёвой стоимости барреля нефти Brent",
            fontsize=13,
            pad=15,
        )
        ax.set_xlabel("Дата")
        ax.set_ylabel("Стоимость, руб./баррель")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        out = self._save_path("fig2_5_brent_rub.png")
        plt.savefig(out, dpi=self.dpi)
        plt.show()
        return out

    def plot_correlation_heatmap(self) -> Path:
        """Строит тепловую карту корреляций доходностей и макрофакторов."""
        stock_returns = self.df.pivot_table(
            index="TRADEDATE",
            columns="SECID",
            values="DAILY_RETURN_PCT",
            aggfunc="mean",
        )
        macro_data = (
            self.df.drop_duplicates("TRADEDATE")
            .set_index("TRADEDATE")
            .sort_index()
        )
        macro_columns = ["BRENT_USD", "USD_RUB", "CNY_RUB", "BRENT_RUB"]
        macro_changes = macro_data[macro_columns].pct_change() * 100

        corr_data = stock_returns.join(macro_changes, how="inner").rename(
            columns={
                "GAZP": "Газпром",
                "LKOH": "ЛУКОЙЛ",
                "NVTK": "НОВАТЭК",
                "ROSN": "Роснефть",
                "SIBN": "Газпром нефть",
                "TATN": "Татнефть",
                "BRENT_USD": "Нефть Brent",
                "USD_RUB": "Курс USD/RUB",
                "CNY_RUB": "Курс CNY/RUB",
                "BRENT_RUB": "Brent в рублях",
            }
        )
        corr = corr_data.corr()

        fig, ax = plt.subplots(figsize=(11, 8))
        image = ax.imshow(
            corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
        )
        labels = corr.columns.tolist()
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(
            "Взаимосвязь котировок акций и рыночных факторов",
            fontsize=13,
            pad=18,
        )

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = corr.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "black",
                )

        colorbar = plt.colorbar(image, ax=ax)
        colorbar.set_label("Коэффициент корреляции Пирсона")
        plt.tight_layout()

        out = self._save_path("fig2_6_correlation.png")
        plt.savefig(out, dpi=self.dpi)
        plt.show()
        return out

    def generate_all(self) -> list[Path]:
        """Строит и сохраняет все шесть диаграмм Раздела 2."""
        saved_paths = [
            self.plot_recommendations_pie(),
            self.plot_company_recommendations(),
            self.plot_company_recommendations_by_year(),
            self.plot_volatility(),
            self.plot_brent_rub_dynamics(),
            self.plot_correlation_heatmap(),
        ]
        print("Все шесть диаграмм успешно построены и сохранены.")
        return saved_paths


if __name__ == "__main__":
    visualizer = MarketVisualizer.from_sources(
        db_path="database.db", csv_fallback="main_dataset.csv"
    )
    visualizer.generate_all()