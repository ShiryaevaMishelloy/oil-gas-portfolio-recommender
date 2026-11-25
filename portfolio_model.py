from __future__ import annotations

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


class OilGasPortfolioRecommender:
    """Обучение моделей регрессии CLOSE, диагностика и рекомендации по акциям."""

    TICKERS_ORDER = ["LKOH", "ROSN", "GAZP", "NVTK", "TATN", "SIBN"]

    COMPANY_TITLES = {
        "LKOH": "ПАО «ЛУКОЙЛ»",
        "ROSN": "ПАО «НК «Роснефть»",
        "GAZP": "ПАО «Газпром»",
        "NVTK": "ПАО «НОВАТЭК»",
        "TATN": "ПАО «Татнефть»",
        "SIBN": "ПАО «Газпром нефть»",
    }

    MACRO_FEATURES = ["BRENT_USD", "USD_RUB", "CNY_RUB", "BRENT_RUB", "VOLUME_MLN"]

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
        test_size: float = 0.20,
        random_state: int = 42,
        threshold_pct: float = 1.5,
        output_dir: Path | str = "figures",
    ) -> None:
        """Готовит выборку с лагом CLOSE_LAG1 и One-Hot признаками."""
        self.test_size = test_size
        self.random_state = random_state
        self.threshold_pct = threshold_pct
        self.output_dir = (
            Path(output_dir) if output_dir is not None else Path(".")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prepared = df.copy()
        prepared["TRADEDATE"] = pd.to_datetime(prepared["TRADEDATE"])
        prepared = prepared.sort_values(["SECID", "TRADEDATE"]).reset_index(
            drop=True
        )
        prepared["VOLUME_MLN"] = prepared["VOLUME"] / 1e6
        prepared["CLOSE_LAG1"] = prepared.groupby("SECID")["CLOSE"].shift(1)
        self.df_clean = prepared.dropna(subset=["CLOSE_LAG1"]).copy()

        secid_dummies = pd.get_dummies(
            self.df_clean["SECID"], prefix="SECID", drop_first=True, dtype=float
        )
        sub_dummies = pd.get_dummies(
            self.df_clean["SUBSECTOR"],
            prefix="SUBSECTOR",
            drop_first=True,
            dtype=float,
        )

        self.X_stage1 = self.df_clean[self.MACRO_FEATURES].copy()
        self.X_stage2 = pd.concat(
            [self.df_clean[["CLOSE_LAG1", *self.MACRO_FEATURES]], sub_dummies, secid_dummies],
            axis=1,
        )
        self.y = self.df_clean["CLOSE"]

        self.X1_train, self.X1_test, self.y_train, self.y_test = train_test_split(
            self.X_stage1,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.X2_train, self.X2_test, _, _ = train_test_split(
            self.X_stage2,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.model_stage1 = LinearRegression()
        self.model_stage2 = LinearRegression()
        self._is_fitted = False

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
        db_path: Path | str = "main_database.db",
        csv_fallback: Path | str = "main_dataset.csv",
        test_size: float = 0.20,
        random_state: int = 42,
        threshold_pct: float = 1.5,
        output_dir: Path | str = "figures",
    ) -> "OilGasPortfolioRecommender":
        """Создаёт модель с загрузкой из .db и резервом на .csv."""
        dataframe = cls.load_data_source(db_path=db_path, csv_fallback=csv_fallback)
        return cls(
            df=dataframe,
            test_size=test_size,
            random_state=random_state,
            threshold_pct=threshold_pct,
            output_dir=output_dir,
        )

    @staticmethod
    def adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
        """Вычисляет скорректированный коэффициент детерминации."""
        return 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    def fit(self) -> "OilGasPortfolioRecommender":
        """Обучает базовую (Этап 1) и улучшенную (Этап 2) модели регрессии."""
        self.model_stage1.fit(self.X1_train, self.y_train)
        self.model_stage2.fit(self.X2_train, self.y_train)
        self._is_fitted = True
        return self

    def _ensure_fitted(self) -> None:
        """Проверяет, что модели обучены перед прогнозом."""
        if not self._is_fitted:
            self.fit()

    def evaluate_stages(self) -> None:
        """Выводит метрики качества моделей Этапа 1 и Этапа 2."""
        self._ensure_fitted()
        stages = [
            (
                "Этап 1 (Первичная модель — только макрофакторы)",
                self.model_stage1,
                self.X1_train,
                self.X1_test,
            ),
            (
                "Этап 2 (Доработанная модель — One-Hot + CLOSE_LAG1)",
                self.model_stage2,
                self.X2_train,
                self.X2_test,
            ),
        ]

        for stage_name, model, x_tr, x_te in stages:
            pred_tr = model.predict(x_tr)
            pred_te = model.predict(x_te)
            r2_tr = r2_score(self.y_train, pred_tr)
            r2_te = r2_score(self.y_test, pred_te)

            print(f"{stage_name}:")
            print(
                f"  Train -> R2={r2_tr:.4f}, "
                f"AdjR2={self.adjusted_r2(r2_tr, len(x_tr), x_tr.shape[1]):.4f}, "
                f"MAE={mean_absolute_error(self.y_train, pred_tr):.2f}, "
                f"RMSE={root_mean_squared_error(self.y_train, pred_tr):.2f}"
            )
            print(
                f"  Test  -> R2={r2_te:.4f}, "
                f"AdjR2={self.adjusted_r2(r2_te, len(x_te), x_te.shape[1]):.4f}, "
                f"MAE={mean_absolute_error(self.y_test, pred_te):.2f}, "
                f"RMSE={root_mean_squared_error(self.y_test, pred_te):.2f}"
            )

        print("\nКоэффициенты модели Этапа 2:")
        print(f"  Intercept: {self.model_stage2.intercept_:.4f}")
        for col_name, coef_val in zip(
            self.X_stage2.columns, self.model_stage2.coef_
        ):
            print(f"  {col_name}: {coef_val:.4f}")

    def evaluate_individual_companies(self) -> None:
        """Обучает и сравнивает индивидуальные модели по каждому эмитенту."""
        print("\nИндивидуальные модели по эмитентам:")
        for ticker in self.TICKERS_ORDER:
            sub = self.df_clean[self.df_clean["SECID"] == ticker].copy()
            xi_1 = sub[self.MACRO_FEATURES]
            xi_2 = sub[["CLOSE_LAG1", *self.MACRO_FEATURES]]
            yi = sub["CLOSE"]

            x1_tr, x1_te, yi_tr, yi_te = train_test_split(
                xi_1, yi, test_size=self.test_size, random_state=self.random_state
            )
            x2_tr, x2_te, _, _ = train_test_split(
                xi_2, yi, test_size=self.test_size, random_state=self.random_state
            )

            m1 = LinearRegression().fit(x1_tr, yi_tr)
            m2 = LinearRegression().fit(x2_tr, yi_tr)
            p1_te = m1.predict(x1_te)
            p2_te = m2.predict(x2_te)
            r2_2_te = r2_score(yi_te, p2_te)

            print(
                f"  {ticker}: Этап 1 R2={r2_score(yi_te, p1_te):.4f}, "
                f"MAE={mean_absolute_error(yi_te, p1_te):.2f} -> "
                f"Этап 2 R2={r2_2_te:.4f}, "
                f"AdjR2={self.adjusted_r2(r2_2_te, len(x2_te), x2_te.shape[1]):.4f}, "
                f"MAE={mean_absolute_error(yi_te, p2_te):.2f}, "
                f"RMSE={root_mean_squared_error(yi_te, p2_te):.2f}"
            )

    def get_latest_recommendations(self) -> pd.DataFrame:
        """Возвращает рекомендации по компаниям на последнюю дату датасета."""
        self._ensure_fitted()
        df_scored = self.df_clean.copy()
        df_scored["PRED_CLOSE"] = self.model_stage2.predict(self.X_stage2)
        df_scored["DIFF_PCT"] = (
            (df_scored["PRED_CLOSE"] - df_scored["CLOSE"]) / df_scored["CLOSE"]
        ) * 100

        def classify(diff_pct: float) -> str:
            if diff_pct > self.threshold_pct:
                return "Покупать (Buy)"
            if diff_pct < -self.threshold_pct:
                return "Продавать (Sell)"
            return "Держать (Hold)"

        df_scored["MODEL_RECOMMENDATION"] = df_scored["DIFF_PCT"].apply(classify)
        latest_date = df_scored["TRADEDATE"].max()
        return df_scored[df_scored["TRADEDATE"] == latest_date][
            [
                "SECID",
                "COMPANY_NAME",
                "CLOSE",
                "PRED_CLOSE",
                "DIFF_PCT",
                "MODEL_RECOMMENDATION",
                "RECOMMENDATION",
            ]
        ].sort_values("DIFF_PCT", ascending=False)

    def plot_diagnostics(self, dpi: int = 300) -> list[Path]:
        """Строит графики сравнения этапов, остатков и моделей по эмитентам."""
        self._ensure_fitted()
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False

        p1_te = self.model_stage1.predict(self.X1_test)
        p2_te = self.model_stage2.predict(self.X2_test)
        residuals_2 = self.y_test - p2_te

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
        axes[0].scatter(
            self.y_test,
            p1_te,
            alpha=0.35,
            s=16,
            color="#64748B",
            label="Этап 1 (Макро + структура)",
        )
        mn, mx = min(self.y_test.min(), p1_te.min()), max(
            self.y_test.max(), p1_te.max()
        )
        axes[0].plot(
            [mn, mx],
            [mn, mx],
            color="#DC2626",
            linestyle="--",
            linewidth=1.8,
            label="Идеальный прогноз",
        )
        axes[0].set_title(
            f"Этап 1: Базовая модель (R² = {r2_score(self.y_test, p1_te):.4f}, "
            f"MAE = {mean_absolute_error(self.y_test, p1_te):.1f} руб.)",
            fontsize=11,
        )
        axes[0].set_xlabel("Фактическая цена CLOSE, руб.")
        axes[0].set_ylabel("Предсказанная цена CLOSE, руб.")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.25)

        axes[1].scatter(
            self.y_test,
            p2_te,
            alpha=0.35,
            s=16,
            color="#2563EB",
            label="Этап 2 (+ лаг CLOSE_LAG1)",
        )
        mn2, mx2 = min(self.y_test.min(), p2_te.min()), max(
            self.y_test.max(), p2_te.max()
        )
        axes[1].plot(
            [mn2, mx2],
            [mn2, mx2],
            color="#DC2626",
            linestyle="--",
            linewidth=1.8,
            label="Идеальный прогноз",
        )
        axes[1].set_title(
            f"Этап 2: Улучшенная модель (R² = {r2_score(self.y_test, p2_te):.4f}, "
            f"MAE = {mean_absolute_error(self.y_test, p2_te):.1f} руб.)",
            fontsize=11,
        )
        axes[1].set_xlabel("Фактическая цена CLOSE, руб.")
        axes[1].set_ylabel("Предсказанная цена CLOSE, руб.")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.25)

        fig.suptitle(
            "Сравнение качества прогноза на тестовой выборке до и после добавления авторегрессионного лага",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        fig1_path = self.output_dir / "fig3_1_stage1_vs_stage2.png"
        plt.savefig(fig1_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].scatter(p2_te, residuals_2, alpha=0.35, s=16, color="#0D9488")
        axes[0].axhline(0, color="#DC2626", linestyle="--", linewidth=1.8)
        axes[0].set_title(
            "Остатки улучшенной модели (Этап 2) против прогноза", fontsize=11.5
        )
        axes[0].set_xlabel("Предсказанная цена CLOSE, руб.")
        axes[0].set_ylabel("Остаток (Факт − Прогноз), руб.")
        axes[0].grid(True, alpha=0.25)

        axes[1].hist(residuals_2, bins=50, color="#3B82F6", edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="#DC2626", linestyle="--", linewidth=1.8)
        axes[1].set_title(
            "Распределение остатков улучшенной модели (тест)", fontsize=11.5
        )
        axes[1].set_xlabel("Величина ошибки (остаток), руб.")
        axes[1].set_ylabel("Частота наблюдений")
        axes[1].grid(True, alpha=0.25)

        plt.tight_layout()
        fig2_path = self.output_dir / "fig3_2_residuals_stage2.png"
        plt.savefig(fig2_path, dpi=dpi)
        plt.show()

        fig, axes = plt.subplots(3, 2, figsize=(13.5, 12))
        axes = axes.flatten()
        for idx, ticker in enumerate(self.TICKERS_ORDER):
            ax = axes[idx]
            sub = self.df_clean[self.df_clean["SECID"] == ticker].copy()
            xi = sub[["CLOSE_LAG1", *self.MACRO_FEATURES]]
            yi = sub["CLOSE"]
            xi_tr, xi_te, yi_tr, yi_te = train_test_split(
                xi, yi, test_size=self.test_size, random_state=self.random_state
            )
            mi = LinearRegression().fit(xi_tr, yi_tr)
            pi_te = mi.predict(xi_te)

            ax.scatter(yi_te, pi_te, alpha=0.45, s=16, color="#2563EB")
            mn_i, mx_i = min(yi_te.min(), pi_te.min()), max(
                yi_te.max(), pi_te.max()
            )
            ax.plot(
                [mn_i, mx_i],
                [mn_i, mx_i],
                color="#DC2626",
                linestyle="--",
                linewidth=1.6,
            )
            ax.set_title(
                f"{self.COMPANY_TITLES[ticker]} "
                f"(R² = {r2_score(yi_te, pi_te):.4f}, "
                f"MAE = {mean_absolute_error(yi_te, pi_te):.1f} руб.)",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("Фактическая цена CLOSE, руб.")
            ax.set_ylabel("Прогноз CLOSE, руб.")
            ax.grid(True, alpha=0.25)

        fig.suptitle(
            "Фактические и предсказанные значения цены закрытия по индивидуальным моделям (Этап 2)",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        fig3_path = self.output_dir / "fig3_3_individual_stage2.png"
        plt.savefig(fig3_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        return [fig1_path, fig2_path, fig3_path]


if __name__ == "__main__":
    recommender = OilGasPortfolioRecommender.from_sources(
        db_path="database.db", csv_fallback="main_dataset.csv"
    ).fit()
    recommender.evaluate_stages()
    recommender.evaluate_individual_companies()

    print("\nРекомендации по компаниям на последнюю дату:")
    print(recommender.get_latest_recommendations().to_string(index=False))
    recommender.plot_diagnostics()