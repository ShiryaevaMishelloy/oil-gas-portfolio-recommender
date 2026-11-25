from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MarketDataCollector:
    """Сбор, синхронизация и обогащение котировок MOEX, ЦБ РФ и нефти Brent."""

    DEFAULT_COMPANIES = {
        "LKOH": {"name": 'ПАО "ЛУКОЙЛ"', "subsector": "Нефтедобыча и переработка"},
        "ROSN": {"name": 'ПАО "НК "Роснефть"', "subsector": "Нефтедобыча и переработка"},
        "GAZP": {"name": 'ПАО "Газпром"', "subsector": "Газовая отрасль"},
        "NVTK": {"name": 'ПАО "НОВАТЭК"', "subsector": "Газовая отрасль"},
        "TATN": {"name": 'ПАО "Татнефть"', "subsector": "Нефтедобыча и переработка"},
        "SIBN": {"name": 'ПАО "Газпром нефть"', "subsector": "Нефтедобыча и переработка"},
    }

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        )
    }

    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str | None = None,
        output_file: Path | str | None = None,
        companies_info: dict | None = None,
        sma_window: int = 20,
        brent_window: int = 252,
        brent_min_periods: int = 60,
        brent_buy_quantile: float = 0.70,
        brent_sell_quantile: float = 0.30,
        buy_ratio: float = 0.975,
        sell_ratio: float = 1.030,
        request_timeout: int = 60,
    ) -> None:
        """Инициализирует параметры ETL-конвейера и расчёта сигналов."""
        self.start_date = start_date
        self.end_date = end_date or (date.today() + timedelta(days=1)).isoformat()
        self.output_file = (
            Path(output_file)
            if output_file is not None
            else Path(__file__).resolve().parent / "main_dataset.csv"
        )
        self.companies_info = companies_info or self.DEFAULT_COMPANIES
        self.sma_window = sma_window
        self.brent_window = brent_window
        self.brent_min_periods = brent_min_periods
        self.brent_buy_quantile = brent_buy_quantile
        self.brent_sell_quantile = brent_sell_quantile
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        self.request_timeout = request_timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Создаёт HTTP-сессию с политикой повторных попыток."""
        session = requests.Session()
        session.headers.update(self.DEFAULT_HEADERS)

        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get_cbr_currency_code(self, iso_char_code: str) -> str:
        """Возвращает внутренний код валюты ЦБ РФ по символьному ISO-коду."""
        url = "https://www.cbr.ru/scripts/XML_valFull.asp"
        response = self.session.get(url, timeout=self.request_timeout)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        for item in root.findall("Item"):
            char_code = item.findtext("ISO_Char_Code")
            if char_code == iso_char_code:
                return item.attrib["ID"].strip()

        raise ValueError(f"В справочнике ЦБ РФ не найдена валюта {iso_char_code}")

    def fetch_cbr_dynamic(self, currency_code: str, column_name: str) -> pd.DataFrame:
        """Загружает динамику курса валюты ЦБ РФ с приведением к номиналу."""
        url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
        params = {
            "date_req1": pd.to_datetime(self.start_date).strftime("%d/%m/%Y"),
            "date_req2": pd.to_datetime(self.end_date).strftime("%d/%m/%Y"),
            "VAL_NM_RQ": currency_code,
        }
        response = self.session.get(
            url, params=params, timeout=self.request_timeout
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        records = []
        for record in root.findall("Record"):
            nominal = float(record.findtext("Nominal").replace(",", "."))
            value = float(record.findtext("Value").replace(",", "."))
            rate = value / nominal
            records.append(
                {
                    "TRADEDATE": pd.to_datetime(
                        record.attrib["Date"], format="%d.%m.%Y"
                    ),
                    column_name: round(rate, 4),
                }
            )

        result = pd.DataFrame(records)
        if result.empty:
            raise RuntimeError(f"ЦБ РФ не вернул данные для {column_name}")

        return result.drop_duplicates("TRADEDATE").sort_values("TRADEDATE")

    def fetch_moex_history(self, ticker: str) -> pd.DataFrame:
        """Постранично загружает историю торгов акцией в режиме TQBR на MOEX."""
        url = (
            "https://iss.moex.com/iss/history/engines/stock/"
            f"markets/shares/boards/TQBR/securities/{ticker}.json"
        )
        all_rows = []
        columns = None
        start = 0

        while True:
            params = {
                "from": self.start_date,
                "till": self.end_date,
                "start": start,
                "iss.meta": "off",
            }
            response = self.session.get(
                url, params=params, timeout=self.request_timeout
            )
            response.raise_for_status()

            payload = response.json()
            history = payload.get("history", {})
            if columns is None:
                columns = history.get("columns", [])

            rows = history.get("data", [])
            if not rows:
                break

            all_rows.extend(rows)
            start += len(rows)
            if len(rows) < 100:
                break
            time.sleep(0.1)

        if not all_rows or not columns:
            raise RuntimeError(f"MOEX не вернула данные для {ticker}")

        result = pd.DataFrame(all_rows, columns=columns)
        required_columns = [
            "TRADEDATE",
            "SECID",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOLUME",
            "VALUE",
        ]
        missing_columns = [
            col for col in required_columns if col not in result.columns
        ]
        if missing_columns:
            raise RuntimeError(
                f"Для {ticker} отсутствуют столбцы: {missing_columns}"
            )

        result = result[required_columns].copy()
        result["TRADEDATE"] = pd.to_datetime(result["TRADEDATE"])

        numeric_columns = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]
        for col in numeric_columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        result.dropna(
            subset=["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE"], inplace=True
        )
        result["COMPANY_NAME"] = self.companies_info[ticker]["name"]
        result["SUBSECTOR"] = self.companies_info[ticker]["subsector"]
        result.drop_duplicates(subset=["TRADEDATE", "SECID"], keep="last", inplace=True)
        return result.sort_values("TRADEDATE")

    def fetch_brent(self) -> pd.DataFrame:
        """Загружает ежедневные котировки закрытия фьючерса Brent (BZ=F)."""
        raw = yf.download(
            "BZ=F",
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=self.request_timeout,
        )
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance не вернул котировки Brent")

        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close", "BZ=F") in raw.columns:
                close = raw[("Close", "BZ=F")]
            else:
                close = raw.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            if "Close" not in raw.columns:
                raise RuntimeError("В данных Brent отсутствует столбец Close")
            close = raw["Close"]

        result = close.rename("BRENT_USD").to_frame().reset_index()
        date_column = "Date" if "Date" in result.columns else result.columns[0]
        result.rename(columns={date_column: "TRADEDATE"}, inplace=True)

        result["TRADEDATE"] = (
            pd.to_datetime(result["TRADEDATE"]).dt.tz_localize(None).dt.normalize()
        )
        result["BRENT_USD"] = pd.to_numeric(result["BRENT_USD"], errors="coerce").round(2)
        result.dropna(subset=["TRADEDATE", "BRENT_USD"], inplace=True)
        result.drop_duplicates("TRADEDATE", keep="last", inplace=True)

        if result.empty:
            raise RuntimeError("После обработки Brent нет данных")

        return result.sort_values("TRADEDATE")

    def calculate_stock_indicators(self, stocks: pd.DataFrame) -> pd.DataFrame:
        """Рассчитывает дневную доходность, SMA и отношение цены к SMA."""
        stocks = stocks.sort_values(["SECID", "TRADEDATE"]).copy()
        stocks["DAILY_RETURN_PCT"] = (
            stocks.groupby("SECID")["CLOSE"].pct_change().mul(100).round(3).fillna(0)
        )
        stocks["SMA_20"] = stocks.groupby("SECID")["CLOSE"].transform(
            lambda values: values.rolling(
                window=self.sma_window, min_periods=self.sma_window
            ).mean()
        )
        stocks["PRICE_SMA_RATIO"] = stocks["CLOSE"] / stocks["SMA_20"]
        return stocks

    def calculate_recommendations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Формирует сигналы Buy/Hold/Sell по скользящим квантилям BRENT_RUB."""
        data = data.sort_values(["SECID", "TRADEDATE"]).copy()

        macro = (
            data[["TRADEDATE", "BRENT_RUB"]]
            .drop_duplicates("TRADEDATE")
            .sort_values("TRADEDATE")
            .copy()
        )
        macro["BRENT_SELL_LIMIT"] = (
            macro["BRENT_RUB"]
            .rolling(window=self.brent_window, min_periods=self.brent_min_periods)
            .quantile(self.brent_sell_quantile)
            .shift(1)
        )
        macro["BRENT_BUY_LIMIT"] = (
            macro["BRENT_RUB"]
            .rolling(window=self.brent_window, min_periods=self.brent_min_periods)
            .quantile(self.brent_buy_quantile)
            .shift(1)
        )

        data = data.merge(
            macro[["TRADEDATE", "BRENT_SELL_LIMIT", "BRENT_BUY_LIMIT"]],
            on="TRADEDATE",
            how="left",
            validate="many_to_one",
        )

        buy_condition = (data["PRICE_SMA_RATIO"] < self.buy_ratio) & (
            data["BRENT_RUB"] >= data["BRENT_BUY_LIMIT"]
        )
        sell_condition = (data["PRICE_SMA_RATIO"] > self.sell_ratio) | (
            data["BRENT_RUB"] <= data["BRENT_SELL_LIMIT"]
        )

        data["RECOMMENDATION"] = np.select(
            [sell_condition, buy_condition],
            ["Продавать (Sell)", "Покупать (Buy)"],
            default="Держать (Hold)",
        )
        return data

    def build_dataset(self) -> pd.DataFrame:
        """Выполняет полный цикл сбора, объединения и расчёта признаков."""
        usd_code = self.get_cbr_currency_code("USD")
        cny_code = self.get_cbr_currency_code("CNY")

        usd_data = self.fetch_cbr_dynamic(usd_code, "USD_RUB")
        cny_data = self.fetch_cbr_dynamic(cny_code, "CNY_RUB")
        brent_data = self.fetch_brent()

        stock_frames = [
            self.fetch_moex_history(ticker) for ticker in self.companies_info
        ]
        stocks_data = self.calculate_stock_indicators(
            pd.concat(stock_frames, ignore_index=True)
        )

        macro_data = brent_data.merge(
            usd_data, on="TRADEDATE", how="inner", validate="one_to_one"
        ).merge(cny_data, on="TRADEDATE", how="inner", validate="one_to_one")

        final_data = stocks_data.merge(
            macro_data, on="TRADEDATE", how="inner", validate="many_to_one"
        )
        if final_data.empty:
            raise RuntimeError("После объединения источников не осталось общих дат")

        final_data = final_data.sort_values(["SECID", "TRADEDATE"]).drop_duplicates(
            ["TRADEDATE", "SECID"], keep="last"
        )
        final_data["BRENT_RUB"] = (
            final_data["BRENT_USD"] * final_data["USD_RUB"]
        ).round(2)
        final_data = self.calculate_recommendations(final_data).reset_index(drop=True)
        final_data.insert(0, "record_id", np.arange(1, len(final_data) + 1))

        output_columns = [
            "record_id",
            "TRADEDATE",
            "SECID",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOLUME",
            "VALUE",
            "COMPANY_NAME",
            "SUBSECTOR",
            "BRENT_USD",
            "USD_RUB",
            "CNY_RUB",
            "BRENT_RUB",
            "DAILY_RETURN_PCT",
            "RECOMMENDATION",
        ]
        export_data = final_data[output_columns].copy()
        export_data["TRADEDATE"] = pd.to_datetime(export_data["TRADEDATE"]).dt.strftime(
            "%Y-%m-%d"
        )
        export_data.to_csv(self.output_file, index=False, encoding="utf-8-sig")
        return final_data

    def print_report_tables(self, data: pd.DataFrame) -> None:
        """Печатает сводные таблицы описательной статистики для отчёта."""
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1400)
        unique_dates = data.drop_duplicates("TRADEDATE")

        print("\n" + "=" * 90)
        print("СВОДКА ДАННЫХ ДЛЯ ОТЧЕТА")
        print("=" * 90)
        print(f"Период запроса: {self.start_date} — {self.end_date}")
        print(
            f"Фактический период: {data['TRADEDATE'].min().date()} — "
            f"{data['TRADEDATE'].max().date()}"
        )
        print(f"Количество строк: {len(data)}")
        print(f"Уникальных торговых дат: {data['TRADEDATE'].nunique()}")
        print(f"Количество столбцов: {len(data.columns)}")
        print(f"Количество пропусков: {int(data.isna().sum().sum())}")
        print(f"Количество полных дубликатов: {int(data.duplicated().sum())}")

        macro_columns = ["BRENT_USD", "USD_RUB", "CNY_RUB", "BRENT_RUB"]
        print("\nТаблица 1.3 — Статистика макропоказателей")
        macro_stats = pd.DataFrame(
            {
                "Минимум": unique_dates[macro_columns].min(),
                "Максимум": unique_dates[macro_columns].max(),
                "Среднее": unique_dates[macro_columns].mean(),
                "Медиана": unique_dates[macro_columns].median(),
                "Стандартное отклонение": unique_dates[macro_columns].std(),
            }
        ).round(2)
        print(macro_stats)

        company_names = {
            "LKOH": "ЛУКОЙЛ",
            "ROSN": "Роснефть",
            "GAZP": "Газпром",
            "NVTK": "НОВАТЭК",
            "TATN": "Татнефть",
            "SIBN": "Газпром нефть",
        }
        for ticker, company_name in company_names.items():
            subset = data[data["SECID"] == ticker]
            stats = pd.DataFrame(
                {
                    "Цена CLOSE, руб.": subset["CLOSE"].agg(
                        ["min", "max", "mean", "median", "var", "std"]
                    ),
                    "Доходность DAILY_RETURN_PCT, %": subset[
                        "DAILY_RETURN_PCT"
                    ].agg(["min", "max", "mean", "median", "var", "std"]),
                }
            ).round(3)
            stats.index = [
                "Минимум",
                "Максимум",
                "Среднее",
                "Медиана",
                "Дисперсия",
                "Стандартное отклонение",
            ]
            print(f"\nСтатистика {ticker} — {company_name}:")
            print(stats)

        print("\nТаблица 1.10 — Распределение по тикерам")
        ticker_counts = data["SECID"].value_counts()
        ticker_table = pd.DataFrame(
            {
                "Абсолютная частота": ticker_counts,
                "Доля, %": (
                    data["SECID"].value_counts(normalize=True) * 100
                ).round(2),
            }
        )
        print(ticker_table.sort_index())

        print("\nТаблица 1.11 — Распределение по подотраслям")
        subsector_table = pd.DataFrame(
            {
                "Абсолютная частота": data["SUBSECTOR"].value_counts(),
                "Доля, %": (
                    data["SUBSECTOR"].value_counts(normalize=True) * 100
                ).round(2),
            }
        )
        print(subsector_table)

        print("\nТаблица 1.12 — Распределение рекомендаций")
        recommendation_table = pd.DataFrame(
            {
                "Абсолютная частота": data["RECOMMENDATION"].value_counts(),
                "Доля, %": (
                    data["RECOMMENDATION"].value_counts(normalize=True) * 100
                ).round(2),
            }
        )
        print(recommendation_table)

        print("\nРаспределение рекомендаций по компаниям")
        company_recommendations = (
            pd.crosstab(data["SECID"], data["RECOMMENDATION"], normalize="index")
            .mul(100)
            .round(2)
        )
        print(company_recommendations)


if __name__ == "__main__":
    collector = MarketDataCollector()
    dataset = collector.build_dataset()
    print(f"\nДатасет сохранён: {collector.output_file}")
    collector.print_report_tables(dataset)