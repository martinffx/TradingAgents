"""
Market data models and type definitions for technical analysis.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Literal

from proto.message import Optional
from pydantic import BaseModel

# Proper type definitions to eliminate Any types
IndicatorParamValue = int | float | str | bool
InputSpec = Literal["close", "ohlc", "ohlcv", "hl"]
OutputSpec = Literal["single", "double", "triple"]
ParamRanges = dict[str, tuple[int | float, int | float]]


class IndicatorConfig(BaseModel):
    """Configuration for technical indicators with proper typing."""

    name: str
    parameters: dict[str, IndicatorParamValue]  # No more Any
    input_types: list[InputSpec]  # Specific requirements
    output_format: OutputSpec  # Precise specification
    param_ranges: ParamRanges  # Type-safe validation
    default_params: dict[str, IndicatorParamValue]
    talib_function: str  # Direct TA-Lib function name
    description: str


class TechnicalAnalysisError(Exception):
    """Clear, actionable error messages for TA-Lib issues."""

    pass


class DataQuality(Enum):
    """Data quality levels for market data."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TechnicalIndicatorData(BaseModel):
    """Technical indicator data point with proper typing."""

    date: str
    value: float | dict[str, float]  # No more Any
    indicator_type: str
    parameters: dict[str, IndicatorParamValue]  # Parameter context
    confidence: float = 0.0  # Signal confidence
    source: str = "talib"  # Always TA-Lib


class MarketDataContext(BaseModel):
    """Market data context for trading analysis."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    price_data: list[dict[str, Any]]
    technical_indicators: dict[str, list[TechnicalIndicatorData]]
    metadata: dict[str, Any]


class TAReportContext(BaseModel):
    """Technical Analysis Report context with enhanced configuration."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    indicator: str
    indicator_data: list[TechnicalIndicatorData]
    analysis_summary: str
    signal_strength: float  # -1.0 to 1.0
    recommendation: str  # "BUY", "SELL", "HOLD"
    indicator_config: IndicatorConfig  # Full config used
    parameter_summary: str  # Human-readable params
    metadata: dict[str, IndicatorParamValue]  # Properly typed


class PriceDataContext(BaseModel):
    """Price Data context for historical price information."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    price_data: list[dict[str, Any]]
    latest_price: float
    price_change: float
    price_change_percent: float
    volume_info: dict[str, Any]
    metadata: dict[str, Any]


# Fundamental Data Models
class FinancialRatio(BaseModel):
    """Financial ratio with calculation metadata."""

    name: str
    value: float | None
    formula: str
    category: str  # "profitability", "liquidity", "leverage", "efficiency"
    interpretation: str


class BalanceSheetData(BaseModel):
    """Balance sheet line items."""

    date: str
    total_assets: float | None = None
    current_assets: float | None = None
    cash_and_equivalents: float | None = None
    accounts_receivable: float | None = None
    inventory: float | None = None
    total_liabilities: float | None = None
    current_liabilities: float | None = None
    accounts_payable: float | None = None
    short_term_debt: float | None = None
    long_term_debt: float | None = None
    total_equity: float | None = None
    retained_earnings: float | None = None


class IncomeStatementData(BaseModel):
    """Income statement line items."""

    date: str
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    ebitda: float | None = None
    cost_of_revenue: float | None = None
    operating_expenses: float | None = None
    interest_expense: float | None = None
    tax_expense: float | None = None
    shares_outstanding: float | None = None
    eps: float | None = None


class CashFlowData(BaseModel):
    """Cash flow statement line items."""

    date: str
    operating_cash_flow: float | None = None
    investing_cash_flow: float | None = None
    financing_cash_flow: float | None = None
    free_cash_flow: float | None = None
    capital_expenditures: float | None = None
    dividends_paid: float | None = None
    stock_repurchases: float | None = None


class BalanceSheetContext(BaseModel):
    """Balance sheet context for fundamental analysis."""

    symbol: str
    start_date: date
    end_date: date
    balance_sheet_data: list[BalanceSheetData]
    key_ratios: list[FinancialRatio]
    data_quality: DataQuality
    source: str
    metadata: dict[str, str]


class IncomeStatementContext(BaseModel):
    """Income statement context for fundamental analysis."""

    symbol: str
    start_date: date
    end_date: date
    income_statement_data: list[IncomeStatementData]
    key_ratios: list[FinancialRatio]
    data_quality: DataQuality
    source: str
    metadata: dict[str, Any]


class CashFlowContext(BaseModel):
    """Cash flow context for fundamental analysis."""

    symbol: str
    start_date: date
    end_date: date
    cash_flow_data: list[CashFlowData]
    key_ratios: list[FinancialRatio]
    data_quality: DataQuality
    source: str
    metadata: dict[str, Any]


class FundamentalContext(BaseModel):
    """Comprehensive fundamental analysis context."""

    symbol: str
    start_date: date
    end_date: date
    balance_sheet: Optional[BalanceSheetContext] = None
    income_statement: Optional[IncomeStatementContext] = None
    cash_flow: Optional[CashFlowContext] = None
    comprehensive_ratios: Optional[list[FinancialRatio]] = None
    valuation_metrics: Optional[dict[str, float | None]] = None
    financial_health_score: float = 0  # 0-100 composite score
    data_quality: DataQuality = DataQuality.LOW
    source: Optional[str] = None
    metadata: Optional[dict[str, str]] = None


# Reported Financials Models (for Finnhub API responses)
@dataclass
class FinancialLineItem:
    """Individual financial statement line item from reported financials."""

    concept: str
    unit: str
    label: str
    value: float | int


@dataclass
class ReportedFinancialsData:
    """Financial statements data containing balance sheet, income statement, and cash flow."""

    bs: list[FinancialLineItem]  # Balance Sheet
    ic: list[FinancialLineItem]  # Income Statement
    cf: list[FinancialLineItem]  # Cash Flow Statement


@dataclass
class ReportedFinancialsResponse:
    """Complete response from Finnhub reported financials API."""

    start_date: str
    end_date: str
    year: int
    quarter: int
    access_number: str
    data: ReportedFinancialsData

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReportedFinancialsResponse":
        """Create ReportedFinancialsResponse from API response dictionary."""
        financial_data = data.get("data", {})

        # Convert line items for each statement type
        bs_items = [
            FinancialLineItem(
                concept=item["concept"],
                unit=item["unit"],
                label=item["label"],
                value=item["value"],
            )
            for item in financial_data.get("bs", [])
        ]

        ic_items = [
            FinancialLineItem(
                concept=item["concept"],
                unit=item["unit"],
                label=item["label"],
                value=item["value"],
            )
            for item in financial_data.get("ic", [])
        ]

        cf_items = [
            FinancialLineItem(
                concept=item["concept"],
                unit=item["unit"],
                label=item["label"],
                value=item["value"],
            )
            for item in financial_data.get("cf", [])
        ]

        return cls(
            start_date=data["start_date"],
            end_date=data["end_date"],
            year=data["year"],
            quarter=data["quarter"],
            access_number=data["access_number"],
            data=ReportedFinancialsData(bs=bs_items, ic=ic_items, cf=cf_items),
        )


# Insider Transactions Models
@dataclass
class InsiderTransaction:
    """Individual insider transaction record."""

    name: str
    share: int
    change: int
    filing_date: str
    transaction_date: str
    transaction_code: str
    transaction_price: float


@dataclass
class InsiderTransactionsResponse:
    """Complete response from Finnhub insider transactions API."""

    data: list[InsiderTransaction]
    symbol: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsiderTransactionsResponse":
        """Create InsiderTransactionsResponse from API response dictionary."""
        transactions = [
            InsiderTransaction(
                name=item["name"],
                share=item["share"],
                change=item["change"],
                filing_date=item["filingDate"],
                transaction_date=item["transactionDate"],
                transaction_code=item["transactionCode"],
                transaction_price=item["transactionPrice"],
            )
            for item in data.get("data", [])
        ]

        return cls(data=transactions, symbol=data["symbol"])


# Insider Sentiment Models
@dataclass
class InsiderSentimentData:
    """Individual insider sentiment data point."""

    symbol: str
    year: int
    month: int
    change: int
    mspr: float  # Monthly Share Purchase Ratio


@dataclass
class InsiderSentimentResponse:
    """Complete response from Finnhub insider sentiment API."""

    data: list[InsiderSentimentData]
    symbol: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsiderSentimentResponse":
        """Create InsiderSentimentResponse from API response dictionary."""
        sentiment_data = [
            InsiderSentimentData(
                symbol=item["symbol"],
                year=item["year"],
                month=item["month"],
                change=item["change"],
                mspr=item["mspr"],
            )
            for item in data.get("data", [])
        ]

        return cls(data=sentiment_data, symbol=data["symbol"])


# Company Profile Models
@dataclass
class CompanyProfile:
    """Company profile information from Finnhub."""

    country: str
    currency: str
    exchange: str
    ipo: str
    market_capitalization: float
    name: str
    phone: str
    share_outstanding: float
    ticker: str
    weburl: str
    logo: str
    finnhub_industry: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompanyProfile":
        """Create CompanyProfile from API response dictionary."""
        return cls(
            country=data.get("country", ""),
            currency=data.get("currency", ""),
            exchange=data.get("exchange", ""),
            ipo=data.get("ipo", ""),
            market_capitalization=data.get("marketCapitalization", 0.0),
            name=data.get("name", ""),
            phone=data.get("phone", ""),
            share_outstanding=data.get("shareOutstanding", 0.0),
            ticker=data.get("ticker", ""),
            weburl=data.get("weburl", ""),
            logo=data.get("logo", ""),
            finnhub_industry=data.get("finnhubIndustry", ""),
        )


# Complete indicator definitions for 20 professional indicators
INDICATOR_DEFINITIONS = {
    # Momentum Indicators (7)
    "RSI": {
        "talib_function": "talib.RSI",
        "input_types": ["close"],
        "default_params": {"timeperiod": 14},
        "param_ranges": {"timeperiod": (2, 100)},
        "output_format": "single",
        "description": "Relative Strength Index",
    },
    "MACD": {
        "talib_function": "talib.MACD",
        "input_types": ["close"],
        "default_params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "param_ranges": {
            "fastperiod": (2, 50),
            "slowperiod": (10, 200),
            "signalperiod": (2, 50),
        },
        "output_format": "triple",
        "description": "Moving Average Convergence Divergence",
    },
    "STOCH": {
        "talib_function": "talib.STOCH",
        "input_types": ["ohlc"],
        "default_params": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
        "param_ranges": {
            "fastk_period": (1, 100),
            "slowk_period": (1, 50),
            "slowd_period": (1, 50),
        },
        "output_format": "double",
        "description": "Stochastic Oscillator",
    },
    "WILLR": {
        "talib_function": "talib.WILLR",
        "input_types": ["ohlc"],
        "default_params": {"timeperiod": 14},
        "param_ranges": {"timeperiod": (2, 100)},
        "output_format": "single",
        "description": "Williams %R",
    },
    "CCI": {
        "talib_function": "talib.CCI",
        "input_types": ["ohlc"],
        "default_params": {"timeperiod": 20},
        "param_ranges": {"timeperiod": (2, 100)},
        "output_format": "single",
        "description": "Commodity Channel Index",
    },
    "ROC": {
        "talib_function": "talib.ROC",
        "input_types": ["close"],
        "default_params": {"timeperiod": 12},
        "param_ranges": {"timeperiod": (1, 100)},
        "output_format": "single",
        "description": "Rate of Change",
    },
    "MFI": {
        "talib_function": "talib.MFI",
        "input_types": ["ohlcv"],
        "default_params": {"timeperiod": 14},
        "param_ranges": {"timeperiod": (2, 100)},
        "output_format": "single",
        "description": "Money Flow Index",
    },
    # Trend Indicators (7)
    "SMA": {
        "talib_function": "talib.SMA",
        "input_types": ["close"],
        "default_params": {"timeperiod": 20},
        "param_ranges": {"timeperiod": (2, 200)},
        "output_format": "single",
        "description": "Simple Moving Average",
    },
    "EMA": {
        "talib_function": "talib.EMA",
        "input_types": ["close"],
        "default_params": {"timeperiod": 20},
        "param_ranges": {"timeperiod": (2, 200)},
        "output_format": "single",
        "description": "Exponential Moving Average",
    },
    "BBANDS": {
        "talib_function": "talib.BBANDS",
        "input_types": ["close"],
        "default_params": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
        "param_ranges": {
            "timeperiod": (2, 100),
            "nbdevup": (0.1, 5.0),
            "nbdevdn": (0.1, 5.0),
        },
        "output_format": "triple",
        "description": "Bollinger Bands",
    },
    "SAR": {
        "talib_function": "talib.SAR",
        "input_types": ["ohlc"],
        "default_params": {"acceleration": 0.02, "maximum": 0.2},
        "param_ranges": {"acceleration": (0.01, 0.1), "maximum": (0.1, 1.0)},
        "output_format": "single",
        "description": "Parabolic SAR",
    },
    "ADX": {
        "talib_function": "talib.ADX",
        "input_types": ["ohlc"],
        "default_params": {"timeperiod": 14},
        "param_ranges": {"timeperiod": (2, 100)},
        "output_format": "single",
        "description": "Average Directional Index",
    },
    "AROON": {
        "talib_function": "talib.AROON",
        "input_types": ["hl"],
        "default_params": {"timeperiod": 25},
        "param_ranges": {"timeperiod": (2, 100)},
        "output_format": "double",
        "description": "Aroon Oscillator",
    },
    "TEMA": {
        "talib_function": "talib.TEMA",
        "input_types": ["close"],
        "default_params": {"timeperiod": 20},
        "param_ranges": {"timeperiod": (2, 200)},
        "output_format": "single",
        "description": "Triple Exponential Moving Average",
    },
    # Volume Indicators (3)
    "OBV": {
        "talib_function": "talib.OBV",
        "input_types": ["ohlcv"],
        "default_params": {},
        "param_ranges": {},
        "output_format": "single",
        "description": "On Balance Volume",
    },
    "AD": {
        "talib_function": "talib.AD",
        "input_types": ["ohlcv"],
        "default_params": {},
        "param_ranges": {},
        "output_format": "single",
        "description": "Accumulation/Distribution",
    },
    "ADOSC": {
        "talib_function": "talib.ADOSC",
        "input_types": ["ohlcv"],
        "default_params": {"fastperiod": 3, "slowperiod": 10},
        "param_ranges": {"fastperiod": (2, 50), "slowperiod": (5, 100)},
        "output_format": "single",
        "description": "Accumulation/Distribution Oscillator",
    },
    # Volatility Indicators (3)
    "ATR": {
        "talib_function": "talib.ATR",
        "input_types": ["ohlc"],
        "default_params": {"timeperiod": 14},
        "param_ranges": {"timeperiod": (1, 100)},
        "output_format": "single",
        "description": "Average True Range",
    },
    "NATR": {
        "talib_function": "talib.NATR",
        "input_types": ["ohlc"],
        "default_params": {"timeperiod": 14},
        "param_ranges": {"timeperiod": (1, 100)},
        "output_format": "single",
        "description": "Normalized Average True Range",
    },
    "TRANGE": {
        "talib_function": "talib.TRANGE",
        "input_types": ["ohlc"],
        "default_params": {},
        "param_ranges": {},
        "output_format": "single",
        "description": "True Range",
    },
}


class IndicatorPresets:
    """Professional indicator presets for different trading styles."""

    @staticmethod
    def get_scalping_presets() -> dict[str, dict[str, IndicatorParamValue]]:
        """Fast scalping presets (1-5 minute timeframes)."""
        return {
            "RSI_SCALPING": {"timeperiod": 5},
            "MACD_SCALPING": {"fastperiod": 5, "slowperiod": 13, "signalperiod": 5},
            "STOCH_SCALPING": {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
            "EMA_SCALPING": {"timeperiod": 9},
            "BBANDS_TIGHT": {"timeperiod": 10, "nbdevup": 1.5, "nbdevdn": 1.5},
            "ATR_SCALPING": {"timeperiod": 5},
        }

    @staticmethod
    def get_day_trading_presets() -> dict[str, dict[str, IndicatorParamValue]]:
        """Day trading presets (5-60 minute timeframes)."""
        return {
            "RSI_DAY_TRADING": {"timeperiod": 14},
            "MACD_DAY_TRADING": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            "STOCH_DAY_TRADING": {
                "fastk_period": 14,
                "slowk_period": 3,
                "slowd_period": 3,
            },
            "EMA_DAY_TRADING": {"timeperiod": 20},
            "BBANDS_STANDARD": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
            "ADX_DAY_TRADING": {"timeperiod": 14},
            "ATR_DAY_TRADING": {"timeperiod": 14},
        }

    @staticmethod
    def get_swing_trading_presets() -> dict[str, dict[str, IndicatorParamValue]]:
        """Swing trading presets (daily timeframes)."""
        return {
            "RSI_SWING": {"timeperiod": 21},
            "MACD_SWING": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            "STOCH_SWING": {"fastk_period": 21, "slowk_period": 5, "slowd_period": 5},
            "SMA_SWING_SHORT": {"timeperiod": 50},
            "SMA_SWING_LONG": {"timeperiod": 200},
            "EMA_SWING": {"timeperiod": 50},
            "BBANDS_SWING": {"timeperiod": 20, "nbdevup": 2.5, "nbdevdn": 2.5},
            "ADX_SWING": {"timeperiod": 21},
            "AROON_SWING": {"timeperiod": 25},
        }

    @staticmethod
    def get_position_trading_presets() -> dict[str, dict[str, IndicatorParamValue]]:
        """Position trading presets (weekly/monthly timeframes)."""
        return {
            "RSI_POSITION": {"timeperiod": 30},
            "MACD_POSITION": {"fastperiod": 20, "slowperiod": 50, "signalperiod": 15},
            "SMA_POSITION_SHORT": {"timeperiod": 100},
            "SMA_POSITION_LONG": {"timeperiod": 300},
            "EMA_POSITION": {"timeperiod": 100},
            "BBANDS_POSITION": {"timeperiod": 50, "nbdevup": 3.0, "nbdevdn": 3.0},
            "ADX_POSITION": {"timeperiod": 30},
            "AROON_POSITION": {"timeperiod": 50},
        }

    @staticmethod
    def get_all_presets() -> dict[str, dict[str, IndicatorParamValue]]:
        """Get all available presets combined."""
        all_presets = {}
        all_presets.update(IndicatorPresets.get_scalping_presets())
        all_presets.update(IndicatorPresets.get_day_trading_presets())
        all_presets.update(IndicatorPresets.get_swing_trading_presets())
        all_presets.update(IndicatorPresets.get_position_trading_presets())
        return all_presets

    @staticmethod
    def get_preset_for_style(style: str) -> dict[str, dict[str, IndicatorParamValue]]:
        """Get presets for a specific trading style."""
        style_map = {
            "scalping": IndicatorPresets.get_scalping_presets(),
            "day_trading": IndicatorPresets.get_day_trading_presets(),
            "swing": IndicatorPresets.get_swing_trading_presets(),
            "position": IndicatorPresets.get_position_trading_presets(),
        }
        return style_map.get(style.lower(), {})
