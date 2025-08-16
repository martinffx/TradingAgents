"""
Market data service that provides structured market context.
"""

import logging
from datetime import datetime, timezone
from typing import Any, cast

import pandas as pd
import talib

from tradingagents.config import TradingAgentsConfig
from tradingagents.domains.marketdata.clients.yfinance_client import YFinanceClient
from tradingagents.domains.marketdata.models import (
    INDICATOR_DEFINITIONS,
    DataQuality,
    IndicatorConfig,
    IndicatorParamValue,
    IndicatorPresets,
    InputSpec,
    OutputSpec,
    ParamRanges,
    PriceDataContext,
    TAReportContext,
    TechnicalAnalysisError,
    TechnicalIndicatorData,
)
from tradingagents.domains.marketdata.repos.market_data_repository import (
    MarketDataRepository,
)

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for market data and technical indicators."""

    def __init__(
        self,
        yfin_client: YFinanceClient,
        repo: MarketDataRepository,
    ):
        """
        Initialize market data service.

        Args:
            yfin_client: Client for live market data
            repo: Repository for historical market data
        """
        self.yfin_client = yfin_client
        self.repo = repo

    @staticmethod
    def build(_config: TradingAgentsConfig):
        client = YFinanceClient()
        repo = MarketDataRepository("")
        return MarketDataService(client, repo)

    def get_market_data_context(
        self, symbol: str, start_date: str, end_date: str
    ) -> PriceDataContext:
        """
        Get focused price data context with key metrics.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            PriceDataContext: Focused price data context
        """
        try:
            # Convert string dates to date objects
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()

            # Get data from repository first
            df = self.repo.get_market_data_df(symbol, start_date_obj, end_date_obj)

            if df.empty:
                # No data in repository, try to fetch from client
                logger.info(f"No local data for {symbol}, fetching from client")
                client_data = self.yfin_client.get_data(symbol, start_date, end_date)
                price_data = client_data.get("data", [])

                # Convert to DataFrame and store in repository
                if price_data:
                    df_to_store = pd.DataFrame(price_data)
                    self.repo.store_marketdata(symbol, df_to_store)
                    df = df_to_store
            else:
                # Convert DataFrame to list of dictionaries
                price_data = df.to_dict("records")

            # Calculate metrics
            latest_price = 0.0
            price_change = 0.0
            price_change_percent = 0.0
            volume_info = {"average_volume": 0, "latest_volume": 0}

            if not df.empty and "Close" in df.columns:
                latest_price = float(df["Close"].iloc[-1])
                if len(df) > 1:
                    previous_price = float(df["Close"].iloc[-2])
                    price_change = latest_price - previous_price
                    price_change_percent = (
                        (price_change / previous_price) * 100
                        if previous_price != 0
                        else 0.0
                    )

                if "Volume" in df.columns:
                    volume_info = {
                        "average_volume": int(df["Volume"].mean()),
                        "latest_volume": int(df["Volume"].iloc[-1]),
                    }

            # Convert DataFrame back to list of dicts for price_data
            price_data = df.to_dict("records") if not df.empty else []

            # Assess data quality
            data_quality = DataQuality.HIGH if len(price_data) > 0 else DataQuality.LOW

            metadata = {
                "data_quality": data_quality.value,
                "service": "market_data",
                "record_count": len(price_data),
                "source": "repository" if not df.empty else "client",
                "retrieved_at": datetime.now(timezone.utc)
                .replace(tzinfo=None)
                .isoformat(),
            }

            return PriceDataContext(
                symbol=symbol,
                period={"start": start_date, "end": end_date},
                price_data=price_data,
                latest_price=latest_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                volume_info=volume_info,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error getting market data context for {symbol}: {e}")
            return PriceDataContext(
                symbol=symbol,
                period={"start": start_date, "end": end_date},
                price_data=[],
                latest_price=0.0,
                price_change=0.0,
                price_change_percent=0.0,
                volume_info={"average_volume": 0, "latest_volume": 0},
                metadata={
                    "data_quality": DataQuality.LOW.value,
                    "service": "market_data",
                    "error": str(e),
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            )

    def get_ta_report_context(
        self,
        symbol: str,
        indicator: str,
        start_date: str,
        end_date: str,
        custom_params: dict[str, IndicatorParamValue] | None = None,
    ) -> TAReportContext:
        """
        Get technical analysis report context for a specific indicator.

        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator name (e.g., 'rsi', 'macd', 'sma')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            TAReportContext: Focused technical analysis context
        """
        try:
            # Get price data first
            price_context = self.get_market_data_context(symbol, start_date, end_date)

            if not price_context.price_data:
                # Create empty indicator config for no data case
                no_data_config = IndicatorConfig(
                    name=indicator.upper(),
                    parameters={},
                    input_types=["close"],
                    output_format="single",
                    param_ranges={},
                    default_params={},
                    talib_function="",
                    description="",
                )

                return TAReportContext(
                    symbol=symbol,
                    period={"start": start_date, "end": end_date},
                    indicator=indicator,
                    indicator_data=[],
                    analysis_summary="No price data available for technical analysis",
                    signal_strength=0.0,
                    recommendation="HOLD",
                    indicator_config=no_data_config,
                    parameter_summary="",
                    metadata={
                        "data_quality": DataQuality.LOW.value,
                        "service": "technical_analysis",
                        "error": "no_price_data",
                    },
                )

            # Calculate technical indicator using TA-Lib
            indicator_data = self._calculate_indicator_talib(
                price_context.price_data, indicator, custom_params
            )

            # Generate analysis and recommendations
            signal_strength = self._calculate_signal_strength(indicator_data, indicator)
            recommendation = self._get_recommendation(signal_strength)
            analysis_summary = self._generate_analysis_summary(
                indicator, signal_strength, recommendation
            )

            # Create indicator config from the calculation
            definition = INDICATOR_DEFINITIONS.get(indicator.upper(), {})
            indicator_config = IndicatorConfig(
                name=indicator.upper(),
                parameters=indicator_data[0].parameters if indicator_data else {},
                input_types=cast(
                    "list[InputSpec]", definition.get("input_types", ["close"])
                ),
                output_format=cast(
                    "OutputSpec", definition.get("output_format", "single")
                ),
                param_ranges=cast("ParamRanges", definition.get("param_ranges", {})),
                default_params=cast(
                    "dict[str, IndicatorParamValue]",
                    definition.get("default_params", {}),
                ),
                talib_function=str(definition.get("talib_function", "")),
                description=str(definition.get("description", "")),
            )

            # Generate parameter summary
            params = indicator_data[0].parameters if indicator_data else {}
            parameter_summary = ", ".join([f"{k}={v}" for k, v in params.items()])

            return TAReportContext(
                symbol=symbol,
                period={"start": start_date, "end": end_date},
                indicator=indicator,
                indicator_data=indicator_data,
                analysis_summary=analysis_summary,
                signal_strength=signal_strength,
                recommendation=recommendation,
                indicator_config=indicator_config,
                parameter_summary=parameter_summary,
                metadata={
                    "data_quality": DataQuality.HIGH.value,
                    "service": "technical_analysis",
                    "indicator_count": len(indicator_data),
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error getting TA report for {symbol} {indicator}: {e}")
            # Create empty indicator config for error case
            error_config = IndicatorConfig(
                name=indicator.upper(),
                parameters={},
                input_types=["close"],
                output_format="single",
                param_ranges={},
                default_params={},
                talib_function="",
                description="",
            )

            return TAReportContext(
                symbol=symbol,
                period={"start": start_date, "end": end_date},
                indicator=indicator,
                indicator_data=[],
                analysis_summary=f"Error calculating {indicator}: {str(e)}",
                signal_strength=0.0,
                recommendation="HOLD",
                indicator_config=error_config,
                parameter_summary="",
                metadata={
                    "data_quality": DataQuality.LOW.value,
                    "service": "technical_analysis",
                    "error": str(e),
                },
            )

    def _validate_parameters(
        self, indicator: str, params: dict[str, IndicatorParamValue]
    ) -> None:
        """Validate indicator parameters against defined ranges."""
        if indicator.upper() not in INDICATOR_DEFINITIONS:
            raise TechnicalAnalysisError(f"Unknown indicator: {indicator}")

        definition = INDICATOR_DEFINITIONS[indicator.upper()]
        param_ranges = cast("ParamRanges", definition.get("param_ranges", {}))

        for param_name, value in params.items():
            if param_name in param_ranges:
                range_tuple = param_ranges[param_name]
                if isinstance(range_tuple, tuple) and len(range_tuple) == 2:
                    min_val, max_val = range_tuple
                    if not isinstance(value, int | float):
                        raise TechnicalAnalysisError(
                            f"Parameter {param_name} must be numeric"
                        )
                    if not (min_val <= value <= max_val):
                        raise TechnicalAnalysisError(
                            f"Parameter {param_name}={value} out of range [{min_val}, {max_val}]"
                        )

    def _prepare_price_arrays(
        self, price_data: list[dict[str, Any]], input_types: list[InputSpec]
    ) -> dict[str, Any]:
        """Prepare price arrays for TA-Lib functions."""
        if not price_data:
            raise TechnicalAnalysisError("No price data provided")

        df = pd.DataFrame(price_data)
        required_columns = []

        for input_type in input_types:
            if input_type == "close":
                required_columns.extend(["Close"])
            elif input_type == "ohlc":
                required_columns.extend(["Open", "High", "Low", "Close"])
            elif input_type == "ohlcv":
                required_columns.extend(["Open", "High", "Low", "Close", "Volume"])
            elif input_type == "hl":
                required_columns.extend(["High", "Low"])

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise TechnicalAnalysisError(f"Missing required columns: {missing_columns}")

        # Convert to numpy arrays for TA-Lib
        arrays = {}
        if "Open" in df.columns:
            arrays["open"] = df["Open"].astype(float).values
        if "High" in df.columns:
            arrays["high"] = df["High"].astype(float).values
        if "Low" in df.columns:
            arrays["low"] = df["Low"].astype(float).values
        if "Close" in df.columns:
            arrays["close"] = df["Close"].astype(float).values
        if "Volume" in df.columns:
            arrays["volume"] = df["Volume"].astype(float).values

        arrays["dates"] = df["Date"].astype(str).values

        return arrays

    def _calculate_indicator_talib(
        self,
        price_data: list[dict[str, Any]],
        indicator: str,
        params: dict[str, IndicatorParamValue] | None = None,
    ) -> list[TechnicalIndicatorData]:
        """Calculate technical indicator using TA-Lib."""
        if not price_data:
            return []

        # Get indicator definition
        indicator_upper = indicator.upper()
        if indicator_upper not in INDICATOR_DEFINITIONS:
            raise TechnicalAnalysisError(f"Unknown indicator: {indicator}")

        definition = INDICATOR_DEFINITIONS[indicator_upper]

        # Use provided params or defaults
        final_params: dict[str, IndicatorParamValue]
        if params is None:
            final_params = cast(
                "dict[str, IndicatorParamValue]", definition["default_params"]
            )
        else:
            # Merge with defaults for missing parameters
            final_params = cast(
                "dict[str, IndicatorParamValue]", definition["default_params"]
            ).copy()
            final_params.update(params)

        # Validate parameters
        self._validate_parameters(indicator, final_params)

        # Prepare price arrays
        arrays = self._prepare_price_arrays(
            price_data, cast("list[InputSpec]", definition["input_types"])
        )

        # Get TA-Lib function
        talib_func_name = str(definition["talib_function"]).split(".")[
            -1
        ]  # Extract function name
        talib_func = getattr(talib, talib_func_name)

        # Prepare function arguments
        func_args = []
        func_kwargs = {}

        # Add required price arrays based on input types
        for input_type in list(definition["input_types"]):
            if input_type == "close":
                func_args.append(arrays["close"])
            elif input_type == "ohlc":
                func_args.extend([arrays["high"], arrays["low"], arrays["close"]])
            elif input_type == "ohlcv":
                func_args.extend(
                    [arrays["high"], arrays["low"], arrays["close"], arrays["volume"]]
                )
            elif input_type == "hl":
                func_args.extend([arrays["high"], arrays["low"]])

        # Add parameters as keyword arguments
        for param_name, param_value in final_params.items():
            func_kwargs[param_name] = param_value

        # Calculate indicator
        try:
            ta_result = talib_func(*func_args, **func_kwargs)
        except Exception as e:
            raise TechnicalAnalysisError(
                f"TA-Lib calculation failed for {indicator}: {str(e)}"
            ) from e

        # Process results based on output format
        result = []
        dates = arrays["dates"]
        output_format = str(definition["output_format"])

        if output_format == "single":
            # Single output array
            for _i, (date, value) in enumerate(zip(dates, ta_result, strict=False)):
                if not pd.isna(value):
                    result.append(
                        TechnicalIndicatorData(
                            date=date,
                            value=float(value),
                            indicator_type=indicator.lower(),
                            parameters=final_params,
                        )
                    )

        elif output_format == "double":
            # Two output arrays (e.g., STOCH, AROON)
            for _i, (date, val1, val2) in enumerate(
                zip(dates, ta_result[0], ta_result[1], strict=False)
            ):
                if not pd.isna(val1) and not pd.isna(val2):
                    # Name outputs based on indicator
                    if indicator_upper == "STOCH":
                        value_dict = {"slowk": float(val1), "slowd": float(val2)}
                    elif indicator_upper == "AROON":
                        value_dict = {"aroondown": float(val1), "aroonup": float(val2)}
                    else:
                        value_dict = {"output1": float(val1), "output2": float(val2)}

                    result.append(
                        TechnicalIndicatorData(
                            date=date,
                            value=value_dict,
                            indicator_type=indicator.lower(),
                            parameters=final_params,
                        )
                    )

        elif output_format == "triple":
            # Three output arrays (e.g., MACD, BBANDS)
            for _i, (date, val1, val2, val3) in enumerate(
                zip(dates, ta_result[0], ta_result[1], ta_result[2], strict=False)
            ):
                if not pd.isna(val1):
                    # Name outputs based on indicator
                    if indicator_upper == "MACD":
                        value_dict = {
                            "macd": float(val1),
                            "signal": float(val2) if not pd.isna(val2) else 0.0,
                            "histogram": float(val3) if not pd.isna(val3) else 0.0,
                        }
                    elif indicator_upper == "BBANDS":
                        value_dict = {
                            "upper": float(val1),
                            "middle": float(val2) if not pd.isna(val2) else 0.0,
                            "lower": float(val3) if not pd.isna(val3) else 0.0,
                        }
                    else:
                        value_dict = {
                            "output1": float(val1),
                            "output2": float(val2) if not pd.isna(val2) else 0.0,
                            "output3": float(val3) if not pd.isna(val3) else 0.0,
                        }

                    result.append(
                        TechnicalIndicatorData(
                            date=date,
                            value=value_dict,
                            indicator_type=indicator.lower(),
                            parameters=final_params,
                        )
                    )

        return result

    def calculate_indicator(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        indicator: str | dict[str, IndicatorParamValue],
        params: dict[str, IndicatorParamValue] | None = None,
    ) -> TAReportContext:
        """
        Three-tier API for technical indicator calculation.

        Usage:
        1. String: calculate_indicator("AAPL", "2024-01-01", "2024-01-31", "RSI")
        2. Preset: calculate_indicator("AAPL", "2024-01-01", "2024-01-31", "RSI_SCALPING")
        3. Custom: calculate_indicator("AAPL", "2024-01-01", "2024-01-31", "RSI", {"timeperiod": 21})

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            indicator: Indicator name (string), preset name, or custom config dict
            params: Optional custom parameters (for string indicators only)

        Returns:
            TAReportContext: Complete technical analysis context
        """
        if isinstance(indicator, dict):
            # Custom configuration provided as dict
            if "name" not in indicator:
                raise TechnicalAnalysisError(
                    "Custom indicator dict must contain 'name' field"
                )
            indicator_name = str(indicator["name"])  # Ensure it's a string
            custom_params = {k: v for k, v in indicator.items() if k != "name"}
            return self.get_ta_report_context(
                symbol, indicator_name, start_date, end_date, custom_params
            )

        # Check if it's a preset name
        all_presets = IndicatorPresets.get_all_presets()
        if indicator in all_presets:
            # Extract indicator name and parameters from preset
            preset_params = all_presets[indicator]
            # Determine base indicator from preset name
            for base_indicator in INDICATOR_DEFINITIONS:
                if indicator.startswith(base_indicator):
                    return self.get_ta_report_context(
                        symbol,
                        base_indicator.lower(),
                        start_date,
                        end_date,
                        preset_params,
                    )

            # If no match found, try to extract from preset name
            indicator_name = indicator.split("_")[0].lower()
            return self.get_ta_report_context(
                symbol, indicator_name, start_date, end_date, preset_params
            )

        # Regular indicator name (string)
        return self.get_ta_report_context(
            symbol, indicator, start_date, end_date, params
        )

    def get_available_indicators(self) -> dict[str, str]:
        """Get list of all available indicators with descriptions."""
        return {
            name: str(info["description"])
            for name, info in INDICATOR_DEFINITIONS.items()
        }

    def get_available_presets(
        self, style: str | None = None
    ) -> dict[str, dict[str, IndicatorParamValue]]:
        """
        Get available indicator presets.

        Args:
            style: Optional trading style filter ("scalping", "day_trading", "swing", "position")

        Returns:
            Dict of preset names to parameter configurations
        """
        if style:
            return IndicatorPresets.get_preset_for_style(style)
        return IndicatorPresets.get_all_presets()

    def get_indicator_info(self, indicator: str) -> IndicatorConfig:
        """
        Get detailed information about a specific indicator.

        Args:
            indicator: Indicator name

        Returns:
            IndicatorConfig with full indicator specifications
        """
        indicator_upper = indicator.upper()
        if indicator_upper not in INDICATOR_DEFINITIONS:
            raise TechnicalAnalysisError(f"Unknown indicator: {indicator}")

        definition = INDICATOR_DEFINITIONS[indicator_upper]
        return IndicatorConfig(
            name=indicator_upper,
            parameters=cast(
                "dict[str, IndicatorParamValue]", definition["default_params"]
            ),
            input_types=cast("list[InputSpec]", definition["input_types"]),
            output_format=cast("OutputSpec", definition["output_format"]),
            param_ranges=cast("ParamRanges", definition["param_ranges"]),
            default_params=cast(
                "dict[str, IndicatorParamValue]", definition["default_params"]
            ),
            talib_function=str(definition["talib_function"]),
            description=str(definition["description"]),
        )

    def _calculate_signal_strength(
        self, indicator_data: list[TechnicalIndicatorData], indicator: str
    ) -> float:
        """Calculate signal strength from indicator data."""
        if not indicator_data:
            return 0.0

        latest = indicator_data[-1]

        if indicator.lower() == "rsi":
            rsi_value = latest.value
            if isinstance(rsi_value, int | float):
                if rsi_value > 70:
                    return -0.8  # Overbought - sell signal
                elif rsi_value < 30:
                    return 0.8  # Oversold - buy signal
                else:
                    return (50 - rsi_value) / 50  # Normalized between -1 and 1

        elif indicator.lower() == "macd":
            if isinstance(latest.value, dict):
                macd_val = latest.value.get("macd", 0)
                signal_val = latest.value.get("signal", 0)
                if macd_val > signal_val:
                    return 0.6  # Bullish
                else:
                    return -0.6  # Bearish

        elif indicator.lower() == "sma":
            # Would need current price to compare with SMA
            return 0.0  # Neutral for now

        return 0.0

    def _get_recommendation(self, signal_strength: float) -> str:
        """Convert signal strength to recommendation."""
        if signal_strength > 0.5:
            return "BUY"
        elif signal_strength < -0.5:
            return "SELL"
        else:
            return "HOLD"

    def _generate_analysis_summary(
        self, indicator: str, signal_strength: float, recommendation: str
    ) -> str:
        """Generate human-readable analysis summary."""
        strength_desc = (
            "strong"
            if abs(signal_strength) > 0.7
            else "moderate"
            if abs(signal_strength) > 0.3
            else "weak"
        )
        direction = (
            "bullish"
            if signal_strength > 0
            else "bearish"
            if signal_strength < 0
            else "neutral"
        )

        return f"{indicator.upper()} indicator shows {strength_desc} {direction} signal. Signal strength: {signal_strength:.2f}. Recommendation: {recommendation}."

    def update_market_data(self, symbol: str, start_date: str, end_date: str):
        """
        Update market data by fetching fresh data from client and storing in repository.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        try:
            logger.info(
                f"Updating market data for {symbol} from {start_date} to {end_date}"
            )

            # Fetch fresh data from client
            client_data = self.yfin_client.get_data(symbol, start_date, end_date)
            price_data = client_data.get("data", [])

            if price_data:
                # Convert to DataFrame
                df = pd.DataFrame(price_data)

                # Store in repository
                self.repo.store_marketdata(symbol, df)
                logger.info(
                    f"Successfully stored {len(price_data)} records for {symbol}"
                )
            else:
                logger.warning(f"No data received for {symbol}")

        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
            raise
