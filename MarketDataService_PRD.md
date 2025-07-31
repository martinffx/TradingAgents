# Product Requirements Document: MarketDataService Completion

## Overview

Complete the `MarketDataService` to provide strongly-typed market data and technical indicators to trading agents using a local-first data strategy with gap detection and intelligent caching.

## Current State Analysis

### Issues to Fix
- **CRITICAL**: Service uses `BaseClient` inheritance but `YFinanceClient` exists and needs refactoring to FinnhubClient standard
- **CRITICAL**: Service calls client methods with string dates instead of date objects
- **CRITICAL**: Need to integrate `stockstats` library for technical analysis calculations instead of legacy utils
- **CRITICAL**: `MarketDataRepository` exists but missing service interface methods
- Missing strongly-typed interface between YFinanceClient and service
- YFinanceClient uses BaseClient inheritance and string dates (needs refactoring)
- No concrete gap detection logic
- Missing technical indicator data sufficiency validation

### What Works
- ✅ Local-first data strategy implementation (`_get_price_data_local_first`)
- ✅ Force refresh logic (`_fetch_and_cache_fresh_data`)
- ✅ `MarketDataContext` Pydantic model for agent consumption
- ✅ Error handling and metadata creation patterns
- ✅ `YFinanceClient` exists with yfinance SDK integration and comprehensive methods
- ✅ `MarketDataRepository` exists with CSV storage and pandas DataFrame operations
- ✅ Service structure ready for `stockstats` integration for technical analysis

## Technical Requirements

### 1. Strongly-Typed Interfaces

#### Client → Service Interface
```python
# YFinanceClient methods (to be refactored)
def get_historical_data(symbol: str, start_date: date, end_date: date) -> dict[str, Any]
def get_price_data(symbol: str, start_date: date, end_date: date) -> dict[str, Any]

# Technical analysis handled in service layer using stockstats
# No get_technical_indicator method needed in client - calculated from OHLCV data
```

#### Service → Repository Interface
```python
# MarketDataRepository methods (to be implemented)
def has_data_for_period(symbol: str, start_date: str, end_date: str) -> bool
def get_data(symbol: str, start_date: str, end_date: str) -> dict[str, Any]
def store_data(symbol: str, cache_data: dict, overwrite: bool) -> bool
def clear_data(symbol: str, start_date: str, end_date: str) -> bool
```

#### Service → Agent Interface
```python
# Service output (already defined)
def get_context(symbol: str, start_date: str, end_date: str, indicators: list[str], force_refresh: bool) -> MarketDataContext
```

### 2. Local-First Data Strategy

#### Flow
1. **Repository Lookup**: Check `MarketDataRepository.has_data_for_period()`
2. **Gap Detection**: Identify missing price data periods using `detect_market_gaps()`
3. **Data Sufficiency Check**: Ensure enough historical data for requested indicators
4. **Selective Fetching**: Fetch only missing data from `YFinanceClient`
5. **Cache Updates**: Store new data via `repository.store_data()`
6. **Context Assembly**: Return validated `MarketDataContext`

#### Gap Detection Implementation
```python
def detect_market_gaps(self, cached_dates: list[str], requested_start: str, requested_end: str) -> list[tuple[str, str]]:
    """
    Returns list of (start, end) tuples for missing periods.

    Example: If requesting 2024-01-01 to 2024-01-31 and cache has:
    - 2024-01-01 to 2024-01-10
    - 2024-01-20 to 2024-01-25
    Returns: [("2024-01-11", "2024-01-19"), ("2024-01-26", "2024-01-31")]

    Accounts for:
    - Weekends (Saturday/Sunday)
    - Market holidays
    - Continuous date ranges to minimize API calls
    """
    # Implementation should use pandas business day logic
```

#### Force Refresh Support
- `force_refresh=True` bypasses local data completely
- Clears existing cache before fetching fresh data
- Stores refreshed data with metadata indicating refresh

#### Cache Invalidation Strategy
- **Historical data is immutable**: Data older than yesterday never changes
- **Today's data needs updates**: During market hours, refresh every 15 minutes
- **After market close**: Today's data becomes immutable
```python
def is_data_stale(self, data_date: date, last_updated: datetime) -> bool:
    today = date.today()
    if data_date < today:
        return False  # Historical data never stale

    # For today's data, check if market is open and last update > 15 min
    if is_market_open() and (datetime.now() - last_updated).minutes > 15:
        return True
    return False
```

### 3. Date Object Conversion

#### Service Boundary Conversion
```python
# Service receives string dates from agents
def get_context(self, symbol: str, start_date: str, end_date: str, ...) -> MarketDataContext:
    # Validate date strings
    try:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

    # Check date order
    if end_dt < start_dt:
        raise ValueError(f"End date {end_date} is before start date {start_date}")

    # Expand date range for technical indicators
    expanded_start = self._calculate_lookback_start(start_dt, indicators)

    # Use date objects when calling YFinanceClient
    price_data = self.yfinance_client.get_historical_data(symbol, expanded_start, end_dt)

    # Calculate technical indicators using stockstats library
    technical_indicators = self._calculate_technical_indicators(price_data, indicators)
```

### 4. Technical Analysis with Stockstats

#### Data Sufficiency Validation
```python
# Minimum data points required for each indicator
INDICATOR_REQUIREMENTS = {
    "sma_20": 20,
    "sma_200": 200,
    "ema_12": 24,      # 2x for exponential smoothing
    "ema_200": 400,
    "rsi_14": 28,      # 2x period for warm-up
    "macd": 34,        # 26 + 8 for signal line
    "bb_upper": 20,    # Based on 20-period SMA
    "atr_14": 28,      # 2x period for accuracy
    "stochrsi_14": 42, # 3x period for double smoothing
}

def _calculate_lookback_start(self, start_date: date, indicators: list[str]) -> date:
    """Calculate how far back we need data to compute indicators accurately."""
    max_lookback = 0
    for indicator in indicators:
        lookback = INDICATOR_REQUIREMENTS.get(indicator, 0)
        max_lookback = max(max_lookback, lookback)

    # Add buffer for weekends/holidays
    business_days_back = max_lookback * 1.5
    return start_date - timedelta(days=int(business_days_back))

def _validate_data_sufficiency(self, data_points: int, indicators: list[str]) -> dict[str, bool]:
    """Check if we have enough data for each indicator."""
    return {
        indicator: data_points >= INDICATOR_REQUIREMENTS.get(indicator, 0)
        for indicator in indicators
    }
```

#### Stockstats Integration
```python
def _calculate_technical_indicators(self, price_data: list[dict], indicators: list[str]) -> dict[str, list[dict]]:
    """
    Calculate technical indicators using stockstats library.

    Args:
        price_data: OHLCV data from YFinanceClient
        indicators: List of requested indicators (e.g., ['rsi_14', 'macd', 'bb_upper', 'sma_20'])

    Returns:
        Dict mapping indicator names to time series data
    """
    import pandas as pd
    from stockstats import StockDataFrame

    # Convert price data to pandas DataFrame
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Check data sufficiency
    sufficiency = self._validate_data_sufficiency(len(df), indicators)

    # Create StockDataFrame for technical analysis
    sdf = StockDataFrame.retype(df)

    # Calculate requested indicators
    indicator_data = {}
    for indicator in indicators:
        if not sufficiency[indicator]:
            logger.warning(f"Insufficient data for {indicator}, need {INDICATOR_REQUIREMENTS[indicator]} points")
            indicator_data[indicator] = []
            continue

        try:
            if indicator in sdf.columns:
                values = sdf[indicator].dropna()
                indicator_data[indicator] = [
                    {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                    for idx, val in values.items()
                ]
        except Exception as e:
            logger.warning(f"Failed to calculate {indicator}: {e}")
            indicator_data[indicator] = []

    return indicator_data
```

### 5. Error Recovery and Partial Data

```python
def handle_partial_price_data(
    self,
    requested_start: str,
    requested_end: str,
    available_data: list[dict]
) -> MarketDataContext:
    """
    Handle cases where only partial date range is available.

    - If no data available: Raise exception
    - If partial data: Return what's available with metadata
    - Mark gaps in metadata
    """
    if not available_data:
        raise ValueError(f"No market data available for {symbol}")

    actual_start = min(d['date'] for d in available_data)
    actual_end = max(d['date'] for d in available_data)

    metadata = {
        "requested_period": {"start": requested_start, "end": requested_end},
        "actual_period": {"start": actual_start, "end": actual_end},
        "partial_data": actual_start > requested_start or actual_end < requested_end,
        "data_points": len(available_data)
    }

    # Return context with available data and metadata
```

### 6. Pydantic Validation

#### Context Structure
```python
@dataclass
class MarketDataContext(BaseModel):
    symbol: str
    period: dict[str, str]  # {"start": "2024-01-01", "end": "2024-01-31"}
    price_data: list[dict[str, Any]]  # OHLCV records
    technical_indicators: dict[str, list[TechnicalIndicatorData]]
    metadata: dict[str, Any]

    @validator('price_data')
    def validate_price_data(cls, v):
        # Ensure OHLCV fields present and valid
        required_fields = {'date', 'open', 'high', 'low', 'close', 'volume'}
        for record in v:
            if not all(field in record for field in required_fields):
                raise ValueError(f"Missing required OHLCV fields")
        return v
```

## Implementation Tasks

### Phase 1: Refactor YFinanceClient

1. **YFinanceClient Refactoring**
   - **Refactor existing** `tradingagents/clients/yfinance_client.py`
   - Remove BaseClient inheritance
   - Update all method signatures to accept `date` objects instead of strings
   - Keep all existing functionality intact
   - Example changes:
   ```python
   # Current (wrong)
   def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> dict[str, Any]:

   # Updated (correct)
   def get_historical_data(self, symbol: str, start_date: date, end_date: date) -> dict[str, Any]:
   ```

2. **Comprehensive Testing**
   - Update `tradingagents/clients/test_yfinance_client.py`
   - Test with date objects
   - Use pytest-vcr for HTTP interaction recording
   - Test error handling and edge cases

### Phase 2: Update MarketDataRepository

3. **Repository Interface Enhancement**
   - Update existing `tradingagents/repositories/market_data_repository.py`
   - Add missing service interface methods: `has_data_for_period()`, `get_data()`, `store_data()`, `clear_data()`
   - Maintain existing CSV/pandas functionality while adding service compatibility
   - Support gap detection and partial data scenarios

### Phase 3: Update MarketDataService

4. **Client Integration Fix**
   - Replace `BaseClient` dependency with `YFinanceClient`
   - File: `tradingagents/services/market_data_service.py:8, 26`
   - Update constructor to accept `yfinance_client: YFinanceClient`

5. **Date Conversion and Validation**
   - Add `date.fromisoformat()` conversion in service methods
   - Add date validation (format, order)
   - Update client calls to use date objects instead of strings
   - File: `tradingagents/services/market_data_service.py:151, 227`

6. **Technical Indicator Integration with Stockstats**
   - Implement `_calculate_technical_indicators()` method using `stockstats` library
   - Add `_calculate_lookback_start()` for data sufficiency
   - Add `_validate_data_sufficiency()` to check if enough data
   - Replace legacy `StockstatsUtils` integration with direct stockstats usage
   - File: `tradingagents/services/market_data_service.py:9, 43, 280-346`

### Phase 4: Type Safety & Validation

7. **Comprehensive Type Checking**
   - Run `mise run typecheck` - must pass with 0 errors
   - Validate all date object conversions
   - Ensure MarketDataContext compliance

8. **Enhanced Testing**
   - Update existing service tests for new YFinanceClient interface
   - Add gap detection test scenarios
   - Test technical indicator data sufficiency
   - Test partial data handling

## Testing Scenarios

### Integration Tests

1. **Gap Detection**
   - Test with empty cache (should fetch all)
   - Test with partial cache (should fetch only missing periods)
   - Test weekend/holiday handling

2. **Technical Indicator Sufficiency**
   - Test SMA_200 with only 100 days of data (should skip indicator)
   - Test RSI_14 with exactly 28 days (should calculate)
   - Test mixed indicators with varying data requirements

3. **Partial Data Recovery**
   - Test when API returns less data than requested
   - Test when some dates are missing (holidays)
   - Test metadata accuracy for partial data

4. **Date Handling**
   - Test invalid date formats
   - Test end_date < start_date
   - Test future dates
   - Test weekend date handling

5. **Cache Staleness**
   - Test historical data (should never refresh)
   - Test today's data during market hours (should refresh if > 15 min)
   - Test today's data after market close (should not refresh)

## Success Criteria

### Functional Requirements
- ✅ Service successfully calls refactored `YFinanceClient` with `date` objects
- ✅ Gap detection correctly identifies missing trading days
- ✅ Technical indicators validate data sufficiency before calculation
- ✅ Partial data scenarios handled gracefully
- ✅ Local-first strategy works: checks cache → identifies gaps → fetches missing → stores updates
- ✅ Returns properly validated `MarketDataContext` to agents
- ✅ Technical indicators calculated from OHLCV data using stockstats library
- ✅ Force refresh bypasses cache and refreshes data

### Technical Requirements
- ✅ Zero type checking errors: `mise run typecheck`
- ✅ Zero linting errors: `mise run lint`
- ✅ All existing tests pass with updated architecture
- ✅ No runtime errors with date conversions
- ✅ Proper error messages for validation failures

### Quality Requirements
- ✅ Strongly-typed interfaces between all components
- ✅ Official yfinance SDK and stockstats library usage
- ✅ Comprehensive error handling and logging
- ✅ Efficient caching with minimal API calls
- ✅ Clear separation of concerns between service, client, and repository

## Data Architecture

### YFinanceClient Response Format
```python
{
    "symbol": "AAPL",
    "period": {"start": "2024-01-01", "end": "2024-01-31"},
    "data": [
        {
            "date": "2024-01-02",  # Note: Jan 1 was a holiday
            "open": 150.0,
            "high": 155.0,
            "low": 149.0,
            "close": 154.0,
            "volume": 1000000,
            "adj_close": 154.0
        },
        ...
    ],
    "metadata": {
        "source": "yfinance",
        "retrieved_at": "2024-01-31T10:00:00Z",
        "data_quality": "HIGH",
        "missing_dates": ["2024-01-01", "2024-01-15"]  # Holidays
    }
}
```

### Technical Indicator Data Format
```python
# MarketDataContext.technical_indicators structure
{
    "rsi_14": [
        {"date": "2024-01-29", "value": 65.5},  # First valid after 28 days
        {"date": "2024-01-30", "value": 67.2},
        ...
    ],
    "sma_200": [],  # Empty if insufficient data
    "macd": [
        {"date": "2024-01-31", "value": {"macd": 2.1, "signal": 1.8, "histogram": 0.3}}
    ],
    "_metadata": {
        "indicators_calculated": ["rsi_14", "macd"],
        "indicators_skipped": {
            "sma_200": "Insufficient data: need 200 points, have 31"
        }
    }
}
```

## Dependencies

### Existing Components (Need Updates)
- ✅ `YFinanceClient` exists but needs refactoring (remove BaseClient, use date objects)
- ✅ `MarketDataRepository` exists with CSV storage but needs service interface methods
- ✅ Tests exist but need updates for new interfaces

### Required
- Official `yfinance` library for market data fetching
- `stockstats` library for technical analysis calculations
- `pandas` for date/time handling and business day calculations
- Working internet connection for live data fetching
- Writable data directory for repository storage

## Timeline

### Immediate (Phase 1)
- Refactor existing YFinanceClient to use date objects
- Remove BaseClient inheritance
- Update tests for new interface

### Phase 2-3
- Add service interface methods to MarketDataRepository
- Update MarketDataService to use refactored YFinanceClient
- Implement data sufficiency validation
- Integrate stockstats library for technical indicators

### Phase 4
- Comprehensive type checking and validation
- Integration testing with gap detection
- Performance optimization and caching efficiency

## Acceptance Criteria

### Must Have
1. **Type Safety**: Service passes `mise run typecheck` with zero errors
2. **Client Refactoring**: YFinanceClient uses date objects, no BaseClient
3. **Gap Detection**: Correctly identifies missing trading days
4. **Data Sufficiency**: Validates enough data for technical indicators
5. **Partial Data**: Service handles incomplete data gracefully
6. **Local-First**: Service checks repository before API calls
7. **Context Validation**: Returns valid `MarketDataContext` with Pydantic validation
8. **Technical Indicators**: Calculated using stockstats with proper validation

### Should Have
1. **Cache Efficiency**: Minimal redundant API calls to Yahoo Finance
2. **Force Refresh**: Complete cache bypass when requested
3. **Stale Data Handling**: Refresh today's data during market hours
4. **Clear Error Messages**: Informative errors for validation failures

### Nice to Have
1. **Performance Metrics**: Timing and cache hit rate logging
2. **Extended Indicators**: Support for 50+ technical indicators
3. **Real-time Data**: WebSocket integration for live prices
4. **Bulk Symbol Support**: Fetch multiple symbols efficiently

---

This PRD focuses on completing the `MarketDataService` as a strongly-typed, local-first data service that integrates OHLCV price data from a refactored `YFinanceClient` and calculates comprehensive technical indicators using the `stockstats` library, with robust gap detection and data sufficiency validation.
