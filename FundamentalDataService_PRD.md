# Product Requirements Document: FundamentalDataService Completion

## Overview

Complete the `FundamentalDataService` to provide strongly-typed fundamental financial data to trading agents using a local-first data strategy with gap detection and intelligent caching.

## Current State Analysis

### Issues to Fix
- **CRITICAL**: Service calls `FinnhubClient` methods with string dates but client expects `date` objects
- **CRITICAL**: References non-existent `self.simfin_client` instead of `self.finnhub_client`
- Missing strongly-typed interfaces between components
- Incomplete local-first strategy implementation
- No concrete gap detection logic
- Missing error recovery for partial data

### What Works
- ✅ `FinnhubClient` fully implemented with strict `date` object interface
- ✅ `FundamentalDataRepository` with dataclass-based storage
- ✅ `FundamentalContext` Pydantic model for agent consumption
- ✅ Basic service structure and error handling

## Technical Requirements

### 1. Strongly-Typed Interfaces

#### Client → Service Interface
```python
# FinnhubClient methods (already implemented)
def get_balance_sheet(symbol: str, frequency: str, report_date: date) -> dict[str, Any]
def get_income_statement(symbol: str, frequency: str, report_date: date) -> dict[str, Any]
def get_cash_flow(symbol: str, frequency: str, report_date: date) -> dict[str, Any]
```

#### Service → Repository Interface
```python
# Repository methods (already implemented)
def has_data_for_period(symbol: str, start_date: str, end_date: str, frequency: str) -> bool
def get_data(symbol: str, start_date: str, end_date: str, frequency: str) -> dict[str, Any]
def store_data(symbol: str, cache_data: dict, frequency: str, overwrite: bool) -> bool
def clear_data(symbol: str, start_date: str, end_date: str, frequency: str) -> bool
```

#### Service → Agent Interface
```python
# Service output (already defined)
def get_context(symbol: str, start_date: str, end_date: str, frequency: str, force_refresh: bool) -> FundamentalContext
```

### 2. Local-First Data Strategy

#### Flow
1. **Repository Lookup**: Check `FundamentalDataRepository.has_data_for_period()`
2. **Gap Detection**: Identify missing data periods using `detect_fundamental_gaps()`
3. **Selective Fetching**: Fetch only missing data from `FinnhubClient`
4. **Cache Updates**: Store new data via `repository.store_data()`
5. **Context Assembly**: Return validated `FundamentalContext`

#### Gap Detection Implementation
```python
def detect_fundamental_gaps(self, symbol: str, start_date: str, end_date: str, frequency: str) -> list[str]:
    """
    Returns list of report dates that need fetching.

    Example: If requesting quarterly from 2024-01-01 to 2024-12-31
    and cache has Q1 and Q3, returns ["2024-06-30", "2024-09-30", "2024-12-31"]

    For quarterly: Check for Q1 (Mar 31), Q2 (Jun 30), Q3 (Sep 30), Q4 (Dec 31)
    For annual: Check for fiscal year ends
    """
    # Implementation should:
    # 1. Get existing report dates from repository
    # 2. Calculate expected report dates in requested period
    # 3. Return difference between expected and existing
```

#### Force Refresh Support
- `force_refresh=True` bypasses local data completely
- Clears existing cache before fetching fresh data
- Stores refreshed data with metadata indicating refresh

#### Cache Invalidation Strategy
- **Fundamental data is immutable**: Once a report is filed, it doesn't change
- **No staleness checks needed**: Reports are valid indefinitely
- **Only fetch if missing**: Never re-fetch existing reports

### 3. Date Object Conversion

#### Service Boundary Conversion
```python
# Service receives string dates from agents
def get_context(self, symbol: str, start_date: str, end_date: str, ...) -> FundamentalContext:
    # Validate date strings
    try:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

    # Check date order
    if end_dt < start_dt:
        raise ValueError(f"End date {end_date} is before start date {start_date}")

    # Use date objects when calling FinnhubClient
    data = self.finnhub_client.get_balance_sheet(symbol, frequency, end_dt)
```

### 4. Error Recovery and Partial Data

```python
def handle_partial_statements(
    self,
    balance_sheet: dict | None,
    income_statement: dict | None,
    cash_flow: dict | None
) -> FundamentalContext:
    """
    Create context even if some statements are missing.

    - If all statements fail: Raise exception
    - If some statements succeed: Return partial context
    - Mark missing statements in metadata
    """
    metadata = {
        "has_balance_sheet": balance_sheet is not None,
        "has_income_statement": income_statement is not None,
        "has_cash_flow": cash_flow is not None,
        "partial_data": any(s is None for s in [balance_sheet, income_statement, cash_flow])
    }

    # Convert available statements to FinancialStatement objects
    # Return FundamentalContext with available data
```

### 5. Pydantic Validation

#### Context Structure
```python
@dataclass
class FundamentalContext(BaseModel):
    symbol: str
    period: dict[str, str]  # {"start": "2024-01-01", "end": "2024-01-31"}
    balance_sheet: FinancialStatement | None
    income_statement: FinancialStatement | None
    cash_flow: FinancialStatement | None
    key_ratios: dict[str, float]
    metadata: dict[str, Any]

    @validator('period')
    def validate_period(cls, v):
        # Ensure start and end dates are present and valid
        return v
```

## Implementation Tasks

### Phase 1: Fix Critical Issues

1. **Date Conversion Fix**
   - Add `date.fromisoformat()` conversion in service methods
   - Add date validation (format, order)
   - Update all `FinnhubClient` method calls to use `date` objects
   - File: `tradingagents/services/fundamental_data_service.py:153, 164, 175`

2. **Client Reference Fix**
   - Replace `self.simfin_client` with `self.finnhub_client`
   - File: `tradingagents/services/fundamental_data_service.py:375`

### Phase 2: Enhanced Local-First Strategy

3. **Gap Detection Logic**
   - Implement `detect_fundamental_gaps()` method
   - Calculate expected report dates based on frequency
   - Compare with cached data to find gaps
   - Handle fiscal year variations

4. **Partial Data Handling**
   - Implement `handle_partial_statements()` method
   - Continue processing if some statements succeed
   - Mark missing data in metadata
   - Only fail if all statements fail

### Phase 3: Type Safety & Validation

5. **Comprehensive Type Checking**
   - Run `mise run typecheck` - must pass with 0 errors
   - Validate all `date` object conversions
   - Ensure Pydantic model compliance

6. **Enhanced Testing**
   - Update existing tests for new date handling
   - Add gap detection test scenarios
   - Test partial data scenarios
   - Test force refresh behavior
   - Test date validation edge cases

## Testing Scenarios

### Integration Tests
1. **Gap Detection**
   - Test with empty cache (should fetch all)
   - Test with partial cache (should fetch only missing)
   - Test with complete cache (should fetch none)

2. **Partial Data Recovery**
   - Test when balance sheet API fails but others succeed
   - Test when only one statement type is available
   - Test when all APIs fail (should raise exception)

3. **Date Handling**
   - Test invalid date formats
   - Test end_date < start_date
   - Test boundary conditions (year start/end)

4. **Force Refresh**
   - Test that force_refresh=True clears cache
   - Test that new data is fetched and stored

## Success Criteria

### Functional Requirements
- ✅ Service successfully calls `FinnhubClient` with `date` objects
- ✅ Gap detection correctly identifies missing reports
- ✅ Partial data scenarios handled gracefully
- ✅ Local-first strategy works: checks cache → identifies gaps → fetches missing → stores updates
- ✅ Returns properly validated `FundamentalContext` to agents
- ✅ Force refresh bypasses cache and refreshes data

### Technical Requirements
- ✅ Zero type checking errors: `mise run typecheck`
- ✅ Zero linting errors: `mise run lint`
- ✅ All existing tests pass
- ✅ No runtime errors with date conversions
- ✅ Proper error messages for validation failures

### Quality Requirements
- ✅ Strongly-typed interfaces between all components
- ✅ Comprehensive error handling and logging
- ✅ Efficient caching with minimal API calls
- ✅ Clear separation of concerns between service, client, and repository

## Dependencies

### Completed
- ✅ `FinnhubClient` with `date` object interface
- ✅ `FundamentalDataRepository` with dataclass storage
- ✅ `FundamentalContext` Pydantic model

### Required
- Working `FinnhubClient` instance with valid API key
- Writable data directory for repository storage

## Timeline

### Immediate (Today)
- Fix critical date conversion and reference issues
- Implement basic gap detection
- Add date validation

### Next Steps
- Implement partial data handling
- Comprehensive testing
- Integration with agent workflows

## Acceptance Criteria

### Must Have
1. **Type Safety**: Service passes `mise run typecheck` with zero errors
2. **Client Integration**: All `FinnhubClient` calls use `date` objects correctly
3. **Gap Detection**: Correctly identifies missing report periods
4. **Partial Data**: Service returns partial context when some statements fail
5. **Local-First**: Service checks repository before API calls
6. **Context Validation**: Returns valid `FundamentalContext` with Pydantic validation
7. **Error Handling**: Graceful handling of API failures and missing data

### Should Have
1. **Cache Efficiency**: Minimal redundant API calls
2. **Force Refresh**: Complete cache bypass when requested
3. **Data Quality**: Metadata indicating data completeness
4. **Clear Error Messages**: Informative errors for date validation failures

### Nice to Have
1. **Performance Metrics**: Timing and cache hit rate logging
2. **Fiscal Year Handling**: Support for non-calendar fiscal years
3. **Bulk Operations**: Fetch multiple symbols efficiently

---

This PRD focuses on completing the `FundamentalDataService` as a strongly-typed, local-first data service that seamlessly integrates with the existing `FinnhubClient` and `FundamentalDataRepository` components while providing robust gap detection and partial data handling.
