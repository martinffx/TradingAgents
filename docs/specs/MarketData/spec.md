# MarketData Domain - PostgreSQL Migration Specification

## Context

**Product**: Multi-agent LLM financial trading framework that mirrors real-world trading firm dynamics for research-based market analysis and trading decisions.

**Domain**: MarketData (85% complete with CSV storage → migrate to PostgreSQL)

**Stack**: PostgreSQL + TimescaleDB + pgvectorscale + OpenRouter

**Current Status**: CSV-based storage in `./data/market_data/` with 85% functionality. Migration target: PostgreSQL with 10x performance improvement and RAG capabilities.

---

## User Story

**Primary Actor**: Dagster Pipeline + AI Agents

> As a Dagster pipeline and AI Agent, I want to collect daily OHLC data from yfinance, insider data from FinnHub, and fundamental data from FinnHub with PostgreSQL + TimescaleDB storage, so that agents have high-performance, RAG-enhanced market data access for comprehensive trading analysis.

---

## Acceptance Criteria

### AC1: API Compatibility
**GIVEN** the MarketData domain migration  
**WHEN** PostgreSQL + TimescaleDB integration is complete  
**THEN** all existing MarketDataService APIs remain 100% compatible with 10x performance improvement

### AC2: Data Collection Pipeline
**GIVEN** daily market data collection  
**WHEN** Dagster pipelines execute  
**THEN** OHLC data from yfinance and insider/fundamental data from FinnHub are stored in TimescaleDB hypertables

### AC3: Performance Requirements
**GIVEN** historical market data queries  
**WHEN** AI agents request technical analysis  
**THEN** responses are delivered within 100ms using TimescaleDB time-series optimization

### AC4: Technical Indicators Preservation
**GIVEN** technical analysis requests  
**WHEN** agents query indicators  
**THEN** all 20 existing TA-Lib indicators are preserved with PostgreSQL-backed data access

### AC5: RAG Integration
**GIVEN** RAG-powered analysis  
**WHEN** agents search for historical patterns  
**THEN** vector similarity search using pgvectorscale returns relevant market conditions within 200ms

### AC6: Concurrent Access
**GIVEN** concurrent agent operations  
**WHEN** multiple agents access market data  
**THEN** PostgreSQL async operations support concurrent reads without file system limitations

### AC7: Data Quality
**GIVEN** data quality requirements  
**WHEN** market data is collected  
**THEN** comprehensive validation, audit trails, and error handling maintain data integrity with PostgreSQL ACID transactions

---

## Business Rules

### BR1: API Preservation
- Preserve 100% API compatibility with existing MarketDataService for seamless migration
- Maintain all existing method signatures in FundamentalDataService and InsiderDataService

### BR2: Data Collection Standards
- Daily automated collection from yfinance (OHLC) and FinnHub (insider + fundamentals) via Dagster pipelines
- FinnHub API rate limiting compliance with proper backoff strategies
- Graceful degradation when external APIs are unavailable

### BR3: Database Architecture
- TimescaleDB hypertables for market_data, fundamental_data, and insider_data tables
- Vector embeddings generation for technical analysis patterns using pgvectorscale

### BR4: Performance Standards
- Sub-100ms query performance for common market data operations
- Data retention policy: 10 years for OHLC, 5 years for fundamentals, 3 years for insider data

### BR5: Audit and Compliance
- Comprehensive audit logging for all data collection and agent queries

---

## Scope

### Included
- PostgreSQL + TimescaleDB + pgvectorscale migration from CSV storage
- Preserve all existing YFinanceClient and FinnhubClient integrations
- Maintain complete MarketDataService, FundamentalDataService, InsiderDataService APIs
- Async PostgreSQL repository operations following news domain patterns
- Vector embeddings for RAG-powered historical pattern matching
- TimescaleDB hypertables for time-series optimization
- Batch data ingestion pipeline for daily Dagster collection
- Comprehensive testing with real PostgreSQL database

### Excluded
- Real-time data streaming (daily batch collection only)
- Additional data providers beyond yfinance and FinnHub
- New technical indicators beyond existing 20 TA-Lib indicators
- Multi-database support (PostgreSQL only)
- GraphQL or REST API endpoints (agent integration only)

---

## Technical Design

### Architecture Pattern

**Router → Service → Repository → Entity → Database**

The migration preserves the existing service interfaces while upgrading the underlying data persistence layer.

### Data Flow

```
External APIs (YFinance + FinnHub) → Dagster Pipeline → PostgreSQL Storage 
                                                  ↓
                                         Repository Layer → Service Layer → Agents
                                                  ↓
                                         pgvectorscale (RAG)
```

### Domain Model

#### MarketDataEntity (new)
OHLC price data with TimescaleDB optimization
- Fields: symbol, timestamp, open_price, high_price, low_price, close_price, volume, adjusted_close
- Vector fields: technical_pattern_embedding (384 dims), price_movement_embedding (384 dims)
- Validation: high >= low, high >= open/close, low <= open/close

#### FundamentalDataEntity (new)
Financial statement data with PostgreSQL storage
- Fields: symbol, report_date, period_type, balance sheet, income statement, cash flow, calculated ratios
- Vector field: financial_health_embedding (384 dims)

#### InsiderDataEntity (new)
SEC insider transaction records with sentiment analysis
- Fields: symbol, transaction_date, insider_name, insider_position, transaction_type, shares_traded, transaction_price, sentiment_score
- Vector field: transaction_pattern_embedding (384 dims)

#### TechnicalIndicatorEntity (new)
Calculated TA-Lib indicator values with vector embeddings
- Fields: symbol, timestamp, all 20 TA-Lib indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, patterns)
- Vector field: indicator_pattern_embedding (384 dims)

---

## Database Schema

### TimescaleDB Hypertables

```sql
-- Market Data (OHLC)
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    adjusted_close DECIMAL(10,2),
    technical_pattern_embedding vector(384),
    price_movement_embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('market_data', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Fundamental Data
CREATE TABLE fundamental_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    report_date TIMESTAMPTZ NOT NULL,
    period_type VARCHAR(10),
    -- Balance Sheet, Income Statement, Cash Flow columns
    financial_health_embedding vector(384),
    UNIQUE(symbol, report_date, period_type)
);
SELECT create_hypertable('fundamental_data', 'report_date', chunk_time_interval => INTERVAL '3 months');

-- Insider Data
CREATE TABLE insider_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    transaction_date TIMESTAMPTZ NOT NULL,
    insider_name VARCHAR(200),
    insider_position VARCHAR(100),
    transaction_type VARCHAR(20),
    shares_traded BIGINT,
    transaction_price DECIMAL(10,2),
    sentiment_score DECIMAL(3,2),
    transaction_pattern_embedding vector(384)
);
SELECT create_hypertable('insider_data', 'transaction_date', chunk_time_interval => INTERVAL '1 month');

-- Technical Indicators
CREATE TABLE technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    -- 20 TA-Lib indicators (sma, ema, rsi, macd, bollinger, atr, obv, patterns)
    indicator_pattern_embedding vector(384)
);
SELECT create_hypertable('technical_indicators', 'timestamp', chunk_time_interval => INTERVAL '1 day');
```

### Vector Indexes

```sql
-- DiskANN indexes for pgvectorscale
CREATE INDEX idx_market_technical_emb ON market_data USING diskann (technical_pattern_embedding);
CREATE INDEX idx_market_price_emb ON market_data USING diskann (price_movement_embedding);
CREATE INDEX idx_fundamental_emb ON fundamental_data USING diskann (financial_health_embedding);
CREATE INDEX idx_insider_emb ON insider_data USING diskann (transaction_pattern_embedding);
CREATE INDEX idx_technical_emb ON technical_indicators USING diskann (indicator_pattern_embedding);
```

---

## Implementation Phases

### Phase 1: Database Infrastructure (2-3 hours)
- Set up PostgreSQL with TimescaleDB and pgvectorscale extensions
- Create database schemas with proper indexing
- Run migrations to create all tables

### Phase 2: Entity Models (4-6 hours)
- MarketDataEntity with TimescaleDB optimization
- FundamentalDataEntity for financial statement data
- InsiderDataEntity for SEC transaction records
- TechnicalIndicatorEntity for calculated indicator values

### Phase 3: Repository Migration (6-8 hours)
- Async PostgreSQL repository operations (match news domain patterns)
- Vector similarity search capabilities
- Batch operations for high-performance data loading

### Phase 4: Data Migration (4-6 hours)
- CSV to PostgreSQL migration scripts
- Data validation and integrity checks
- Generate vector embeddings for all data

### Phase 5: Service Preservation (4-6 hours)
- Update MarketDataService with PostgreSQL backend
- Update FundamentalDataService with PostgreSQL backend
- Update InsiderDataService with PostgreSQL backend
- Add RAG-enhanced analysis features

### Phase 6: Testing & Integration (4-6 hours)
- Data integrity tests
- API compatibility validation
- Performance benchmarks
- Concurrent access testing

---

## Configuration

### Environment Variables

```bash
DATABASE_URL="postgresql://..."
FINNHUB_API_KEY="..."
OPENROUTER_API_KEY="..."
```

### Dependencies

```toml
# Already implemented
yfinance = "*"
finnhub-python = "*"
talib = "*"
psycopg2-binary = "*"
asyncpg = "*"
sqlalchemy = {extras = ["asyncio"]}
```

---

## Success Metrics

### Performance Targets
- **10x query performance improvement** over CSV-based storage
- **Sub-100ms market data operations** for common agent queries
- **Sub-200ms RAG queries** for vector similarity search
- **Support for 500+ tickers** with concurrent agent access

### Compatibility Standards
- **100% existing API preservation** without breaking changes
- **Seamless migration** without agent disruption
- **Efficient bulk data ingestion** for Dagster pipelines

### Quality Assurance
- **85%+ test coverage maintained** across all components
- **Comprehensive data validation** and audit trails
- **PostgreSQL ACID transactions** for data integrity
