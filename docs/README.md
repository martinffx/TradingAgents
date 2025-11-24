# TradingAgents Documentation

This directory contains the complete documentation for the TradingAgents project, including business context, implementation specifications, and technical standards.

## Structure

- **[Product](product/)** - Business context, user requirements, and development roadmap
- **[Specifications](specs/)** - Detailed implementation specifications for each domain
- **[Standards](standards/)** - Technical architecture, coding standards, and development practices

## Quick Reference

### Implementation Order
```
Entity → Repository → Service → Client → Agent
```

### Data Flow
```
Request → Agent → Service → Repository → Entity → Database
```

### Core Technologies
- **Python 3.13** with asyncio
- **PostgreSQL + TimescaleDB + pgvectorscale**
- **OpenRouter** for LLM integration
- **pytest-vcr** for HTTP testing
- **ruff** for code quality

## Domain Status

| Domain | Progress | Status |
|-------|----------|--------|
| **News** | 95% Complete | ✅ Production-ready |
| **MarketData** | 15% Complete | 🟡 PostgreSQL migration in progress |
| **SocialMedia** | 0% Complete | 🟡 Planning complete |

## Technical Standards

- **[standards/coding.md](standards/coding.md)** - Stub-driven TDD approach, testing strategies, code style
- **[standards/architecture.md](standards/architecture.md)** - Layered architecture patterns, database design

---

**Last Updated**: 2025-11-24