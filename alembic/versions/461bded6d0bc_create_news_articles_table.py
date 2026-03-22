"""create_news_articles_table

Revision ID: 461bded6d0bc
Revises: 
Create Date: 2026-03-22 11:47:30.206474

Purpose:
    Create the news_articles table with all required columns including
    vector embeddings for semantic similarity search.
"""

from alembic import op
import sqlalchemy as sa


revision: str = "461bded6d0bc"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE news_articles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            headline TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            source VARCHAR(100) NOT NULL,
            published_date DATE NOT NULL,
            summary TEXT,
            entities JSONB,
            sentiment_score FLOAT,
            sentiment_confidence FLOAT,
            sentiment_label VARCHAR(50),
            author VARCHAR(255),
            category VARCHAR(100),
            symbol VARCHAR(20),
            title_embedding vector(1536),
            content_embedding vector(1536),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("CREATE INDEX idx_symbol_date ON news_articles (symbol, published_date)")
    op.execute("CREATE INDEX idx_published_date ON news_articles (published_date)")
    op.execute("CREATE INDEX idx_news_sentiment_label ON news_articles (sentiment_label)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_news_sentiment_label")
    op.execute("DROP INDEX IF EXISTS idx_published_date")
    op.execute("DROP INDEX IF EXISTS idx_symbol_date")
    op.execute("DROP TABLE IF EXISTS news_articles")
