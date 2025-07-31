"""
Reddit API integration for social sentiment analysis.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import praw

# from .api_clients import RateLimiter

logger = logging.getLogger(__name__)

# Extended ticker to company mapping
ticker_to_company = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "TSM": "Taiwan Semiconductor Manufacturing Company OR TSMC",
    "JPM": "JPMorgan Chase OR JP Morgan",
    "JNJ": "Johnson & Johnson OR JNJ",
    "V": "Visa",
    "WMT": "Walmart",
    "META": "Meta OR Facebook",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "BABA": "Alibaba",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "PYPL": "PayPal",
    "PLTR": "Palantir",
    "MU": "Micron",
    "SQ": "Block OR Square",
    "ZM": "Zoom",
    "CSCO": "Cisco",
    "SHOP": "Shopify",
    "ORCL": "Oracle",
    "X": "Twitter OR X",
    "SPOT": "Spotify",
    "AVGO": "Broadcom",
    "ASML": "ASML",
    "TWLO": "Twilio",
    "SNAP": "Snap Inc.",
    "TEAM": "Atlassian",
    "SQSP": "Squarespace",
    "UBER": "Uber",
    "ROKU": "Roku",
    "PINS": "Pinterest",
    # Additional popular tickers
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "GME": "GameStop",
    "AMC": "AMC Entertainment",
    "BB": "BlackBerry",
    "NOK": "Nokia",
    "COIN": "Coinbase",
    "HOOD": "Robinhood",
    "RBLX": "Roblox",
    "DKNG": "DraftKings",
    "PENN": "Penn Entertainment",
    "SOFI": "SoFi Technologies",
}


class RedditClient:
    """Client for Reddit API with rate limiting and caching."""

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit client.

        Args:
            client_id: Reddit application client ID
            client_secret: Reddit application client secret
            user_agent: User agent string for API requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent

        # Reddit allows 100 requests per minute for script applications
        # self.rate_limiter = RateLimiter(100, 60)

        # Initialize Reddit instance
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )

        # Default subreddits for different categories
        self.subreddits = {
            "investing": ["investing", "SecurityAnalysis", "ValueInvesting", "stocks"],
            "trading": ["wallstreetbets", "StockMarket", "pennystocks", "options"],
            "global_news": ["news", "worldnews", "Economics", "business"],
            "company_news": ["investing", "stocks", "StockMarket", "SecurityAnalysis"],
        }

    def test_connection(self) -> bool:
        """Test if the Reddit API connection is working."""
        try:
            # Test by fetching user info (read-only operation)
            _ = self.reddit.user.me()
            return True
        except Exception as e:
            logger.error(f"Reddit connection test failed: {e}")
            return False

    def search_posts(
        self,
        query: str,
        subreddit_names: list[str],
        limit: int = 25,
        time_filter: str = "week",
    ) -> list[dict[str, Any]]:
        """
        Search for posts across multiple subreddits.

        Args:
            query: Search query
            subreddit_names: List of subreddit names to search
            limit: Maximum number of posts per subreddit
            time_filter: Time filter ('day', 'week', 'month', 'year', 'all')

        Returns:
            List of post dictionaries
        """
        posts = []

        for subreddit_name in subreddit_names:
            try:
                # self.rate_limiter.wait_if_needed()

                subreddit = self.reddit.subreddit(subreddit_name)

                # Search posts in the subreddit
                search_results = subreddit.search(
                    query=query, sort="relevance", time_filter=time_filter, limit=limit
                )

                for submission in search_results:
                    post_data = {
                        "title": submission.title,
                        "content": submission.selftext,
                        "url": submission.url,
                        "upvotes": submission.ups,
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "created_utc": submission.created_utc,
                        "subreddit": subreddit_name,
                        "author": str(submission.author)
                        if submission.author
                        else "[deleted]",
                    }
                    posts.append(post_data)

            except Exception as e:
                logger.error(f"Error searching subreddit {subreddit_name}: {e}")
                continue

        # Sort by score (upvotes - downvotes) descending
        posts.sort(key=lambda x: x["score"], reverse=True)
        return posts

    def get_top_posts(
        self, subreddit_names: list[str], limit: int = 25, time_filter: str = "week"
    ) -> list[dict[str, Any]]:
        """
        Get top posts from multiple subreddits.

        Args:
            subreddit_names: List of subreddit names
            limit: Maximum number of posts per subreddit
            time_filter: Time filter ('day', 'week', 'month', 'year', 'all')

        Returns:
            List of post dictionaries
        """
        posts = []

        for subreddit_name in subreddit_names:
            try:
                # self.rate_limiter.wait_if_needed()

                subreddit = self.reddit.subreddit(subreddit_name)

                # Get top posts from the subreddit
                top_posts = subreddit.top(time_filter=time_filter, limit=limit)

                for submission in top_posts:
                    post_data = {
                        "title": submission.title,
                        "content": submission.selftext,
                        "url": submission.url,
                        "upvotes": submission.ups,
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "created_utc": submission.created_utc,
                        "subreddit": subreddit_name,
                        "author": str(submission.author)
                        if submission.author
                        else "[deleted]",
                    }
                    posts.append(post_data)

            except Exception as e:
                logger.error(f"Error fetching top posts from {subreddit_name}: {e}")
                continue

        # Sort by score descending
        posts.sort(key=lambda x: x["score"], reverse=True)
        return posts

    def filter_posts_by_date(
        self, posts: list[dict[str, Any]], start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """
        Filter posts by date range.

        Args:
            posts: List of post dictionaries
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Filtered list of posts
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(
            days=1
        )  # Include end date

        filtered_posts = []
        for post in posts:
            post_dt = datetime.fromtimestamp(post["created_utc"])
            if start_dt <= post_dt <= end_dt:
                # Add formatted date string
                post["posted_date"] = post_dt.strftime("%Y-%m-%d")
                filtered_posts.append(post)

        return filtered_posts

    def filter_posts_by_company(
        self, posts: list[dict[str, Any]], ticker: str
    ) -> list[dict[str, Any]]:
        """
        Filter posts that mention a specific company/ticker.

        Args:
            posts: List of post dictionaries
            ticker: Stock ticker symbol

        Returns:
            Filtered list of posts that mention the company
        """
        if ticker not in ticker_to_company:
            # If ticker not in mapping, search for ticker directly
            search_terms = [ticker.upper()]
        else:
            # Get company names and ticker
            company_names = ticker_to_company[ticker]
            if "OR" in company_names:
                search_terms = [name.strip() for name in company_names.split(" OR")]
            else:
                search_terms = [company_names]
            search_terms.append(ticker.upper())

        filtered_posts = []
        for post in posts:
            title_text = post["title"].lower()
            content_text = post["content"].lower()

            # Check if any search term appears in title or content
            found = False
            for term in search_terms:
                term_lower = term.lower()
                if re.search(
                    r"\b" + re.escape(term_lower) + r"\b", title_text
                ) or re.search(r"\b" + re.escape(term_lower) + r"\b", content_text):
                    found = True
                    break

            if found:
                filtered_posts.append(post)

        return filtered_posts


def fetch_top_from_category(
    category: str,
    date: str,
    max_limit: int,
    query: str | None = None,
    data_path: str = "reddit_data",
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str | None = None,
) -> list[dict[str, Any]]:
    """
    Legacy function to maintain backward compatibility.
    Now uses live Reddit API if credentials are provided.

    Args:
        category: Category ('global_news', 'company_news', etc.)
        date: Date in YYYY-MM-DD format
        max_limit: Maximum number of posts
        query: Optional search query (ticker for company_news)
        data_path: Unused in API mode
        client_id: Reddit client ID
        client_secret: Reddit client secret
        user_agent: Reddit user agent

    Returns:
        List of post dictionaries
    """
    if not all([client_id, client_secret, user_agent]):
        logger.warning("Reddit API credentials not provided. Returning empty data.")
        return []

    try:
        # Type check ensures these are not None
        assert client_id is not None
        assert client_secret is not None
        assert user_agent is not None
        client = RedditClient(client_id, client_secret, user_agent)

        # Determine subreddits based on category
        if category == "global_news":
            subreddit_names = client.subreddits["global_news"]
        elif category == "company_news":
            subreddit_names = client.subreddits["company_news"]
        else:
            # Default to investing subreddits
            subreddit_names = client.subreddits["investing"]

        # Calculate time filter based on date (Reddit doesn't support exact date filtering)
        post_date = datetime.strptime(date, "%Y-%m-%d")
        days_ago = (datetime.now() - post_date).days

        if days_ago <= 1:
            time_filter = "day"
        elif days_ago <= 7:
            time_filter = "week"
        elif days_ago <= 30:
            time_filter = "month"
        else:
            time_filter = "year"

        # Get posts
        if query and category == "company_news":
            # Search for specific company
            posts = client.search_posts(
                query=query,
                subreddit_names=subreddit_names,
                limit=max_limit // len(subreddit_names),
                time_filter=time_filter,
            )
            # Filter by company mentions
            posts = client.filter_posts_by_company(posts, query)
        else:
            # Get top posts
            posts = client.get_top_posts(
                subreddit_names=subreddit_names,
                limit=max_limit // len(subreddit_names),
                time_filter=time_filter,
            )

        # Filter by date (approximate)
        start_date = date
        end_date = date
        posts = client.filter_posts_by_date(posts, start_date, end_date)

        # Limit results
        return posts[:max_limit]

    except Exception as e:
        logger.error(f"Error fetching Reddit data: {e}")
        return []
