"""
Repository for social media data (Reddit posts and social media content).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """Represents a social media post."""

    title: str
    content: str
    author: str
    source: str  # "reddit", "twitter", etc.
    platform_id: (
        str  # Unique identifier for deduplication (Reddit post ID, tweet ID, etc.)
    )
    created_date: date

    # Optional fields
    subreddit: str | None = None  # Reddit-specific
    score: int = 0
    comments_count: int = 0
    upvote_ratio: float | None = None  # Reddit-specific
    url: str | None = None
    sentiment_score: float | None = None
    engagement_score: int = 0
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)


@dataclass
class SocialData:
    """Container for social media data with metadata."""

    query: str
    date: date
    source: str  # "reddit", "twitter", etc.
    posts: list[SocialPost]


class SocialRepository:
    """Repository for accessing cached social media data with source separation."""

    def __init__(self, data_dir: str, **kwargs):
        """
        Initialize social media repository.

        Args:
            data_dir: Base directory for social media data storage
            **kwargs: Additional configuration
        """
        self.social_data_dir = Path(data_dir) / "social_data"
        self.social_data_dir.mkdir(parents=True, exist_ok=True)

    def get_social_data(
        self,
        query: str,
        start_date: date,
        end_date: date,
        sources: list[str] | None = None,
    ) -> dict[date, list[SocialData]]:
        """
        Get cached social media data for a query and date range across sources.

        Args:
            query: Search query or symbol
            start_date: Start date
            end_date: End date
            sources: List of sources to check (default: ["reddit"])

        Returns:
            Dict[date, list[SocialData]]: Social data keyed by date, with list of source data
        """
        if sources is None:
            sources = ["reddit"]

        social_data = {}

        for source in sources:
            source_dir = self.social_data_dir / source / query

            if not source_dir.exists():
                logger.debug(f"No data directory found for {source}/{query}")
                continue

            # Scan for JSON files in the source/query directory
            for json_file in source_dir.glob("*.json"):
                try:
                    # Parse date from filename (YYYY-MM-DD.json)
                    date_str = json_file.stem
                    file_date = date.fromisoformat(date_str)

                    # Filter by date range
                    if start_date <= file_date <= end_date:
                        with open(json_file) as f:
                            data = json.load(f)

                        # Create SocialPost objects from JSON data
                        posts = []
                        for post_data in data.get("posts", []):
                            # Convert date strings back to date objects
                            post_data_copy = post_data.copy()
                            if "created_date" in post_data_copy:
                                post_data_copy["created_date"] = date.fromisoformat(
                                    post_data_copy["created_date"]
                                )

                            post = SocialPost(**post_data_copy)
                            posts.append(post)

                        # Create SocialData container
                        social_data_item = SocialData(
                            query=query, date=file_date, source=source, posts=posts
                        )

                        # Group by date (multiple sources per date)
                        if file_date not in social_data:
                            social_data[file_date] = []
                        social_data[file_date].append(social_data_item)

                except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(f"Error reading social data from {json_file}: {e}")
                    continue

        logger.info(
            f"Retrieved social data for {len(social_data)} dates for query '{query}'"
        )
        return social_data

    def store_social_posts(
        self,
        query: str,
        date: date,
        source: str,
        posts: list[SocialPost],
    ) -> tuple[date, SocialData]:
        """
        Store social media posts for a query, date, and source, merging with existing data.

        Args:
            query: Search query or symbol
            date: Date of the social media posts
            source: Social media source ("reddit", "twitter", etc.)
            posts: List of social media posts

        Returns:
            Tuple[date, SocialData]: The stored date and social data
        """
        # Create source/query directory
        source_dir = self.social_data_dir / source / query

        # Create JSON file path
        file_path = source_dir / f"{date.isoformat()}.json"

        try:
            # Merge with existing posts if file exists
            merged_posts = self._merge_posts_with_existing(file_path, posts)

            # Prepare data for JSON serialization
            posts_data = []
            for post in merged_posts:
                post_dict = asdict(post)
                # Convert date objects to ISO format strings for JSON
                if post_dict.get("created_date"):
                    post_dict["created_date"] = post_dict["created_date"].isoformat()
                posts_data.append(post_dict)

            data = {
                "query": query,
                "date": date.isoformat(),
                "source": source,
                "posts": posts_data,
                "metadata": {
                    "post_count": len(merged_posts),
                    "stored_at": date.today().isoformat(),
                    "repository": "social_repository",
                },
            }

            # Write to JSON file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Create SocialData result
            social_data = SocialData(
                query=query, date=date, source=source, posts=merged_posts
            )

            logger.info(
                f"Stored {len(posts)} new posts for {query} on {date} from {source} (total: {len(merged_posts)})"
            )
            return (date, social_data)

        except Exception as e:
            logger.error(
                f"Error storing social posts for {query} on {date} from {source}: {e}"
            )
            raise

    def store_social_data_batch(
        self,
        query: str,
        social_data_by_source: dict[str, dict[date, list[SocialPost]]],
    ) -> dict[date, list[SocialData]]:
        """
        Store multiple social media data sets for a query across sources.

        Args:
            query: Search query or symbol
            social_data_by_source: Nested dict of {source: {date: [posts]}}

        Returns:
            Dict[date, list[SocialData]]: The stored social data organized by date
        """
        stored_data = {}

        for source, date_posts in social_data_by_source.items():
            for post_date, posts in date_posts.items():
                try:
                    stored_date, stored_social_data = self.store_social_posts(
                        query, post_date, source, posts
                    )

                    # Group by date
                    if stored_date not in stored_data:
                        stored_data[stored_date] = []
                    stored_data[stored_date].append(stored_social_data)

                except Exception as e:
                    logger.error(
                        f"Failed to store social data for {query} on {post_date} from {source}: {e}"
                    )
                    continue

        total_dates = len(stored_data)
        total_sources = sum(len(social_list) for social_list in stored_data.values())
        logger.info(
            f"Stored social data for {total_dates} dates, {total_sources} source entries for query '{query}'"
        )
        return stored_data

    def _merge_posts_with_existing(
        self, file_path: Path, new_posts: list[SocialPost]
    ) -> list[SocialPost]:
        """
        Merge new posts with existing posts, deduplicating by platform_id.

        Args:
            file_path: Path to existing JSON file
            new_posts: New posts to merge

        Returns:
            List[SocialPost]: Merged and deduplicated posts
        """
        existing_posts = []

        # Load existing posts if file exists
        if file_path.exists():
            try:
                with open(file_path) as f:
                    data = json.load(f)

                for existing_data in data.get("posts", []):
                    # Convert date strings back to date objects
                    existing_data_copy = existing_data.copy()
                    if "created_date" in existing_data_copy:
                        existing_data_copy["created_date"] = date.fromisoformat(
                            existing_data_copy["created_date"]
                        )

                    existing_post = SocialPost(**existing_data_copy)
                    existing_posts.append(existing_post)

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error reading existing file {file_path}: {e}")
                existing_posts = []

        # Merge posts, deduplicating by platform_id (keep newer data)
        posts_by_id = {}

        # Add existing posts
        for post in existing_posts:
            posts_by_id[post.platform_id] = post

        # Add/update with new posts (they take precedence)
        for post in new_posts:
            posts_by_id[post.platform_id] = post

        # Return as sorted list
        merged_posts = list(posts_by_id.values())
        merged_posts.sort(key=lambda x: x.created_date, reverse=True)  # Newest first

        return merged_posts


# Alias for backwards compatibility
SocialMediaRepository = SocialRepository
