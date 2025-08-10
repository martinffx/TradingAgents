"""Type stubs for newspaper (newspaper4k package)."""

from datetime import datetime

class Config:
    """Configuration for newspaper Article."""

    browser_user_agent: str
    request_timeout: int
    fetch_images: bool

    def __init__(self) -> None: ...

class Article:
    """Article class for parsing web articles."""

    text: str
    title: str | None
    authors: list[str]
    publish_date: datetime | None
    top_image: str | None
    movies: list[str]
    keywords: list[str]
    summary: str

    def __init__(self, url: str, config: Config | None = None) -> None: ...
    def download(self) -> None: ...
    def parse(self) -> None: ...
    def nlp(self) -> None: ...

def article(url: str) -> Article: ...
