from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ProductData:
    url: str = ""
    query_name: str = ""
    name_fa: str = ""
    name_en: str = ""
    product_id: str = ""

    rate: Optional[float] = None
    review_count: Optional[int] = None
    wishlist_count: Optional[int] = None

    is_out_of_stuck: bool = False
    has_discount: bool = False
    discount_percentage: Optional[str] = None
    real_price: Optional[str] = None
    discounted_price: Optional[str] = None

    image_urls: dict[str, list[str]] = field(default_factory=dict)
    image_ids: List[str] = field(default_factory=list)

    product_intro: list[str] = field(default_factory=list)
    usage_instructions: list[str] = field(default_factory=list)

    error: Optional[str] = None

    @property
    def is_failed(self) -> bool:
        """Determines if the product failed scraping criteria."""
        if self.error and self.error.strip():
            return True
        if not self.name_fa or not self.name_fa.strip():
            return True
        return False
