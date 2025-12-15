from pydantic import BaseModel
from typing import Optional

class TransactionData(BaseModel):
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: str
    CountryCode: str
    ProviderId: int
    ProductId: int
    ProductCategory: str
    ChannelId: int
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: Optional[str] = None
    FraudResult: Optional[int] = None
