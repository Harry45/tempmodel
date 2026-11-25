"""Temperature futures and options: section 4 of Benth & Šaltytė Benth (2007).

The paper works in continuous time. Contracts settle on daily values, and the model
is fitted to daily values, so everything here uses the daily model. Prices therefore
agree exactly with ``predict`` and ``sample``.
"""

from tempmodel.pricing.indices import realised_indices
from tempmodel.pricing.period import DeliveryPeriod, next_months
from tempmodel.pricing.pricer import Pricer

__all__ = ["DeliveryPeriod", "Pricer", "next_months", "realised_indices"]
