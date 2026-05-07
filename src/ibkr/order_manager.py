import logging
from dataclasses import dataclass, field
from typing import Optional

from ib_insync import IB, LimitOrder, Stock

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    direction: int  # 0=HOLD, 1=BUY, 2=SELL
    size: float     # fraction 0..1
    confidence: float = 0.0


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.avg_cost)


@dataclass
class AccountSummary:
    cash: float
    equity: float
    buying_power: float


class OrderManager:
    def __init__(self, ib: IB):
        self.ib = ib
        self.pending_orders: dict[int, object] = {}
        self.positions: dict[str, Position] = {}

    def place_trade(self, signal: TradeSignal, account: AccountSummary) -> Optional[int]:
        if signal.direction == 0:
            return None

        contract = Stock(signal.symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        mid_price = self._get_mid_price(contract)
        if mid_price <= 0:
            logger.warning(f"Could not get valid price for {signal.symbol}, skipping trade")
            return None

        if signal.direction == 1:
            buy_value = account.cash * signal.size
            # Use a low minimum for small accounts (configurable via risk limits)
            min_trade = getattr(self, '_min_trade_value', 50)
            if buy_value < min_trade:
                logger.info(f"Buy value ${buy_value:.0f} below minimum (${min_trade}), skipping")
                return None
            qty = int(buy_value / mid_price)
            if qty <= 0:
                return None
            limit_price = self._get_limit_price(contract, is_buy=True)
            if limit_price <= 0:
                return None
            order = LimitOrder("BUY", qty, limit_price)
        else:
            pos = self.positions.get(signal.symbol)
            if not pos or pos.quantity <= 0:
                return None
            qty = int(pos.quantity * signal.size)
            if qty <= 0:
                return None
            limit_price = self._get_limit_price(contract, is_buy=False)
            if limit_price <= 0:
                return None
            order = LimitOrder("SELL", qty, limit_price)

        trade = self.ib.placeOrder(contract, order)
        self.pending_orders[trade.order.orderId] = trade
        logger.info(f"Placed {signal.direction} {qty} {signal.symbol}")
        return trade.order.orderId

    def cancel_order(self, order_id: int):
        if order_id in self.pending_orders:
            trade = self.pending_orders.pop(order_id)
            self.ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order {order_id}")

    def sync_positions(self):
        for pos in self.ib.positions():
            # Get current market price for the position
            mkt_price = 0.0
            try:
                ticker = self.ib.reqMktData(pos.contract, "", False, False)
                self.ib.sleep(0.3)
                mkt_price = float(ticker.last or ticker.close or 0)
                if mkt_price <= 0 or mkt_price != mkt_price:
                    # Fallback to historical bar
                    bars = self.ib.reqHistoricalData(
                        pos.contract, endDateTime="", durationStr="1 D",
                        barSizeSetting="1 min", whatToShow="TRADES",
                        useRTH=False, formatDate=1,
                    )
                    if bars:
                        mkt_price = float(bars[-1].close)
                self.ib.cancelMktData(pos.contract)
            except Exception:
                pass

            self.positions[pos.contract.symbol] = Position(
                symbol=pos.contract.symbol,
                quantity=float(pos.position),
                avg_cost=float(pos.avgCost),
                market_price=mkt_price,
            )

    def get_account_summary(self) -> AccountSummary:
        values = {v.tag: v.value for v in self.ib.accountSummary()}
        return AccountSummary(
            cash=float(values.get("AvailableFunds", 0)),
            equity=float(values.get("NetLiquidation", 0)),
            buying_power=float(values.get("BuyingPower", 0)),
        )

    def _get_mid_price(self, contract) -> float:
        """Get current price using historical bars (free, no subscription needed)."""
        # Skip reqMktData — requires subscription
        # Use historical bars directly
        for duration, bar_size in [("1 D", "1 min"), ("5 D", "1 day")]:
            try:
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr=duration,
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=False, formatDate=1,
                )
                if bars and len(bars) > 0:
                    return float(bars[-1].close)
            except Exception:
                continue

        return 0.0

    def _get_limit_price(self, contract, is_buy: bool) -> float:
        mid = self._get_mid_price(contract)
        if mid <= 0:
            return 0.0
        if is_buy:
            return round(mid * 0.9995, 2)
        else:
            return round(mid * 1.0005, 2)
