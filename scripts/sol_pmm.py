import logging
import os
from decimal import Decimal
from typing import Dict, List

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import MarketDict, OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


class SolPMMConfig(StrategyV2ConfigBase):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    controllers_config: List[str] = []
    exchange: str = Field("binance_paper_trade", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Exchange (e.g. binance_paper_trade):"))
    trading_pair: str = Field("SOL-USDT", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Trading pair (e.g. SOL-USDT):"))
    order_amount: Decimal = Field(Decimal("50"), client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order amount (base asset):"))
    bid_spread: Decimal = Field(Decimal("0.0005"), client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Bid spread (e.g. 0.001 = 0.1%):"))
    ask_spread: Decimal = Field(Decimal("0.0005"), client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Ask spread (e.g. 0.001 = 0.1%):"))
    order_refresh_time: int = Field(60, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order refresh time (seconds):"))
    order_levels: int = Field(3, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Number of order levels:"))
    level_spread: Decimal = Field(Decimal("0.0004"), client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Spread between levels:"))
    target_base_pct: Decimal = Field(Decimal("0.5"), client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Target base asset % (0.5 = 50%):"))
    skew_factor: Decimal = Field(Decimal("2.0"), client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Inventory skew factor:"))

    def update_markets(self, markets: MarketDict) -> MarketDict:
        markets[self.exchange] = markets.get(self.exchange, set()) | {self.trading_pair}
        return markets


class SolPMM(StrategyV2Base):
    create_timestamp = 0

    def __init__(self, connectors: Dict[str, ConnectorBase], config: SolPMMConfig):
        super().__init__(connectors, config)
        self.config = config

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal = self.create_proposal()
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.config.order_refresh_time + self.current_timestamp

    def get_inventory_skew(self):
        connector = self.connectors[self.config.exchange]
        base, quote = self.config.trading_pair.split("-")
        base_bal = float(connector.get_balance(base))
        quote_bal = float(connector.get_balance(quote))
        mid = float(connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice))
        base_val = base_bal * mid
        total_val = base_val + quote_bal
        if total_val == 0:
            return Decimal("1.0"), Decimal("1.0")
        current_pct = Decimal(str(base_val / total_val))
        skew = (current_pct - self.config.target_base_pct) * self.config.skew_factor
        bid_mult = max(Decimal("0.1"), Decimal("1.0") - skew)
        ask_mult = max(Decimal("0.1"), Decimal("1.0") + skew)
        return bid_mult, ask_mult

    def create_proposal(self) -> List[OrderCandidate]:
        connector = self.connectors[self.config.exchange]
        ref_price = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)
        bid_mult, ask_mult = self.get_inventory_skew()
        orders = []
        for i in range(1, self.config.order_levels + 1):
            bid_spread = self.config.bid_spread + self.config.level_spread * (i - 1)
            ask_spread = self.config.ask_spread + self.config.level_spread * (i - 1)
            amt_mult = Decimal("1.0") + Decimal("0.3") * (i - 1)
            orders.append(OrderCandidate(
                trading_pair=self.config.trading_pair, is_maker=True,
                order_type=OrderType.LIMIT, order_side=TradeType.BUY,
                amount=self.config.order_amount * amt_mult * bid_mult,
                price=ref_price * Decimal(1 - float(bid_spread))
            ))
            orders.append(OrderCandidate(
                trading_pair=self.config.trading_pair, is_maker=True,
                order_type=OrderType.LIMIT, order_side=TradeType.SELL,
                amount=self.config.order_amount * amt_mult * ask_mult,
                price=ref_price * Decimal(1 + float(ask_spread))
            ))
        return orders

    def adjust_proposal_to_budget(self, proposal):
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(proposal, all_or_none=False)

    def place_orders(self, proposal):
        for order in proposal:
            if order.order_side == TradeType.SELL:
                self.sell(connector_name=self.config.exchange, trading_pair=order.trading_pair,
                          amount=order.amount, order_type=order.order_type, price=order.price)
            elif order.order_side == TradeType.BUY:
                self.buy(connector_name=self.config.exchange, trading_pair=order.trading_pair,
                         amount=order.amount, order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} @ {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        connector = self.connectors[self.config.exchange]
        base, quote = self.config.trading_pair.split("-")
        mid = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)
        bid_m, ask_m = self.get_inventory_skew()
        return "\n".join([
            f"\n  SOL PMM V2 | {self.config.trading_pair}",
            f"  Mid: {float(mid):.4f} | Bid mult: {float(bid_m):.2f} | Ask mult: {float(ask_m):.2f}",
            f"  Balance: {float(connector.get_balance(base)):.2f} {base} | {float(connector.get_balance(quote)):.2f} {quote}",
            f"  Active orders: {len(self.get_active_orders(self.config.exchange))}",
        ])
