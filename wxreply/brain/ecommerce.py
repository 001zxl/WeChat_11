"""电商客服上下文：订单、物流、商品、售后政策查询。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

DEFAULT_ORDERS = {
    "TB20261117001": {
        "order_id": "TB20261117001", "buyer_id": "buyer_001", "shop": "淘宝店",
        "item": "户外折叠桌 FT-120", "qty": 500, "amount": 2890.0,
        "status": "已发货", "logistics": "顺丰 SF1234567890", "paid": True,
    },
    "JD20261116008": {
        "order_id": "JD20261116008", "buyer_id": "buyer_002", "shop": "京东店",
        "item": "折叠椅 YC-08", "qty": 20, "amount": 760.0,
        "status": "待发货", "logistics": "", "paid": True,
    },
}

AFTER_SALE_POLICY = (
    "售后政策：非定制商品 7 天无理由（不影响二次销售）；"
    "质量问题包退换，运费商家承担；定制商品下单后不支持无理由退换；"
    "发票默认电子普票，可开专票；价保 7 天。"
)


@dataclass
class OrderInfo:
    order_id: str
    buyer_id: str = ""
    shop: str = ""
    item: str = ""
    qty: int = 0
    amount: float = 0.0
    status: str = ""
    logistics: str = ""
    paid: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EcommerceService:
    def __init__(self, orders_path: str = "") -> None:
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.orders_path = orders_path
        self._load()

    def _load(self) -> None:
        data = None
        if self.orders_path and os.path.exists(self.orders_path):
            try:
                with open(self.orders_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                data = None
        if isinstance(data, list):
            self.orders = {str(item.get("order_id")): item for item in data if item.get("order_id")}
        elif isinstance(data, dict):
            self.orders = {str(key): value for key, value in data.items() if isinstance(value, dict)}
        else:
            self.orders = dict(DEFAULT_ORDERS)

    def find(self, text: str = "", context: Optional[Dict[str, Any]] = None,
             peer: str = "") -> Optional[OrderInfo]:
        context = context or {}
        candidates: List[str] = []
        for key in ("order_id", "orderId", "trade_id"):
            if context.get(key):
                candidates.append(str(context[key]))
        for order_id in self.orders:
            if order_id and order_id in (text or ""):
                candidates.append(order_id)
        for order_id in candidates:
            item = self.orders.get(order_id)
            if item:
                return OrderInfo(
                    order_id=order_id,
                    buyer_id=str(item.get("buyer_id") or context.get("buyer_id") or ""),
                    shop=str(item.get("shop") or context.get("shop") or ""),
                    item=str(item.get("item") or ""), qty=int(item.get("qty") or 0),
                    amount=float(item.get("amount") or 0), status=str(item.get("status") or ""),
                    logistics=str(item.get("logistics") or ""), paid=bool(item.get("paid")),
                )
        buyer = context.get("buyer_id") or peer.split(":", 1)[-1]
        for order_id, item in self.orders.items():
            if str(item.get("buyer_id")) == str(buyer):
                return OrderInfo(
                    order_id=order_id, buyer_id=str(buyer), shop=str(item.get("shop") or ""),
                    item=str(item.get("item") or ""), qty=int(item.get("qty") or 0),
                    amount=float(item.get("amount") or 0), status=str(item.get("status") or ""),
                    logistics=str(item.get("logistics") or ""), paid=bool(item.get("paid")),
                )
        return None

    def describe(self, peer: str = "", text: str = "", context: Optional[Dict[str, Any]] = None) -> str:
        order = self.find(text, context, peer)
        if order is None:
            return "【电商客服】未匹配到订单；如客户提供订单号/手机号后四位，可先核验订单。\n" + AFTER_SALE_POLICY
        lines = [
            "【电商客服上下文】",
            f"订单：{order.order_id}（{order.shop}）",
            f"商品：{order.item} × {order.qty}",
            f"金额：¥{order.amount:.2f} / 支付状态：{'已支付' if order.paid else '未支付'}",
            f"订单状态：{order.status or '未知'}",
            f"物流：{order.logistics or '暂无物流'}",
            AFTER_SALE_POLICY,
        ]
        return "\n".join(lines)
