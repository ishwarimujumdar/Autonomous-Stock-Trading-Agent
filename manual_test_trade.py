"""
One-off manual smoke test: places a real (paper) BUY for 1 share, confirms the
fill, then sells it back - immediately, deliberately, independent of any
strategy's entry criteria.

This exists to answer a different question than a real trading session does.
A session asks "did a real setup appear today?" - and on a quiet day the
honest answer can be "no, all session," which is correct behavior, not a
failure. This script instead asks "does the execution/fill/reconciliation
pipeline actually work end to end?" - by using the exact same functions
(execute_order, wait_for_fill) the real graph uses, not a reimplementation.

Run it while the market is open:

    python manual_test_trade.py [SYMBOL]

SYMBOL defaults to a cheap, liquid name so 1 share ties up very little
capital. Nothing here is a strategy decision - it always buys, then always
sells, regardless of price or momentum. That's the point: it's a mechanics
test, not a trade you'd want your real session copying.
"""

import asyncio
import sys

from src.alpaca.mcp_client import MCPToolError, alpaca_mcp_session
from src.execution.broker import execute_order, get_account_state

DEFAULT_SYMBOL = "INTC"  # liquid, cheap enough that 1 share is a trivial amount
QTY = 1


async def main() -> int:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else DEFAULT_SYMBOL

    async with alpaca_mcp_session() as client:
        account = await get_account_state(client, symbols=[symbol])
        clock = account["clock"]

        if not clock.get("is_open", False):
            print("Market is closed - a market order won't fill until the next open.")
            print(f"Next open: {clock.get('next_open')}")
            return 1

        price = account["last_prices"].get(symbol)
        if not price:
            print(f"No current price for {symbol} - try a different symbol.")
            return 1

        print(f"Market open. {symbol} @ ${price:.2f}. Buying {QTY} share(s)...")

        buy = None
        try:
            buy = await execute_order(client, {"symbol": symbol, "action": "BUY", "target_qty": QTY})
        except (MCPToolError, ValueError) as exc:
            print(f"BUY failed before it could be placed: {exc}")
            return 1

        buy_fill = buy["fill"]
        print(f"BUY order result: {buy['order_result']}")
        print(f"BUY fill: {buy_fill}")

        if buy_fill.get("status") != "filled":
            print(
                f"\nBUY did not reach 'filled' (status: {buy_fill.get('status')}). "
                "Check the Alpaca dashboard before doing anything else - do NOT "
                "assume no position was opened."
            )
            return 1

        print(f"\nFilled at ${buy_fill.get('filled_avg_price')}. Selling it back now...")

        try:
            sell = await execute_order(client, {"symbol": symbol, "action": "SELL", "target_qty": QTY})
        except (MCPToolError, ValueError) as exc:
            print(f"\n!! SELL failed to even submit: {exc}")
            print(f"!! You are holding {QTY} share(s) of {symbol}. Close this manually on Alpaca.")
            return 1

        sell_fill = sell["fill"]
        print(f"SELL order result: {sell['order_result']}")
        print(f"SELL fill: {sell_fill}")

        if sell_fill.get("status") != "filled":
            print(
                f"\n!! SELL did not reach 'filled' (status: {sell_fill.get('status')}). "
                f"You may still be holding {symbol} - check the Alpaca dashboard."
            )
            return 1

        buy_price = buy_fill.get("filled_avg_price") or price
        sell_price = sell_fill.get("filled_avg_price") or price
        pnl = (sell_price - buy_price) * QTY

        print("\n=== ROUND TRIP COMPLETE ===")
        print(f"Bought {QTY} {symbol} @ {buy_price}, sold @ {sell_price}")
        print(f"P&L: ${pnl:+.2f} (a real, tiny paper P&L from a real fill - not a strategy result)")
        print("No open position left. Check the Alpaca dashboard to confirm.")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
