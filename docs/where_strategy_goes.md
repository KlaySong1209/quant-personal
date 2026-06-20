# Where Strategy Logic Goes

The current strategy is only a trivial placeholder. It exists so local data
ingestion, backtesting, and the simulated paper account can be tested end to end.
It is not a trading strategy and not investment advice.

When you later have a strategy idea, put it in one Python file under:

```text
src/quant/strategy/
```

The class should implement:

```python
def generate_weights(self, prices):
    ...
```

Input: adjusted daily prices as a timestamp-by-symbol table.

Output: target weights with the same dates and symbols.

Register the class in:

```text
src/quant/strategy/__init__.py
```

Do not shift signals inside the strategy. The backtest engine already applies
the one-period execution shift. The paper account receives daily target weights
and records simulated account state only.
