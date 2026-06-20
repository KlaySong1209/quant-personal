# How To Build A Strategy

The repository currently ships only a trivial placeholder. It is not a real
strategy and not advice.

A future strategy is a Python class under `src/quant/strategy/` with:

```python
def generate_weights(self, prices):
    ...
```

Input: adjusted daily prices as a timestamp-by-symbol DataFrame.

Output: target weights with the same index and columns.

Do not shift your signals. The engine applies the one-period execution shift.

Create a new file under `src/quant/strategy/` and register it in
`src/quant/strategy/__init__.py`. Keep data loading, account state, and broker
simulation out of the strategy file.

Common mistakes guarded by the repo:

- same-day look-ahead: engine shifts targets by one period
- silent risk clipping: risk checks reject instead
- zero transaction costs: rejected unless explicitly marked for tests
- survivorship bias: use point-in-time universe config
