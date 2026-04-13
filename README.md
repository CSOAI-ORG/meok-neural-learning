# Adding Neural Learning to Any MEOK MCP Server

Add these 3 lines at the top of server.py (after imports):

```python
from meok_neural_learning import InteractionLogger, NeuralPredictor, get_learning_tools
_logger = InteractionLogger("YOUR-SERVER-NAME")
_predictor = NeuralPredictor("YOUR-SERVER-NAME")
```

Then in each tool function, add logging after the result:

```python
@mcp.tool()
def your_tool(arg: str) -> str:
    result = do_work(arg)
    _logger.log("your_tool", {"arg": arg}, result)  # ← add this line
    return result
```

And register the dashboard tools:

```python
stats_fn, train_fn, rate_fn = get_learning_tools("YOUR-SERVER-NAME")
mcp.tool()(stats_fn)
mcp.tool()(train_fn)
mcp.tool()(rate_fn)
```

This gives every server:
- `get_learning_stats` — See how many interactions logged, model status
- `trigger_training` — Train neural net from collected data
- `rate_last_interaction` — User feedback loop (1-5 rating + corrections)
