# MEOK interaction patterns

`meok-neural-learning` is a local SQLite interaction logger and an exact-input lookup library. Package and class names remain for compatibility. It does not train neural weights or demonstrate generalisation, autonomous improvement, legal compliance or production readiness.

## Use

```python
from meok_neural_learning import InteractionLogger, NeuralPredictor

logger = InteractionLogger("example-server")
logger.log("lookup", {"topic": "example"}, {"answer": "reviewed example"}, user_rating=5)
predictor = NeuralPredictor("example-server")
result = predictor.train("lookup", logger.get_training_data("lookup"))
answer = predictor.predict("lookup", {"topic": "example"})  # None if no exact match
```

Data is stored under `MEOK_DATA_DIR` or `~/.meok-ai`. Only log data you are authorised to retain and use. This library does not provide consent, redaction, identity verification, access control or a retention policy.

## Feedback and matching

- By default, only explicitly rated examples (3–5) without outstanding corrections are eligible. Unrated data is not an endorsement. A free-text correction excludes the old answer until the caller resolves it separately.
- At least 50 eligible examples for the requested tool are needed; each retained input appears at least three times. These are implementation thresholds, not statistical evidence of quality.
- The highest-rated answer is selected. Conflicting answers tied at the highest rating produce no pattern; row order cannot adjudicate disagreements.
- Prediction requires identical JSON input, including every key and value type. A missing or extra key returns `None`. An empty input pattern matches only an empty input.
- `support_count` records frequency, not confidence or a probability.
- An unsuccessful build preserves existing patterns. A successful `train` call still writes and activates local patterns immediately. Keep it behind your application's explicit review and authorisation boundary.

Ratings are caller-supplied feedback, not proof of a human reviewer's identity or mandate. This package alone does not implement Council OS review, reproduction, signing or GSPC admission. Do not attach a GSPC verdict to these patterns or present training-set reuse as held-out performance.

## Compatibility

Legacy patterns remain readable but now require exact matching. Former subset matching is intentionally removed. Unrated training examples now yield `insufficient_data`; conflicting top-rated answers are withheld. New patterns expose `support_count` and `best_rating` instead of the former uncalibrated `confidence` field.

Existing files have no admission record. Applications needing reviewed-only runtime use must quarantine them and separately validate their activation policy before loading them. This change does not automatically migrate or approve existing data.

New inputs must be JSON objects with string keys and recursively JSON-compatible values; tuples, arbitrary objects and non-finite numbers are rejected before storage. Use JSON-object outputs: the legacy output encoding does not reliably round-trip lists, scalars or text beginning with `{`. An explicit output-format migration remains separate work.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

Tests use temporary databases and make no network or model calls.

MIT — see [LICENSE](LICENSE).

Measurement, not certification. [Public Council verification](https://councilof.ai/verify).
