"""
MEOK Neural Learning Module — Shared across all MCP servers
Logs interactions and builds exact-input lookup patterns from rated examples.
This is a frequency lookup, not neural training or evidence of improved quality.

Usage in any MCP server:
    from meok_neural_learning import InteractionLogger, NeuralPredictor

    logger = InteractionLogger("fishkeeper-ai")
    logger.log("diagnose_disease", {"symptoms": "white spots"}, {"diagnosis": "ich"}, user_rating=5)

    predictor = NeuralPredictor("fishkeeper-ai")
    if predictor.ready:
        improved = predictor.predict("diagnose_disease", {"symptoms": "white spots"})
"""

import json
import sqlite3
import os
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MEOK_DATA_DIR = Path(os.environ.get("MEOK_DATA_DIR", str(Path.home() / ".meok-ai")))


class InteractionLogger:
    """Stores explicitly supplied interactions for local feedback analysis."""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.db_path = MEOK_DATA_DIR / server_name / "interactions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(str(self.db_path))) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT NOT NULL,
                    user_rating INTEGER DEFAULT NULL,
                    user_correction TEXT DEFAULT NULL,
                    latency_ms REAL DEFAULT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT DEFAULT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    accuracy REAL,
                    samples_trained INTEGER,
                    trained_at TEXT NOT NULL,
                    model_path TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool ON interactions(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON interactions(timestamp)")

    def log(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        user_rating: Optional[int] = None,
        user_correction: Optional[str] = None,
        latency_ms: Optional[float] = None,
        session_id: Optional[str] = None,
    ):
        _validate_rating(user_rating, allow_none=True)
        _input_key(input_data)
        with closing(sqlite3.connect(str(self.db_path))) as conn, conn:
            conn.execute(
                "INSERT INTO interactions (tool_name, input_data, output_data, user_rating, user_correction, latency_ms, timestamp, session_id) VALUES (?,?,?,?,?,?,?,?)",
                (
                    tool_name,
                    json.dumps(input_data, default=str),
                    json.dumps(output_data, default=str) if not isinstance(output_data, str) else output_data,
                    user_rating,
                    user_correction,
                    latency_ms,
                    datetime.now(timezone.utc).isoformat(),
                    session_id,
                ),
            )

    def get_training_data(self, tool_name: str, min_rating: int = 3) -> List[Dict]:
        _validate_rating(min_rating)
        with closing(sqlite3.connect(str(self.db_path))) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT input_data, output_data, user_rating FROM interactions WHERE tool_name=? AND user_rating >= ? AND (user_correction IS NULL OR TRIM(user_correction) = '') ORDER BY timestamp, id",
                (tool_name, min_rating),
            ).fetchall()
        return [{"input": json.loads(r["input_data"]), "output": json.loads(r["output_data"]) if r["output_data"].startswith("{") else r["output_data"], "rating": r["user_rating"]} for r in rows]

    def get_stats(self) -> Dict:
        with closing(sqlite3.connect(str(self.db_path))) as conn, conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            by_tool = conn.execute("SELECT tool_name, COUNT(*) as cnt FROM interactions GROUP BY tool_name ORDER BY cnt DESC").fetchall()
            rated = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_rating IS NOT NULL").fetchone()[0]
            corrected = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_correction IS NOT NULL").fetchone()[0]
        eligible_counts = self._eligible_counts()
        return {
            "server": self.server_name,
            "total_interactions": total,
            "by_tool": {row[0]: row[1] for row in by_tool},
            "rated": rated,
            "corrected": corrected,
            "ready_for_training": any(count >= 50 for count in eligible_counts.values()),
            "eligible_by_tool": eligible_counts,
            "db_path": str(self.db_path),
        }

    def _eligible_counts(self) -> Dict[str, int]:
        with closing(sqlite3.connect(str(self.db_path))) as conn, conn:
            rows = conn.execute(
                "SELECT tool_name, COUNT(*) FROM interactions WHERE user_rating BETWEEN 3 AND 5 AND (user_correction IS NULL OR TRIM(user_correction) = '') GROUP BY tool_name"
            ).fetchall()
        return dict(rows)


class NeuralPredictor:
    """Exact-input lookup over locally stored patterns; None means no match."""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.model_dir = MEOK_DATA_DIR / server_name / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        for model_file in self.model_dir.glob("*.json"):
            tool_name = model_file.stem
            try:
                self._models[tool_name] = json.loads(model_file.read_text())
            except Exception:
                pass

    @property
    def ready(self) -> bool:
        return len(self._models) > 0

    def predict(self, tool_name: str, input_data: Dict) -> Optional[Dict]:
        """Return a stored output for identical JSON input, or None; no quality claim."""
        if not isinstance(input_data, dict):
            raise ValueError("input_data must be a dictionary")
        input_key = _input_key(input_data)
        model = self._models.get(tool_name)
        if not model:
            return None
        patterns = model.get("patterns", [])
        for pattern in patterns:
            match = pattern.get("match")
            if isinstance(match, dict) and _input_key(match) == input_key:
                return pattern.get("output")
        return None

    def train(self, tool_name: str, training_data: List[Dict]) -> Dict:
        """Build lookup patterns from rated, uncorrected examples, not a neural model.

        Ratings are caller-supplied feedback, not authenticated human approval.
        Callers remain responsible for consent, rights, review and activation policy.
        """
        supplied_samples = len(training_data)
        training_data = [item for item in training_data if
                         type(item.get("rating")) is int and 3 <= item["rating"] <= 5
                         and not item.get("correction") and not item.get("user_correction")]
        for item in training_data:
            if not isinstance(item.get("input"), dict) or "output" not in item:
                raise ValueError("Each eligible example needs a dictionary input and an output")
            _input_key(item["input"])
        if len(training_data) < 50:
            return {"status": "insufficient_data", "samples": len(training_data), "supplied_samples": supplied_samples, "needed": 50}

        # Phase 1: Extract frequent input→output patterns
        from collections import Counter
        pattern_counts: Counter = Counter()
        for item in training_data:
            key = _input_key(item["input"])
            pattern_counts[key] += 1

        # Keep patterns seen 3+ times
        patterns = []
        for key, count in pattern_counts.most_common(100):
            if count >= 3:
                input_data = json.loads(key)
                # Find best output for this input
                matching = [item for item in training_data if _input_key(item["input"]) == key]
                best_rating = max(item["rating"] for item in matching)
                best = [item for item in matching if item["rating"] == best_rating]
                # Equal-rated conflicting answers require review; ordering is not adjudication.
                if len({json.dumps(item["output"], sort_keys=True, allow_nan=False) for item in best}) != 1:
                    continue
                patterns.append({"match": input_data, "output": best[0]["output"], "support_count": count, "best_rating": best_rating})

        if not patterns:
            return {"status": "no_usable_patterns", "patterns": 0, "samples": len(training_data)}

        model = {"tool_name": tool_name, "version": 1, "patterns": patterns, "trained_at": datetime.now(timezone.utc).isoformat(), "samples": len(training_data)}
        model_path = self.model_dir / f"{tool_name}.json"
        model_path.write_text(json.dumps(model, indent=2, default=str))
        self._models[tool_name] = model

        return {"status": "trained", "patterns": len(patterns), "samples": len(training_data), "model_path": str(model_path)}


# ── Training Dashboard Tool (add to any MCP server) ──

def get_learning_tools(server_name: str):
    """Returns MCP tool functions for the neural learning dashboard."""
    logger = InteractionLogger(server_name)
    predictor = NeuralPredictor(server_name)

    def get_learning_stats() -> str:
        """Get neural learning statistics for this MCP server."""
        stats = logger.get_stats()
        model_info = {name: {"patterns": len(m.get("patterns", [])), "trained_at": m.get("trained_at")} for name, m in predictor._models.items()}
        stats["models"] = model_info
        return json.dumps(stats, indent=2)

    def trigger_training(tool_name: str) -> str:
        """Build local exact-input patterns from rated, uncorrected interactions."""
        data = logger.get_training_data(tool_name)
        result = predictor.train(tool_name, data)
        return json.dumps(result, indent=2)

    def rate_last_interaction(tool_name: str, rating: int, correction: str = "") -> str:
        """Rate the quality of the last tool interaction (1-5) and optionally provide correction."""
        _validate_rating(rating)
        with closing(sqlite3.connect(str(logger.db_path))) as conn, conn:
            last = conn.execute("SELECT id FROM interactions WHERE tool_name=? ORDER BY id DESC LIMIT 1", (tool_name,)).fetchone()
            if last:
                conn.execute("UPDATE interactions SET user_rating=?, user_correction=? WHERE id=?", (rating, correction or None, last[0]))
                return json.dumps({"status": "rated", "interaction_id": last[0], "rating": rating})
        return json.dumps({"error": "No interactions found for this tool"})

    return get_learning_stats, trigger_training, rate_last_interaction


def _validate_rating(rating: Optional[int], allow_none: bool = False) -> None:
    if allow_none and rating is None:
        return
    if type(rating) is not int or not 1 <= rating <= 5:
        raise ValueError("rating must be an integer from 1 to 5")


def _input_key(value: Dict) -> str:
    """Preserve JSON value types and reject lossy/non-finite inputs."""
    if type(value) is not dict:
        raise ValueError("input_data must be a JSON object")

    def validate(item):
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("input contains a non-JSON value")

    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
