"""
MEOK Neural Learning Module — Shared across all MCP servers
Logs interactions, trains lightweight models, improves predictions over time.

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

MEOK_DATA_DIR = Path(os.environ.get("MEOK_DATA_DIR", str(Path.home() / ".meok-ai")))


class InteractionLogger:
    """Logs every MCP tool call for future neural net training."""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.db_path = MEOK_DATA_DIR / server_name / "interactions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
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
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO interactions (tool_name, input_data, output_data, user_rating, user_correction, latency_ms, timestamp, session_id) VALUES (?,?,?,?,?,?,?,?)",
                (
                    tool_name,
                    json.dumps(input_data, default=str),
                    json.dumps(output_data, default=str) if not isinstance(output_data, str) else output_data,
                    user_rating,
                    user_correction,
                    latency_ms,
                    datetime.utcnow().isoformat(),
                    session_id,
                ),
            )

    def get_training_data(self, tool_name: str, min_rating: int = 3) -> List[Dict]:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT input_data, output_data, user_rating FROM interactions WHERE tool_name=? AND (user_rating IS NULL OR user_rating >= ?) ORDER BY timestamp",
                (tool_name, min_rating),
            ).fetchall()
        return [{"input": json.loads(r["input_data"]), "output": json.loads(r["output_data"]) if r["output_data"].startswith("{") else r["output_data"], "rating": r["user_rating"]} for r in rows]

    def get_stats(self) -> Dict:
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            by_tool = conn.execute("SELECT tool_name, COUNT(*) as cnt FROM interactions GROUP BY tool_name ORDER BY cnt DESC").fetchall()
            rated = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_rating IS NOT NULL").fetchone()[0]
            corrected = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_correction IS NOT NULL").fetchone()[0]
        return {
            "server": self.server_name,
            "total_interactions": total,
            "by_tool": {row[0]: row[1] for row in by_tool},
            "rated": rated,
            "corrected": corrected,
            "ready_for_training": total >= 100,
            "db_path": str(self.db_path),
        }


class NeuralPredictor:
    """Lightweight prediction engine — uses trained models when available, falls back to None."""

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
        """Return improved prediction if model exists, else None (use rule-based)."""
        model = self._models.get(tool_name)
        if not model:
            return None
        # Phase 1: Pattern matching from historical data
        # Phase 2: Replace with actual PyTorch/MLX model inference
        patterns = model.get("patterns", [])
        for pattern in patterns:
            if all(input_data.get(k) == v for k, v in pattern.get("match", {}).items()):
                return pattern.get("output")
        return None

    def train(self, tool_name: str, training_data: List[Dict]) -> Dict:
        """Train a lightweight model from interaction data. Phase 1: frequency-based patterns."""
        if len(training_data) < 50:
            return {"status": "insufficient_data", "samples": len(training_data), "needed": 50}

        # Phase 1: Extract frequent input→output patterns
        from collections import Counter
        pattern_counts: Counter = Counter()
        for item in training_data:
            key = json.dumps(item["input"], sort_keys=True, default=str)
            pattern_counts[key] += 1

        # Keep patterns seen 3+ times
        patterns = []
        for key, count in pattern_counts.most_common(100):
            if count >= 3:
                input_data = json.loads(key)
                # Find best output for this input
                outputs = [item["output"] for item in training_data if json.dumps(item["input"], sort_keys=True, default=str) == key]
                rated = [item for item in training_data if json.dumps(item["input"], sort_keys=True, default=str) == key and item.get("rating")]
                best_output = rated[0]["output"] if rated else outputs[0]
                patterns.append({"match": input_data, "output": best_output, "confidence": min(count / 10, 1.0)})

        model = {"tool_name": tool_name, "version": 1, "patterns": patterns, "trained_at": datetime.utcnow().isoformat(), "samples": len(training_data)}
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
        """Trigger neural net training for a specific tool using collected interaction data."""
        data = logger.get_training_data(tool_name)
        result = predictor.train(tool_name, data)
        return json.dumps(result, indent=2)

    def rate_last_interaction(tool_name: str, rating: int, correction: str = "") -> str:
        """Rate the quality of the last tool interaction (1-5) and optionally provide correction."""
        with sqlite3.connect(str(logger.db_path)) as conn:
            last = conn.execute("SELECT id FROM interactions WHERE tool_name=? ORDER BY id DESC LIMIT 1", (tool_name,)).fetchone()
            if last:
                conn.execute("UPDATE interactions SET user_rating=?, user_correction=? WHERE id=?", (rating, correction or None, last[0]))
                return json.dumps({"status": "rated", "interaction_id": last[0], "rating": rating})
        return json.dumps({"error": "No interactions found for this tool"})

    return get_learning_stats, trigger_training, rate_last_interaction
