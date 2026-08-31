import os
import sys
from pathlib import Path

# Unit tests are offline: never trace scripted fake-model runs to LangSmith,
# even when the developer's .env has tracing enabled.
os.environ["LANGSMITH_TRACING"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))
