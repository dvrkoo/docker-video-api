import logging
import logging.handlers
import os
import sys

rotating = logging.handlers.RotatingFileHandler(
    os.getenv("LOG_FILE", "app.log"),
    maxBytes=1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)

stdout = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[rotating, stdout],
)

logger = logging.getLogger(__name__)
logger.info("Logging system initialized")
