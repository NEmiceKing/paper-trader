import asyncio
import logging
from typing import Optional

from ib_insync import IB

logger = logging.getLogger(__name__)


class IBKRClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib: Optional[IB] = None

    def connect(self):
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            raise

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    @property
    def is_connected(self) -> bool:
        return self.ib is not None and self.ib.isConnected()

    def reconnect(self):
        self.disconnect()
        self.connect()
