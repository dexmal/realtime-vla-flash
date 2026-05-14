import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.

    Each infer response dict may include ``server_timing`` (milliseconds):

    - ``ws_recv_ms``: await ``recv`` (wait for bytes on the wire).
    - ``ws_unpack_ms``: msgpack decode of the observation payload.
    - ``infer_ms`` / ``policy_time_ms``: wall time inside ``policy.infer`` (including policy preprocessing).
    - ``ws_pack_ms``: msgpack encode of the action payload (measured on a first pack; value is included in a second pack).
    - ``serve_time_ms``: server-side request handling time excluding network wait/send (unpack + policy + pack).
    - ``prev_total_ms`` / ``prev_ws_send_ms``: like before, timings attributed to the *previous* response's handler.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        reset_policy_on_connect: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._reset_policy_on_connect = bool(reset_policy_on_connect)
        self._reset_policy_connection_active = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        if self._reset_policy_on_connect:
            if self._reset_policy_connection_active:
                logger.warning("Rejecting concurrent policy connection from %s", websocket.remote_address)
                await websocket.close(code=1013, reason="Another policy connection is active.")
                return
            self._reset_policy_connection_active = True
            self._policy.reset()
            logger.info("Reset policy state for new connection from %s", websocket.remote_address)
        try:
            packer = msgpack_numpy.Packer()

            await websocket.send(packer.pack(self._metadata))

            prev_total_time = None
            prev_send_time = None
            while True:
                try:
                    t0 = time.monotonic()
                    raw = await websocket.recv()
                    t1 = time.monotonic()
                    recv_timestamp_s = time.time()
                    obs = msgpack_numpy.unpackb(raw)
                    t2 = time.monotonic()
                    infer_t0 = time.monotonic()
                    action = self._policy.infer(obs)
                    infer_t1 = time.monotonic()

                    if not isinstance(action, dict):
                        raise TypeError(f"policy.infer must return dict, got {type(action)!r}")

                    pol_st = action.get("server_timing")
                    st: dict[str, float] = {}
                    if isinstance(pol_st, dict):
                        for k, v in pol_st.items():
                            if isinstance(v, (int, float)):
                                st[str(k)] = float(v)

                    st["ws_recv_ms"] = (t1 - t0) * 1000.0
                    st["ws_unpack_ms"] = (t2 - t1) * 1000.0
                    st["infer_ms"] = (infer_t1 - infer_t0) * 1000.0
                    st["policy_time_ms"] = st["infer_ms"]
                    st["server_recv_timestamp_s"] = recv_timestamp_s

                    if prev_total_time is not None:
                        st["prev_total_ms"] = prev_total_time * 1000.0
                    if prev_send_time is not None:
                        st["prev_ws_send_ms"] = prev_send_time * 1000.0

                    action["server_timing"] = st
                    t_pack0 = time.monotonic()
                    packed = packer.pack(action)
                    t_pack1 = time.monotonic()
                    pack_ms = (t_pack1 - t_pack0) * 1000.0
                    st["ws_pack_ms"] = pack_ms
                    st["serve_time_ms"] = st["ws_unpack_ms"] + st["policy_time_ms"] + st["ws_pack_ms"]
                    st["server_response_timestamp_s"] = time.time()
                    action["server_timing"] = st
                    packed = packer.pack(action)

                    t_send0 = time.monotonic()
                    await websocket.send(packed)
                    t_done = time.monotonic()
                    prev_send_time = t_done - t_send0
                    prev_total_time = t_done - t0

                    logger.debug(
                        "ws_timing recv=%.2f unpack=%.2f infer=%.2f pack=%.2f send=%.2f total=%.2f ms",
                        st["ws_recv_ms"],
                        st["ws_unpack_ms"],
                        st["infer_ms"],
                        pack_ms,
                        prev_send_time * 1000.0,
                        prev_total_time * 1000.0,
                    )

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            if self._reset_policy_on_connect:
                self._reset_policy_connection_active = False


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
