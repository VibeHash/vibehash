"""Tor and other SOCKS proxies for node traffic.

A proxy URL is `socks5h://host:port` (or `socks4a://`): the trailing h / a means the proxy resolves the node's name, so
nothing on this computer ever asks DNS who the node is. `socks5://` and `socks4://` resolve the name locally first and
hand the proxy an address, which tells whoever runs the resolver which node you use; they are refused, not warned about.

Two planes (client.NodeClient): the control plane (JSON: status, transactions, units) always goes through the proxy; the
data plane (checkpoints and datasets, which can be gigabytes) goes direct under tor mode "control-only" when large, and
is refused under "strict". The process's choice is set once by cli.main (`activate`) and is what a NodeClient or Hub
made without an explicit one uses. Release checks (update.py) are not node traffic and are not routed here.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

TOR_PORT, TOR_BROWSER_PORT = 9050, 9150
MODES = ("control-only", "strict")
STRICT_COMMANDS = frozenset({"wallet", "send", "stake", "unstake", "job post", "post-job"})
SCHEMES = {"socks5h": 5, "socks4a": 4}
_LEAK = ("{scheme}:// resolves DNS on this computer and so leaks the node's name to your resolver; "
         "use socks5h:// (the proxy resolves it), e.g. socks5h://127.0.0.1:9050")

_active: tuple[str | None, str | None] = (None, None)


def activate(url: str | None, mode: str | None) -> None:
    """Make (url, mode) what NodeClient and Hub use when not told otherwise. (None, None) is no proxy."""
    global _active
    _active = (url, mode)


def active() -> tuple[str | None, str | None]:
    return _active


def check(url: str) -> str:
    """`url` if it is socks5h://host:port or socks4a://host:port; ValueError otherwise, saying why."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in ("socks5", "socks4", "http", "https"):
        raise ValueError(_LEAK.format(scheme=scheme))
    if scheme not in SCHEMES:
        raise ValueError(f"can't use {url!r} as a proxy: it must look like socks5h://127.0.0.1:9050")
    try:
        port = parts.port
    except ValueError:
        port = None
    if not parts.hostname or not port:
        raise ValueError(f"can't use {url!r} as a proxy: it needs a host and a port, like socks5h://127.0.0.1:9050")
    return url


def _listening(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


def resolve(args=None, cfg: dict | None = None) -> str | None:
    """The proxy to use: --proxy, else --tor (Tor on 9050, or Tor Browser's 9150 if only that one listens), else the
    saved "proxy" setting; None when none is set. Raises ValueError for a proxy URL that would leak (check)."""
    flag = getattr(args, "proxy", None)
    if flag:
        return check(flag)
    if getattr(args, "tor", False):
        if not _listening(TOR_PORT) and _listening(TOR_BROWSER_PORT):
            return f"socks5h://127.0.0.1:{TOR_BROWSER_PORT}"
        return f"socks5h://127.0.0.1:{TOR_PORT}"
    saved = (cfg or {}).get("proxy")
    return check(saved) if saved else None


def tor_mode(args, cfg: dict | None, command: str) -> str | None:
    """--tor-mode, else the saved "tor_mode", else by command: strict for what moves money or keys, control-only for
    what runs for hours and moves checkpoints. None when no proxy is set (a flag or a setting)."""
    cfg = cfg or {}
    if not (getattr(args, "proxy", None) or getattr(args, "tor", False) or cfg.get("proxy")):
        return None
    chosen = getattr(args, "tor_mode", None) or cfg.get("tor_mode")
    if chosen:
        if chosen not in MODES:
            raise ValueError(f"tor mode must be one of {', '.join(MODES)}, not {chosen!r}")
        return chosen
    return "strict" if command in STRICT_COMMANDS else "control-only"


def is_onion(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower().endswith(".onion")


def is_local(url: str) -> bool:
    """Whether `url` is this computer (localhost or a loopback address): Tor can't reach it, and nothing about it leaks."""
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def onion_hint(status) -> str | None:
    """What to tell someone using a proxy whose node (its /status) has a hidden-service address; None otherwise."""
    onion = status.get("onion") if isinstance(status, dict) else None
    if not (_active[0] and isinstance(onion, str) and onion.lower().endswith(".onion") and len(onion) < 100
            and onion.isprintable()):
        return None
    return f"reach this node at {onion} to avoid exit-node blocks"


def handler(proxy_url: str, context=None, pin: str | None = None):
    """The urllib handler that opens http and https URLs through `proxy_url`, giving the proxy the host name to resolve
    (rdns). For https, `context` verifies the server and `pin` (a SHA-256 fingerprint, tls.py) must match its
    certificate; both are the node's trust as tls.opener_for works it out."""
    import socks
    import sockshandler

    parts = urlsplit(check(proxy_url))
    kind = socks.PROXY_TYPE_SOCKS5 if SCHEMES[parts.scheme.lower()] == 5 else socks.PROXY_TYPE_SOCKS4

    class _Connection(sockshandler.SocksiPyConnectionS):
        def connect(self):
            super().connect()
            if pin:
                from . import tls

                got = tls.fingerprint(self.sock.getpeercert(binary_form=True) or b"")
                if got != pin:
                    self.close()
                    import ssl

                    raise ssl.SSLCertVerificationError(1, f"the node's certificate isn't the pinned one (it has sha256 "
                                                          f"{got}, expected {pin}); refusing to talk to it")

    class _Handler(sockshandler.SocksiPyHandler):
        def https_open(self, req):
            def build(host, port=None, timeout=0, **kw):
                kw = sockshandler.merge_dict(self.kw, kw)
                if context is not None:
                    kw["context"] = context
                return _Connection(*self.args, host=host, port=port, timeout=timeout, **kw)

            return self.do_open(build, req)

    return _Handler(kind, parts.hostname, parts.port, rdns=True)


def is_proxied(opener) -> bool:
    """Whether `opener` sends through a proxy (a SOCKS handler is in it)."""
    if opener is None:
        return False
    import sockshandler

    return any(isinstance(h, sockshandler.SocksiPyHandler) for h in opener.handlers)
