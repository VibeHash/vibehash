"""What a node does for the people who run the network (the admin app, admin.py): a few routes behind a token,
served only when the node was started with one (`vibehash node --admin-token`, or $VIBEHASH_ADMIN_TOKEN).

    GET  /admin/overview              this node, its jobs (held or not, and why they wait), the devices that registered
                                      lately and where they connected from, the validators and peers, the mempool
    POST /admin/jobs/<id>/hold        deny: this node leads no round for the job until it is released ({"why": ...})
    POST /admin/jobs/<id>/release     confirm: the job takes its turns again
    POST /admin/review                {"on": true}: every job posted from now on starts held until an admin confirms it
    POST /admin/login                 {"code": "123456"}: an authenticator app's code for a sign-in good for 12 hours
    POST /admin/prune                 delete checkpoints no recent round needs (only between rounds): bytes freed
    POST /admin/backup                ask this server's backup service (setup-node.sh) for a backup now
    POST /admin/restart               {"force": false}: end the node so systemd starts it again; 409 when that would
                                      leave the validators online with two thirds of the stake or less

/admin/overview's "ops" is the server's health for the admin app's Infrastructure panel: the release, how long the
node has run, the disk and what its checkpoints take, the last backup ($DATA/backup/last_backup.json, written by the
backup service) and health check ($DATA/health.json), whether it halted, and the quorum: which validators signed a
block lately (from the commit certificates) and whether finality survives this node going away for a restart.

The three operations above act on the server itself, so each needs a fresh authenticator code as well
(`check(..., fresh=True)`: X-Admin-Code, taken once like a sign-in's code); a node without an authenticator key
(--admin-mfa) refuses them.

Every request carries `Authorization: Bearer <token>`; without the right one the answer is 401, and a node with no
token answers 404 to all of them, as if they weren't there. The token is compared in constant time and never logged,
and one shorter than MIN_TOKEN characters is refused at start (`vibehash admin --new-token` makes a good one).

A second factor (`node --admin-mfa KEY`, or $VIBEHASH_ADMIN_MFA): the key an authenticator app holds (Google or
Microsoft Authenticator, 1Password, ...; `vibehash admin --new-mfa` makes one and says how to add it). With it, the
token alone opens nothing but /admin/login, which takes the app's current 6-digit code (TOTP, RFC 6238: 30-second
steps, one step either side for a clock that drifts, each code good once) and gives a sign-in, sent from then on as
X-Admin-Login; it lasts SESSION_S and is kept in memory only, so a restart signs everyone out. Every admin of a node
adds the same key to their own app. A stolen token is then not enough: the phone is needed too.

Wrong guesses: an address that sends MAX_FAILS wrong tokens or codes within FAIL_WINDOW_S is turned away (403,
Retry-After) for the rest of that window, before its token is even looked at; and MAX_CODE_FAILS wrong codes from
anywhere, which take the right token to send, stop codes being taken at all for the window, since someone holding the
token is trying codes. A wrong token from the open internet can't lock an admin out: it counts against its own address. Behind a reverse
proxy that holds only if the proxy sets X-Forwarded-For (`client_ip`): without it every client looks like the proxy,
and anyone's wrong guesses lock everyone out.

The three orders (hold, release, review) change what the network does, so on a node with an authenticator key each
needs a fresh code on top of the sign-in (`check(..., fresh=True)`: X-Admin-Code, the app's code of now, taken once
like a sign-in's, so not the one the sign-in just used). /admin/overview and the device list need only the sign-in.
A node with no key has no code to ask for; one bound beyond this computer refuses to start without a key
(`vibehash node`; --admin-allow-no-mfa for a test rig).

A hold is this node's own policy, not a rule of the chain: the coordinator skips a held job when it picks what the
next round trains (Coordinator.policy -> `reason`), and the job's budget stays in escrow untouched. Another validator
that doesn't hold the job still leads rounds for it, so the admin app tells every node it knows. The holds and the
review setting are kept in the node's database (Store meta "admin"), so they survive a restart.

Devices: a worker's registration comes through POST /register, and the node notes the address it came from (the
proxy's X-Forwarded-For when the connection is the proxy's own, else the socket's peer) with what the worker said of
its machine. Kept in memory for the admin app's map; never served on a public route, since an IP is a person's."""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

from . import __version__, hardware
from .consensus import quorum
from .crypto import address_of
from .ledger import SYSTEM

log = logging.getLogger("vibehash.admin")

META = "admin"
MAX_DEVICES = 5000
ONLINE_S = 600  # a device that registered in the last 10 minutes is online, as netstats counts miners
MAX_WHY = 200

MIN_TOKEN = 24  # characters; `new_token` gives 43, about 256 bits
MIN_MFA_BYTES = 16  # an authenticator key of at least 128 bits (`new_mfa_key` gives 160, the usual)
CODE_STEP = 30  # seconds per authenticator code
CODE_DIGITS = 6
SESSION_S = 12 * 3600  # how long a sign-in with a code lasts
MAX_SESSIONS = 16
LOGIN_HEADER = "X-Admin-Login"
NEED_CODE = "an authenticator code is needed: sign in with POST /admin/login"  # (the admin app looks for these words)
CODE_HEADER = "X-Admin-Code"  # a fresh authenticator code, for operations on the server itself (check(fresh=True))
NEED_FRESH = "this needs a fresh authenticator code"  # (the admin app looks for these words)
SIGNED_WINDOW = 30  # blocks: a validator whose vote is in one of the last this many commit certificates is online
BACKUP_REQUEST_S = 600  # a backup asked for this recently, and not begun, isn't asked for again
MAX_FAILS = 10  # wrong tokens or codes from one address within the window: it is turned away for the rest of it
FAIL_WINDOW_S = 900
MAX_CODE_FAILS = 30  # wrong codes from anywhere within the window: codes are refused for the rest of it
MAX_TRACKED = 10_000  # addresses whose wrong tries are remembered


class Forbidden(Exception):
    """A request for an admin route without the right token, or the right code (401)."""


class NeedCode(Forbidden):
    """The token is right but the node also wants an authenticator code, and this request has no sign-in (401)."""


class Conflict(Exception):
    """An operation that would do harm now, or can't be done here (409): what to do instead is in the message."""


class TooMany(Exception):
    """Too many wrong tries lately (403, with Retry-After: a 429 would have clients wait and resend)."""

    def __init__(self, msg: str, retry_after: float):
        super().__init__(msg)
        self.retry_after = max(1, int(retry_after))


def new_token() -> str:
    return secrets.token_urlsafe(32)


def new_mfa_key() -> str:
    """A fresh authenticator key, in the base32 an authenticator app takes as a "setup key"."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def mfa_key(text: str) -> bytes:
    """An authenticator key's bytes from its base32 (spaces, dashes and case don't matter); ValueError if it isn't one."""
    clean = re.sub(r"[\s-]", "", str(text or "")).upper()
    try:
        key = base64.b32decode(clean + "=" * (-len(clean) % 8))
    except (ValueError, TypeError):
        raise ValueError("the authenticator key isn't base32 (letters A-Z and digits 2-7): make one with "
                         "`vibehash admin --new-mfa`") from None
    if len(key) < MIN_MFA_BYTES:
        raise ValueError(f"the authenticator key is too short ({len(key) * 8} bits, at least {MIN_MFA_BYTES * 8}): "
                         "make one with `vibehash admin --new-mfa`")
    return key


def check_token(token: str | None) -> None:
    """ValueError for a token too short to be safe from guessing."""
    if token and len(token) < MIN_TOKEN:
        raise ValueError(f"the admin token is {len(token)} characters; use at least {MIN_TOKEN} (`vibehash admin "
                         "--new-token` makes one)")


def totp(key: bytes, counter: int, digits: int = CODE_DIGITS) -> str:
    """The code an authenticator app shows for this 30-second step (RFC 6238 over RFC 4226, HMAC-SHA1)."""
    mac = hmac.new(key, int(counter).to_bytes(8, "big"), hashlib.sha1).digest()
    at = mac[-1] & 0x0F
    return str((int.from_bytes(mac[at:at + 4], "big") & 0x7FFFFFFF) % 10**digits).zfill(digits)


def otpauth_uri(key_text: str, label: str = "admin") -> str:
    """The link an authenticator app opens to add the key (the same thing a QR code holds)."""
    secret = re.sub(r"[\s-]", "", key_text).upper()
    return (f"otpauth://totp/{quote('Vibe Hash:' + label)}?secret={secret}&issuer={quote('Vibe Hash')}"
            f"&algorithm=SHA1&digits={CODE_DIGITS}&period={CODE_STEP}")


def _local(text: str) -> bool | None:
    """Whether an address is this computer's or a private network's (a reverse proxy's); None when it isn't one."""
    try:
        addr = ipaddress.ip_address(text.split("%")[0])
    except ValueError:
        return None
    return addr.is_loopback or addr.is_private


def client_ip(peer: str, forwarded_for: str | None) -> str:
    """Where a request came from. When the connection itself is from this computer or a private network (a reverse
    proxy in front of the node), X-Forwarded-For is read from its right end, the hop the proxy itself appended,
    skipping any further proxies on local addresses: the first address from the open internet is the client's. The
    header's left end is whatever the client chose to send, so it is never believed, and a header on a connection
    from the open internet is ignored altogether."""
    if _local(peer) and forwarded_for:
        for hop in reversed([h.strip() for h in forwarded_for.split(",")]):
            local = _local(hop)
            if local is None:
                break  # not an address: nothing past it can be trusted
            if not local:
                return hop
    return peer


class Admin:
    def __init__(self, store, token: str | None, now=time.time, mfa: str | None = None):
        check_token(token)
        self.store, self.token, self.now = store, token or None, now
        if mfa and not self.token:
            raise ValueError("an authenticator key (--admin-mfa) needs an admin token (--admin-token) too")
        self.mfa: bytes | None = mfa_key(mfa) if mfa else None
        self.lock = threading.Lock()
        self.sessions: dict[str, float] = {}  # a sign-in -> when it ends
        self.last_code = -1  # the step of the last code taken: no code is taken twice
        self.fails: dict[str, list[float]] = {}  # an address -> when it sent a wrong token or code, lately
        self.code_fails: list[float] = []  # when a wrong code came, from anywhere
        saved = (store.load_meta(META) if store is not None else None) or {}
        self.held: dict[str, dict] = dict(saved.get("held") or {})  # job id -> {"why", "when"}
        self.review: bool = bool(saved.get("review", False))
        self.confirmed: set[str] = set(saved.get("confirmed") or [])  # jobs an admin let through under review
        self.devices: dict[str, dict] = {}  # worker address -> what it registered with, and from where
        self.changed = lambda job_id: None  # the node hooks this to drop a job's "waits" line when it is released

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def check(self, authorization: str | None, login: str | None = None, ip: str | None = None,
              code: str | None = None, fresh: bool = False) -> None:
        """A request for an admin route: KeyError (404) on a node with no token; TooMany (403) from an address with
        too many wrong tries lately; Forbidden (401) with the wrong token; NeedCode (401) when the node wants an
        authenticator code and `login` (X-Admin-Login) is no sign-in of now. `fresh`: an operation on the server
        itself, which also takes `code` (X-Admin-Code), the authenticator app's code of now, used once; Forbidden on a
        node with no authenticator key, which can't ask for one."""
        self._token(authorization, ip)
        if self.mfa is None:
            if fresh:
                raise Forbidden(f"{NEED_FRESH}, and this node has no authenticator key: start it with --admin-mfa "
                                "(`vibehash admin --new-mfa` makes one)")
            return
        with self.lock:
            ends = self.sessions.get(login or "")
            if ends is None or ends <= self.now():
                self.sessions.pop(login or "", None)
                raise NeedCode(NEED_CODE)
        if fresh:
            if not re.sub(r"\s", "", str(code or "")):
                raise Forbidden(f"{NEED_FRESH} ({CODE_HEADER})")
            self._take_code(code, ip)

    def login(self, authorization: str | None, code, ip: str | None = None) -> dict:
        """POST /admin/login: the token and the authenticator app's code of now give a sign-in for SESSION_S."""
        self._token(authorization, ip)
        if self.mfa is None:
            return {"ok": True, "mfa": False, "session": None, "expires": None}
        self._take_code(code, ip)
        with self.lock:
            now = self.now()
            session = secrets.token_urlsafe(32)
            self.sessions = {s: t for s, t in self.sessions.items() if t > now}
            while len(self.sessions) >= MAX_SESSIONS:  # the one ending soonest goes
                del self.sessions[min(self.sessions, key=self.sessions.get)]
            self.sessions[session] = now + SESSION_S
        log.info("admin: signed in with a code from %s", ip or "?")
        return {"ok": True, "mfa": True, "session": session, "expires": now + SESSION_S}

    def _take_code(self, code, ip: str | None) -> None:
        """The authenticator app's code of now (one step either side), taken once: Forbidden if it's wrong or used,
        TooMany after too many wrong codes. A wrong one counts against its address and against codes as a whole."""
        code = re.sub(r"\s", "", str(code or ""))
        with self.lock:
            now = self.now()
            self.code_fails = [t for t in self.code_fails if now - t < FAIL_WINDOW_S]
            if len(self.code_fails) >= MAX_CODE_FAILS:
                wait = FAIL_WINDOW_S - (now - self.code_fails[0])
                raise TooMany(f"too many wrong codes lately, from anyone: codes are refused for {_minutes(wait)} "
                              "(someone with the token may be guessing: make a new token)", wait)
            step, taken = int(now // CODE_STEP), None
            if len(code) == CODE_DIGITS and code.isdigit():
                for c in (step - 1, step, step + 1):
                    if hmac.compare_digest(totp(self.mfa, c), code):
                        taken = c
            if taken is not None and taken <= self.last_code:  # right, but used: not a guess, so no strike
                raise Forbidden("that code was just used: wait for the next one")
            if taken is None:
                self._strike(ip, now)
                self.code_fails.append(now)
                log.warning("admin: a wrong authenticator code from %s, with the right token", ip or "?")
                raise Forbidden("the authenticator code is wrong")
            self.last_code = taken

    def _token(self, authorization: str | None, ip: str | None) -> None:
        if not self.enabled:
            raise KeyError("admin")
        with self.lock:
            now = self.now()
            recent = [t for t in self.fails.get(ip or "?", ()) if now - t < FAIL_WINDOW_S]
            if len(recent) >= MAX_FAILS:
                wait = FAIL_WINDOW_S - (now - recent[0])
                raise TooMany(f"too many wrong tries from this address: try again in {_minutes(wait)}", wait)
        given = (authorization or "").strip()
        if given[:7].lower() == "bearer ":
            given = given[7:].strip()
        if not given or not hmac.compare_digest(given.encode(), self.token.encode()):
            with self.lock:
                self._strike(ip, self.now())
            log.warning("admin: a wrong admin token from %s", ip or "?")
            raise Forbidden("the admin token is missing or wrong")

    def _strike(self, ip: str | None, now: float) -> None:
        """A wrong try from `ip` (the lock held)."""
        key = ip or "?"
        self.fails[key] = [t for t in self.fails.get(key, ()) if now - t < FAIL_WINDOW_S] + [now]
        if len(self.fails) > MAX_TRACKED:  # the addresses whose last try is oldest go first
            for old in sorted(self.fails, key=lambda a: self.fails[a][-1])[:len(self.fails) - MAX_TRACKED]:
                del self.fails[old]

    # --- the job policy ------------------------------------------------------------------

    def reason(self, job_id: str, owner: str | None) -> str | None:
        """Why this node leads no round for the job, or None: it is held, or review is on and no admin confirmed it
        (a starter job, owned by nobody, is never held that way)."""
        if job_id in self.held:  # the reason given stays with the admins (`held` in the overview): this text is on /status
            return "an admin holds it"
        if self.review and owner != SYSTEM and job_id not in self.confirmed:
            return "it awaits an admin's review"
        return None

    def hold(self, job_id: str, why: str = "") -> dict:
        with self.lock:
            self.held[job_id] = {"why": str(why or "")[:MAX_WHY], "when": self.now()}
            self.confirmed.discard(job_id)
            self._save()
        self.changed(job_id)
        return {"ok": True, "job_id": job_id, "held": self.held[job_id]}

    def release(self, job_id: str) -> dict:
        with self.lock:
            self.held.pop(job_id, None)
            self.confirmed.add(job_id)
            self._save()
        self.changed(job_id)
        return {"ok": True, "job_id": job_id, "held": None}

    def set_review(self, on: bool, current_jobs=()) -> dict:
        """Turning review on confirms every job already on chain, so only new ones wait."""
        with self.lock:
            self.review = bool(on)
            if self.review:
                self.confirmed.update(current_jobs)
            self._save()
        for job_id in current_jobs:
            self.changed(job_id)
        return {"ok": True, "review": self.review}

    def _save(self) -> None:
        if self.store is not None:
            self.store.save_meta(META, {"held": self.held, "review": self.review, "confirmed": sorted(self.confirmed)})

    # --- devices -----------------------------------------------------------------------------

    def saw(self, address: str, ip: str | None, body: dict, round_no: int | None = None) -> None:
        """A worker registered: what it said of its machine, and where from."""
        klass = body.get("hardware")
        entry = {"address": address, "ip": ip, "when": self.now(), "round": round_no,
                 "flops_per_sec": body.get("flops_per_sec"), "memory_mb": body.get("memory_mb"),
                 "hardware": klass if hardware.valid_class(klass) else None}
        with self.lock:
            self.devices[address] = entry
            if len(self.devices) > MAX_DEVICES:  # the ones not seen for longest go first
                for old in sorted(self.devices, key=lambda a: self.devices[a]["when"])[:len(self.devices) - MAX_DEVICES]:
                    del self.devices[old]

    def device_list(self) -> list[dict]:
        now = self.now()
        with self.lock:
            out = [dict(d, online=now - d["when"] <= ONLINE_S) for d in self.devices.values()]
        out.sort(key=lambda d: -d["when"])
        return out

    # --- the overview --------------------------------------------------------------------------

    def overview(self, node) -> dict:
        """Everything the admin app shows of this node, in one answer."""
        now = self.now()
        s = node.chain.state
        net = node.network()  # cached: the jobs' summaries (netstats), the census
        summaries = {j["id"]: j for j in net.get("jobs", [])}
        with node.coord.lock:
            r = node.coord.r
            mempool = [{"txid": tx.txid.hex(), "kind": tx.kind, "sender": tx.sender, "recipient": tx.recipient,
                        "amount": tx.amount, "fee": tx.fee, "memo": tx.memo[:120]} for tx in node.coord.mempool[:50]]
            waiting = dict(node.coord.waiting)
            rnd = {"no": r.round_no if r else None, "phase": node.coord.phase.value, "job_id": r.job_id if r else None,
                   "registered": len(r.regs) if r else 0}
        jobs = []
        for job_id, spec in list(s.jobs.items()):
            owner = s.job_owner.get(job_id)
            summary = summaries.get(job_id, {})
            jobs.append({"id": job_id, "owner": owner, "starter": owner == SYSTEM, "escrow": s.escrow.get(job_id, 0),
                         "model": summary.get("model"), "status": summary.get("status"),
                         "fee_per_round": summary.get("fee_per_round"), "rounds_trained": summary.get("rounds_trained"),
                         "units": summary.get("units"), "metric": summary.get("metric"), "start": summary.get("start"),
                         "now": summary.get("now"), "last_trained": summary.get("last_trained"),
                         "held": self.held.get(job_id), "policy": self.reason(job_id, owner), "waiting": waiting.get(job_id),
                         "spec_bytes": len(spec)})
        voting = s.validator_set()
        dead = dict(node.peers.dead_until)
        return {
            "now": now, "review": self.review,
            "node": {"url": node.announce, "address": node.keys.address, "version": net.get("version") or node.status().get("version"),
                     "chain_id": node.chain.chain_id, "height": node.chain.tip.height, "tip": node.chain.tip.hash.hex(),
                     "tip_time": node.chain.tip.timestamp, "round": rnd, "halted": node.halted_public,
                     "validator": node.keys.address in voting, "supply": _safe(node.supply),
                     "subsidy": s.subsidy(node.chain.tip.height + 1)},
            "validators": {a: {"url": u, "stake": s.stake(a), "votes": a in voting} for a, u in s.validators.items()},
            "peers": [{"url": u, "alive": dead.get(u, 0) <= now} for u in node.peers.urls()],
            "mempool": mempool, "jobs": jobs, "devices": self.device_list(),
            "miners": net.get("miners"), "hardware": net.get("hardware"),
            "ops": _safe(lambda: ops(node, now)),
        }


# --- the server's health, and operations on it -------------------------------------------------------------------

def supervised(env=None) -> bool:
    """Whether something starts this node again when it ends: systemd sets INVOCATION_ID for its services
    (setup-node.sh's units); VIBEHASH_SUPERVISED=1 says so for any other supervisor."""
    env = os.environ if env is None else env
    return bool(env.get("INVOCATION_ID") or env.get("VIBEHASH_SUPERVISED") == "1")


def backup_dir(node) -> Path:
    """Where the backup service (setup-node.sh) keeps its request and report: $DATA/backup, $DATA being the folder
    that holds the node's --data-dir; $VIBEHASH_BACKUP_DIR elsewhere."""
    return Path(os.environ.get("VIBEHASH_BACKUP_DIR") or Path(node.store.dir).resolve().parent / "backup")


def _read_json(path: Path, limit: int = 65536) -> dict | None:
    try:
        if path.stat().st_size > limit:
            return None
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    total += e.stat().st_size if e.is_file(follow_symlinks=False) else 0
                except OSError:
                    pass
    except OSError:
        pass
    return total


def quorum_view(node) -> dict:
    """The validators, which of them signed a block lately (their votes in the last SIGNED_WINDOW commit
    certificates), and whether finality survives this node going away: the others online must hold more than two
    thirds of the stake (consensus.quorum)."""
    s = node.chain.state
    vals = s.validator_set()
    tip = node.chain.tip.height
    me = node.keys.address
    if not vals:
        return {"bft": False, "validators": 0, "online_pct": None, "safe_to_restart": False,
                "why": "this node seals the chain's blocks alone: none are made until it's back", "signers": {}}
    last: dict[str, int | None] = {a: None for a in vals}
    for height, cert in sorted(node.store.certificates(max(1, tip - SIGNED_WINDOW + 1), tip).items()):
        for v in (cert.get("votes") or []) if isinstance(cert, dict) else []:
            try:
                a = address_of(bytes.fromhex(v["pubkey"]))
            except (KeyError, TypeError, ValueError):
                continue
            if a in last:
                last[a] = height
    total = sum(vals.values())
    online = {a for a, h in last.items() if h is not None}
    power = sum(vals[a] for a in online)
    others = sum(vals[a] for a in online if a != me)
    if me not in vals:
        safe, why = True, "this node isn't a validator: blocks go on without it"
    elif quorum(others, total):
        safe, why = True, None
    else:
        safe = False
        why = (f"without this node, the validators that signed lately hold {100 * others / total:.0f}% of the stake; "
               "blocks stop until it's back (more than two thirds must sign)")
    return {"bft": True, "validators": len(vals), "online": len(online), "online_pct": round(100 * power / total, 1),
            "quorum_now": quorum(power, total), "safe_to_restart": safe, "why": why, "window": SIGNED_WINDOW,
            "signers": {a: {"stake": vals[a], "last_signed": last[a], "online": a in online,
                            "url": s.validators.get(a), "me": a == me} for a in vals}}


def ops(node, now: float | None = None) -> dict:
    """The server's health, for the admin app's Infrastructure panel."""
    now = time.time() if now is None else now
    data = Path(node.store.dir)
    try:
        du = shutil.disk_usage(data)
        disk = {"total_bytes": du.total, "used_bytes": du.used, "free_bytes": du.free,
                "free_pct": round(100 * du.free / du.total, 1) if du.total else None}
    except OSError:
        disk = None
    bdir = backup_dir(node)
    backup = _read_json(bdir / "last_backup.json")
    if backup is not None and isinstance(backup.get("timestamp"), (int, float)):
        backup["age_hours"] = round((now - backup["timestamp"]) / 3600, 2)
    request = _read_json(bdir / "request")
    started = getattr(node, "started_at", None)
    return {"version": __version__, "uptime_s": round(now - started) if started else None, "disk": disk,
            "blobs_bytes": _dir_bytes(node.blobs.root), "datasets_bytes": _dir_bytes(node.store.datasets.root),
            "backup": backup, "backup_requested": (request or {}).get("requested_at"),
            "backups_set_up": bdir.is_dir(), "health": _read_json(bdir.parent / "health.json"),
            "halted": bool(node.halted), "halted_why": node.halted_public, "supervised": supervised(),
            "quorum": quorum_view(node)}


def prune(node, stale_part_s: float) -> dict:
    """Delete the checkpoints and dispute evidence no recent round needs, as each round's end does (Node._end_round),
    now. Only between rounds: a round's own uploads aren't in its records yet, and pruning then would delete them."""
    before = _dir_bytes(node.blobs.root)
    with node.coord.lock:  # no round opens while this runs
        if node.coord.r is not None:
            raise Conflict("a round is open, and its checkpoints are in use: what it doesn't need is deleted when it "
                           "ends anyway. Try again between rounds")
        removed = node.blobs.prune(node._evidence())  # noqa: SLF001
    removed += node.store.datasets.prune_partial(older_than=stale_part_s)
    return {"ok": True, "removed": removed, "freed_bytes": max(0, before - _dir_bytes(node.blobs.root))}


def request_backup(node, now: float | None = None) -> dict:
    """Ask this server's backup service for a run: it watches $DATA/backup/request (a systemd path unit) and
    deletes the file when it starts. The node itself never runs anything as another user."""
    now = time.time() if now is None else now
    bdir = backup_dir(node)
    if not bdir.is_dir():
        raise Conflict(f"backups aren't set up on this server (no {bdir}): setup-node.sh installs the backup service")
    pending = (_read_json(bdir / "request") or {}).get("requested_at")
    if isinstance(pending, (int, float)) and now - pending < BACKUP_REQUEST_S:
        return {"ok": True, "requested_at": pending, "already": True}
    tmp = bdir / "request.tmp"
    tmp.write_text(json.dumps({"requested_at": int(now)}), encoding="utf-8")
    os.replace(tmp, bdir / "request")
    log.info("admin: a backup was asked for")
    return {"ok": True, "requested_at": int(now), "already": False}


def restart(node, force: bool = False) -> dict:
    """End the node so its supervisor starts it again (the `node` command's loop sees `restart_requested`). Refused
    (Conflict) when nothing would start it again, and, unless `force`, when finality wouldn't survive its absence."""
    if not supervised():
        raise Conflict("this node isn't run by systemd (or another supervisor): nothing would start it again")
    q = quorum_view(node)
    if not q["safe_to_restart"] and not force:
        raise Conflict(f"{q['why']}. Send force to restart it anyway")
    node.restart_requested = "an admin asked" + (" (forced)" if force and not q["safe_to_restart"] else "")
    log.warning("admin: restart asked for: %s", node.restart_requested)
    return {"ok": True, "restarting": True, "safe": q["safe_to_restart"]}


def _minutes(seconds: float) -> str:
    m = max(1, round(seconds / 60))
    return f"{m} minute{'s' if m != 1 else ''}"


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001  (a view: nothing in it may fail the whole answer)
        return None
