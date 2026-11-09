// vibehash.org/watch's alerts and counts: a Cloudflare Worker with a D1 database (schema.sql) and a cron every minute.
//
//   GET  /api/key                       the VAPID public key a page subscribes to push with
//   POST /api/subscribe   {subscription, addresses, payments}   alerts on for these addresses (at most MAX_WATCH)
//   POST /api/unsubscribe {endpoint}    alerts off
//   GET  /api/messages?sub=<id>         a phone's waiting alerts, taken (the service worker asks after each push)
//   POST /api/hello       {id, standalone}   counted: an install, or a day's open (no address, no IP stored)
//   GET  /api/stats                     the counts, with Authorization: Bearer STATS_TOKEN
//
// Every minute it reads the blocks since the last check from the public node and, for each watched address, queues:
// a slash (always), a missed deadline (at most hourly), "stopped working" when it had been working and has done
// nothing for STOPPED_S while others earn (once, and "back" when it returns), and, for phones that want them, its pay
// summed at most hourly. A phone with something new gets a push with no payload; its service worker then fetches the
// text from /api/messages. So the Worker never encrypts a payload, and the push services never see what it says.
//
// Secrets (npx wrangler secret put NAME): VAPID_PRIVATE (the JWK keys.mjs prints), STATS_TOKEN, SALT (any long random
// string, for hashing callers' addresses in rate limits). Vars (wrangler.jsonc): VAPID_PUBLIC, VAPID_SUBJECT, NODE.

const SYN = 100_000_000;
const MAX_WATCH = 5;                 // addresses one phone may watch
const MAX_SUBS = 20_000;             // phones in all
const WINDOW = 100;                  // the most blocks read in one check (the node's page)
const STOPPED_S = 30 * 60;           // no work for this long, while others work, is "stopped"
const PAY_EVERY_S = 3600;            // payment alerts, summed, at most this often
const MISSED_EVERY_S = 3600;         // missed-deadline alerts at most this often
const KEEP_MESSAGES = 20;            // per phone
const ADDRESS = /^vibe1[0-9a-f]{36}$/;
const ID = /^[0-9a-f-]{36}$/;
// Where a push endpoint may point: the browsers' own push services, never anywhere a caller chooses.
const PUSH_HOSTS = [/^web\.push\.apple\.com$/, /^fcm\.googleapis\.com$/, /^updates\.push\.services\.mozilla\.com$/,
                    /\.notify\.windows\.com$/, /^android\.googleapis\.com$/];
const LIMITS = { subscribe: [20, 3600], hello: [60, 3600], messages: [120, 600] };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors() });
    try {
      return await route(request, env, url);
    } catch (e) {
      return json({ error: e.message }, e.status ?? 500);
    }
  },
  async scheduled(event, env, ctx) {
    ctx.waitUntil(check(env));
  },
};

async function route(request, env, url) {
  const p = url.pathname, m = request.method;
  if (p === "/api/key" && m === "GET") return json({ key: env.VAPID_PUBLIC });
  if (p === "/api/subscribe" && m === "POST") return subscribe(env, await body(request), request);
  if (p === "/api/unsubscribe" && m === "POST") return unsubscribe(env, await body(request));
  if (p === "/api/messages" && m === "GET") return messages(env, url.searchParams.get("sub"), request);
  if (p === "/api/hello" && m === "POST") return hello(env, await body(request), request);
  if (p === "/api/stats" && m === "GET") return stats(env, request);
  if (p === "/api/test" && m === "POST") return testAlert(env, request);
  throw fail(404, "not found");
}

// The operators' check that alerts reach phones: "Alerts work" to every phone with alerts on (STATS_TOKEN).
async function testAlert(env, request) {
  const token = (request.headers.get("Authorization") ?? "").replace(/^Bearer /, "");
  if (!env.STATS_TOKEN || !(await same(token, env.STATS_TOKEN))) throw fail(401, "the stats token is missing or wrong");
  const subs = (await env.DB.prepare("SELECT id, endpoint FROM subs").all()).results;
  const now = Date.now();
  if (subs.length) {
    await env.DB.batch(subs.map((s) => env.DB.prepare("INSERT INTO messages (sub, at, title, body, tag) VALUES (?, ?, ?, ?, ?)")
      .bind(s.id, now, "Alerts work", "This is a test from Vibe Hash. Real alerts look like this.", "test")));
  }
  const results = [];
  for (const s of subs) {
    const status = await push(env, s.endpoint);
    if (status === 404 || status === 410) await unsubscribe(env, { endpoint: s.endpoint });
    results.push(status);
  }
  return json({ phones: subs.length, delivered: results.filter((r) => r >= 200 && r < 300).length, gone: results.filter((r) => r === 404 || r === 410).length });
}

// ---- alerts on and off ----

async function subscribe(env, { subscription, addresses, payments = true }, request) {
  await limit(env, request, "subscribe");
  const endpoint = String(subscription?.endpoint ?? "");
  let host;
  try {
    const u = new URL(endpoint);
    if (u.protocol !== "https:") throw new Error();
    host = u.hostname;
  } catch {
    throw fail(400, "that isn't a push subscription");
  }
  if (!PUSH_HOSTS.some((h) => h.test(host))) throw fail(400, "that push service isn't one this sends to");
  const list = [...new Set((Array.isArray(addresses) ? addresses : []).map((a) => String(a).toLowerCase()))];
  if (!list.length || list.length > MAX_WATCH || !list.every((a) => ADDRESS.test(a))) {
    throw fail(400, `send 1 to ${MAX_WATCH} wallet addresses`);
  }
  const id = await sha256(endpoint);
  const known = await env.DB.prepare("SELECT id FROM subs WHERE id = ?").bind(id).first();
  if (!known) {
    const n = (await env.DB.prepare("SELECT COUNT(*) AS n FROM subs").first()).n;
    if (n >= MAX_SUBS) throw fail(503, "alerts are full right now; try again later");
  }
  const now = Date.now();
  const steps = [
    env.DB.prepare("INSERT INTO subs (id, endpoint, payments, created) VALUES (?, ?, ?, ?) " +
                   "ON CONFLICT(id) DO UPDATE SET payments = excluded.payments")
      .bind(id, endpoint, payments ? 1 : 0, now),
    env.DB.prepare(`DELETE FROM watches WHERE sub = ? AND address NOT IN (${list.map(() => "?").join(",")})`).bind(id, ...list),
    ...list.map((a) => env.DB.prepare("INSERT OR IGNORE INTO watches (sub, address) VALUES (?, ?)").bind(id, a)),
  ];
  await env.DB.batch(steps);
  return json({ ok: true, id, addresses: list, payments: !!payments });
}

async function unsubscribe(env, { endpoint }) {
  const id = await sha256(String(endpoint ?? ""));
  await env.DB.batch([env.DB.prepare("DELETE FROM watches WHERE sub = ?").bind(id),
                      env.DB.prepare("DELETE FROM messages WHERE sub = ?").bind(id),
                      env.DB.prepare("DELETE FROM subs WHERE id = ?").bind(id)]);
  return json({ ok: true });
}

// The service worker's call after a push: what to show. Taken, so a second push doesn't show them again.
async function messages(env, sub, request) {
  await limit(env, request, "messages");
  if (!/^[0-9a-f]{64}$/.test(sub ?? "")) throw fail(400, "no such phone");
  const rows = (await env.DB.prepare("SELECT at, title, body, tag FROM messages WHERE sub = ? ORDER BY at").bind(sub).all()).results;
  await env.DB.prepare("DELETE FROM messages WHERE sub = ?").bind(sub).run();
  return json({ messages: rows });
}

// ---- counts ----

async function hello(env, { id, standalone }, request) {
  await limit(env, request, "hello");
  if (!ID.test(String(id ?? ""))) throw fail(400, "no id");
  const day = new Date().toISOString().slice(0, 10);
  const steps = [env.DB.prepare("INSERT OR IGNORE INTO hits (kind, id, day) VALUES (?, ?, ?)").bind(standalone ? "open" : "visit", id, day)];
  if (standalone) steps.push(env.DB.prepare("INSERT OR IGNORE INTO hits (kind, id, day) SELECT 'install', ?, ? " +
                                            "WHERE NOT EXISTS (SELECT 1 FROM hits WHERE kind = 'install' AND id = ?)").bind(id, day, id));
  await env.DB.batch(steps);
  return json({ ok: true });
}

async function stats(env, request) {
  const token = (request.headers.get("Authorization") ?? "").replace(/^Bearer /, "");
  if (!env.STATS_TOKEN || !(await same(token, env.STATS_TOKEN))) throw fail(401, "the stats token is missing or wrong");
  const day = (n) => new Date(Date.now() - n * 86400_000).toISOString().slice(0, 10);
  const one = async (sql, ...args) => (await env.DB.prepare(sql).bind(...args).first()).n;
  return json({
    installs: await one("SELECT COUNT(*) AS n FROM hits WHERE kind = 'install'"),
    installs_7d: await one("SELECT COUNT(*) AS n FROM hits WHERE kind = 'install' AND day >= ?", day(6)),
    home_screen_users_today: await one("SELECT COUNT(DISTINCT id) AS n FROM hits WHERE kind = 'open' AND day = ?", day(0)),
    home_screen_users_7d: await one("SELECT COUNT(DISTINCT id) AS n FROM hits WHERE kind = 'open' AND day >= ?", day(6)),
    browser_visitors_7d: await one("SELECT COUNT(DISTINCT id) AS n FROM hits WHERE kind = 'visit' AND day >= ?", day(6)),
    alert_phones: await one("SELECT COUNT(*) AS n FROM subs"),
    watched_addresses: await one("SELECT COUNT(DISTINCT address) AS n FROM watches"),
  });
}

// ---- the check, every minute ----

export async function check(env, now = Date.now() / 1000) {
  const node = env.NODE || "https://node.vibehash.org";
  const status = await getJSON(node + "/status");
  const tip = status.height;
  const cursor = Number((await env.DB.prepare("SELECT value FROM meta WHERE key = 'height'").first())?.value ?? NaN);
  if (!Number.isFinite(cursor) || tip - cursor > WINDOW * 10) {  // first run, or far behind: start from now, alert nothing old
    await setCursor(env, tip);
    return { checked: 0 };
  }
  if (tip <= cursor) return { checked: 0 };
  const from = Math.max(cursor + 1, tip - WINDOW + 1);
  const blocks = await getJSON(`${node}/blocks?from=${from}&to=${tip}`);
  const seen = new Map();  // address -> {paid, active, slashed: [reasons], missed, last}
  let othersWorked = false;
  for (const b of blocks) {
    const s = b.summary ?? {};
    if (Object.values(s.worker_flops ?? {}).some((f) => f > 0)) othersWorked = true;
    const touch = (a) => { if (!seen.has(a)) seen.set(a, { paid: 0, active: false, slashed: [], missed: false, last: 0 }); return seen.get(a); };
    for (const t of b.txs ?? []) if (t.kind === "reward") { const x = touch(t.recipient); x.paid += t.amount; x.active = true; x.last = b.timestamp; }
    for (const a of Object.keys(s.worker_flops ?? {})) if (s.worker_flops[a] > 0) { const x = touch(a); x.active = true; x.last = b.timestamp; }
    for (const a of Object.keys(s.verifier_flops ?? {})) if (s.verifier_flops[a] > 0) { const x = touch(a); x.active = true; x.last = b.timestamp; }
    for (const a of s.missed ?? []) touch(a).missed = true;
    for (const [a, , why] of s.slashes ?? []) touch(a).slashed.push(`block ${b.height}: ${why}`);
  }
  const rows = (await env.DB.prepare(
    "SELECT w.sub, w.address, w.last_active, w.stopped, w.paid_unsent, w.last_paid, w.last_missed, s.payments, s.endpoint " +
    "FROM watches w JOIN subs s ON s.id = w.sub").all()).results;
  const queued = new Map(), updates = [];
  const say = (sub, title, body, tag) => { if (!queued.has(sub)) queued.set(sub, []); queued.get(sub).push({ title, body, tag }); };
  for (const r of rows) {
    const x = seen.get(r.address), short = r.address.slice(0, 9) + "…" + r.address.slice(-4);
    let { last_active, stopped, paid_unsent, last_paid, last_missed } = r;
    if (x?.slashed.length) say(r.sub, "Slashed", `${short} was slashed: ${x.slashed[0]}`, "slash");
    if (x?.missed && (!last_missed || now - last_missed >= MISSED_EVERY_S)) { say(r.sub, "Missed a deadline", `${short} registered for a round but didn't finish in time.`, "missed"); last_missed = now; }
    if (x?.active) {
      if (stopped) { say(r.sub, "Back to work", `${short} is training again.`, "stopped"); stopped = 0; }
      last_active = Math.max(last_active ?? 0, x.last);
    } else if (last_active && !stopped && othersWorked && now - last_active > STOPPED_S) {
      say(r.sub, "Stopped working", `${short} hasn't done any work for ${Math.round((now - last_active) / 60)} minutes while others are earning. Is its computer on?`, "stopped");
      stopped = 1;
    }
    paid_unsent += x?.paid ?? 0;
    if (r.payments && paid_unsent > 0 && (!last_paid || now - last_paid >= PAY_EVERY_S)) {
      say(r.sub, `Paid ${vibe(paid_unsent)} VIBE`, `${short} earned ${vibe(paid_unsent)} VIBE${last_paid ? " since the last alert" : ""}.`, "paid");
      paid_unsent = 0; last_paid = now;
    }
    if (!r.payments) paid_unsent = 0;
    updates.push(env.DB.prepare("UPDATE watches SET last_active = ?, stopped = ?, paid_unsent = ?, last_paid = ?, last_missed = ? WHERE sub = ? AND address = ?")
      .bind(last_active ?? null, stopped, paid_unsent, last_paid, last_missed, r.sub, r.address));
  }
  for (const [sub, list] of queued) {
    for (const m of list) updates.push(env.DB.prepare("INSERT INTO messages (sub, at, title, body, tag) VALUES (?, ?, ?, ?, ?)").bind(sub, Math.round(now * 1000), m.title, m.body, m.tag));
    updates.push(env.DB.prepare(`DELETE FROM messages WHERE sub = ? AND rowid NOT IN (SELECT rowid FROM messages WHERE sub = ? ORDER BY at DESC LIMIT ${KEEP_MESSAGES})`).bind(sub, sub));
  }
  if (updates.length) await env.DB.batch(updates);
  await setCursor(env, tip);
  const endpoints = new Map(rows.map((r) => [r.sub, r.endpoint]));
  let sent = 0;
  for (const sub of queued.keys()) {
    const res = await push(env, endpoints.get(sub));
    if (res === 404 || res === 410) await unsubscribe(env, { endpoint: endpoints.get(sub) });  // the phone turned alerts off
    else if (res >= 200 && res < 300) sent++;
  }
  return { checked: blocks.length, alerts: [...queued.values()].reduce((n, l) => n + l.length, 0), sent };
}

async function setCursor(env, height) {
  await env.DB.prepare("INSERT INTO meta (key, value) VALUES ('height', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value")
    .bind(String(height)).run();
}

// ---- web push, with no payload: VAPID (RFC 8292) is all the push service needs ----

export async function push(env, endpoint) {
  const aud = new URL(endpoint).origin;
  const jwt = await vapidJWT(env, aud);
  const r = await fetch(endpoint, { method: "POST", headers: {
    Authorization: `vapid t=${jwt}, k=${env.VAPID_PUBLIC}`, TTL: "86400", Urgency: "normal", "Content-Length": "0",
  } });
  return r.status;
}

export async function vapidJWT(env, aud, now = Math.floor(Date.now() / 1000)) {
  const key = await crypto.subtle.importKey("jwk", JSON.parse(env.VAPID_PRIVATE), { name: "ECDSA", namedCurve: "P-256" }, false, ["sign"]);
  const enc = (o) => b64url(new TextEncoder().encode(JSON.stringify(o)));
  const head = enc({ typ: "JWT", alg: "ES256" }) + "." + enc({ aud, exp: now + 12 * 3600, sub: env.VAPID_SUBJECT || "mailto:hello@vibehash.org" });
  const sig = await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, key, new TextEncoder().encode(head));  // r || s, as JWS wants
  return head + "." + b64url(new Uint8Array(sig));
}

// ---- small things ----

async function limit(env, request, kind) {
  const [n, s] = LIMITS[kind];
  const ip = request.headers.get("CF-Connecting-IP") ?? "local";
  const key = kind + ":" + (await sha256((env.SALT ?? "") + ip)).slice(0, 32);
  const since = Date.now() - s * 1000;
  const count = (await env.DB.prepare("SELECT COUNT(*) AS n FROM limits WHERE key = ? AND at > ?").bind(key, since).first()).n;
  if (count >= n) throw fail(429, "too many requests; try again later");
  await env.DB.prepare("INSERT INTO limits (key, at) VALUES (?, ?)").bind(key, Date.now()).run();
  if (Math.random() < 0.02) await env.DB.prepare("DELETE FROM limits WHERE at < ?").bind(Date.now() - 86400_000).run();
}

async function getJSON(url) {
  const r = await fetch(url, { headers: { "User-Agent": "vibehash-watch-push" } });
  if (!r.ok) throw fail(502, `${url}: ${r.status}`);
  return r.json();
}

async function sha256(text) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function same(a, b) {
  return (await sha256("cmp:" + a)) === (await sha256("cmp:" + b));
}

function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function vibe(syn) {
  const v = syn / SYN;
  return v.toLocaleString("en-US", { maximumFractionDigits: v < 10 ? 4 : 2 });
}

async function body(request) {
  try { return await request.json(); } catch { throw fail(400, "send a JSON body"); }
}

function fail(status, message) {
  return Object.assign(new Error(message), { status });
}

function cors() {
  // No cookies and nothing secret in any answer, so any page may call it (the preview on a home network too).
  return { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST",
           "Access-Control-Allow-Headers": "Content-Type, Authorization", "Access-Control-Max-Age": "86400" };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...cors() } });
}
