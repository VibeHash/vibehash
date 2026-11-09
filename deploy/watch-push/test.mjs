// node test.mjs: worker.js against a real SQLite database (node:sqlite, Node 22.5+), a stand-in node and a stand-in
// push service. Exits 1 on the first failure.
import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import worker, { check, vapidJWT } from "./worker.js";

// ---- D1, over SQLite: prepare().bind().first()/all()/run(), and batch() in one transaction ----
function d1() {
  const db = new DatabaseSync(":memory:");
  db.exec(readFileSync(new URL("./schema.sql", import.meta.url), "utf8"));
  const stmt = (sql, args = []) => ({
    bind: (...a) => stmt(sql, a),
    first: async () => db.prepare(sql).get(...args) ?? null,
    all: async () => ({ results: db.prepare(sql).all(...args) }),
    run: async () => { db.prepare(sql).run(...args); return {}; },
    _run: () => db.prepare(sql).run(...args),
  });
  return { prepare: (sql) => stmt(sql), batch: async (list) => { db.exec("BEGIN"); try { list.forEach((s) => s._run()); db.exec("COMMIT"); } catch (e) { db.exec("ROLLBACK"); throw e; } }, raw: db };
}

// ---- the keys, the env ----
const pair = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign", "verify"]);
const PUBLIC = Buffer.from(await crypto.subtle.exportKey("raw", pair.publicKey)).toString("base64url");
const env = { DB: d1(), VAPID_PUBLIC: PUBLIC, VAPID_PRIVATE: JSON.stringify(await crypto.subtle.exportKey("jwk", pair.privateKey)),
              STATS_TOKEN: "stats-secret", SALT: "salt", NODE: "https://node.test" };

// ---- the stand-in node and push service ----
const A = "vibe1" + "a".repeat(36), B = "vibe1" + "b".repeat(36), C = "vibe1" + "c".repeat(36);
let chain = { tip: 100, blocks: {} }, pushes = [], gone = new Set();
function block(h, t, { paid = {}, work = [], check = [], missed = [], slashes = [] } = {}) {
  const worker_flops = Object.fromEntries(work.map((a) => [a, 10 ** 12]));
  const txs = Object.entries(paid).map(([a, v]) => ({ kind: "reward", sender: "system", recipient: a, amount: v * 1e8, memo: "work" }));
  chain.blocks[h] = { height: h, timestamp: t, txs, summary: { job_id: "j", worker_flops, verifier_flops: Object.fromEntries(check.map((a) => [a, 1])), missed, slashes } };
  chain.tip = Math.max(chain.tip, h);
}
globalThis.fetch = async (url, opts = {}) => {
  const u = new URL(url);
  if (u.host === "node.test") {
    if (u.pathname === "/status") return Response.json({ height: chain.tip, job_id: "j" });
    if (u.pathname === "/blocks") {
      const lo = +u.searchParams.get("from"), hi = +u.searchParams.get("to");
      return Response.json(Array.from({ length: hi - lo + 1 }, (_, i) => chain.blocks[lo + i] ?? { height: lo + i, timestamp: 0, txs: [], summary: {} }));
    }
  }
  if (u.host === "web.push.apple.com") {
    pushes.push({ url, headers: opts.headers });
    return new Response(null, { status: gone.has(url) ? 410 : 201 });
  }
  throw new Error("unexpected fetch " + url);
};
const call = async (method, path, body, headers = {}) => {
  const r = await worker.fetch(new Request("https://watch-api.test" + path, { method, headers: { "Content-Type": "application/json", "CF-Connecting-IP": headers.ip ?? "1.2.3.4", ...headers }, body: body ? JSON.stringify(body) : undefined }), env);
  return { status: r.status, body: await r.json().catch(() => null), cors: r.headers.get("Access-Control-Allow-Origin") };
};
const sub = (n) => ({ endpoint: `https://web.push.apple.com/phone-${n}`, keys: { p256dh: "x", auth: "y" } });
const take = async (endpoint) => {
  const id = Buffer.from(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(endpoint))).toString("hex");
  return (await call("GET", "/api/messages?sub=" + id)).body.messages.map((m) => m.title);
};
let passed = 0;
const it = async (name, fn) => { await fn(); passed++; console.log("ok  " + name); };

await it("the key and CORS", async () => {
  const r = await call("GET", "/api/key");
  assert.equal(r.body.key, PUBLIC); assert.equal(r.cors, "*");
});

await it("subscribe takes only push services and real addresses", async () => {
  assert.equal((await call("POST", "/api/subscribe", { subscription: { endpoint: "https://evil.example/x" }, addresses: [A] })).status, 400);
  assert.equal((await call("POST", "/api/subscribe", { subscription: { endpoint: "http://web.push.apple.com/x" }, addresses: [A] })).status, 400);
  assert.equal((await call("POST", "/api/subscribe", { subscription: sub(1), addresses: ["vibe1nope"] })).status, 400);
  assert.equal((await call("POST", "/api/subscribe", { subscription: sub(1), addresses: [A, B, C, A + "", "vibe1" + "d".repeat(36), "vibe1" + "e".repeat(36), "vibe1" + "f".repeat(36)] })).status, 400);
  const ok = await call("POST", "/api/subscribe", { subscription: sub(1), addresses: [A.toUpperCase(), B], payments: true });
  assert.equal(ok.status, 200); assert.deepEqual(ok.body.addresses, [A, B]);
  const again = await call("POST", "/api/subscribe", { subscription: sub(1), addresses: [A], payments: true });  // B dropped
  assert.deepEqual(again.body.addresses, [A]);
  assert.equal(env.DB.raw.prepare("SELECT COUNT(*) n FROM watches WHERE sub = ?").get(again.body.id).n, 1);
  await call("POST", "/api/subscribe", { subscription: sub(2), addresses: [A], payments: false });  // a second phone, problems only
});

await it("the first check starts from now and alerts nothing old", async () => {
  block(100, 1000, { paid: { [A]: 5 }, work: [A] });
  const r = await check(env, 1000);
  assert.equal(r.checked, 0); assert.equal(pushes.length, 0);
});

await it("a payment: one push to the phone that wants them, signed, carrying nothing", async () => {
  block(101, 1060, { paid: { [A]: 4.21 }, work: [A] });
  const r = await check(env, 1060);
  assert.equal(r.sent, 1); assert.equal(pushes.length, 1);
  const p = pushes[0];
  assert.equal(p.url, sub(1).endpoint); assert.equal(p.headers["Content-Length"], "0");
  const [, t, k] = p.headers.Authorization.match(/^vapid t=([^,]+), k=(.+)$/);
  assert.equal(k, PUBLIC);
  const [h, c, s] = t.split(".");
  const claims = JSON.parse(Buffer.from(c, "base64url"));
  assert.equal(claims.aud, "https://web.push.apple.com"); assert.ok(claims.exp > Date.now() / 1000);
  assert.ok(await crypto.subtle.verify({ name: "ECDSA", hash: "SHA-256" }, pair.publicKey, Buffer.from(s, "base64url"), new TextEncoder().encode(h + "." + c)), "the VAPID signature checks");
  assert.deepEqual(await take(sub(1).endpoint), ["Paid 4.21 VIBE"]);
  assert.deepEqual(await take(sub(1).endpoint), [], "taken: not shown twice");
  assert.deepEqual(await take(sub(2).endpoint), [], "the problems-only phone got nothing");
});

await it("payments are summed, at most hourly", async () => {
  block(102, 1120, { paid: { [A]: 1 }, work: [A] });
  await check(env, 1120);
  assert.deepEqual(await take(sub(1).endpoint), []);
  block(103, 1060 + 3700, { paid: { [A]: 2 }, work: [A] });
  await check(env, 1060 + 3700);
  assert.deepEqual(await take(sub(1).endpoint), ["Paid 3 VIBE"]);
});

await it("a slash and a missed deadline reach both phones; missed at most hourly", async () => {
  const t = 1060 + 3800;
  block(104, t, { slashes: [[A, B, "a segment didn't match"]], missed: [A] });
  await check(env, t);
  assert.deepEqual(await take(sub(1).endpoint), ["Slashed", "Missed a deadline"]);
  assert.deepEqual(await take(sub(2).endpoint), ["Slashed", "Missed a deadline"]);
  block(105, t + 60, { missed: [A] });
  await check(env, t + 60);
  assert.deepEqual(await take(sub(1).endpoint), []);
});

await it("stopped working while others earn, once, then back", async () => {
  let t = 1060 + 3700;  // A last worked here
  for (let i = 0; i < 4; i++) { t += 600; block(106 + i, t, { paid: { [B]: 1 }, work: [B] }); await check(env, t); }  // 40 minutes of B only
  assert.deepEqual(await take(sub(2).endpoint), ["Stopped working"]);
  t += 600; block(110, t, { paid: { [B]: 1 }, work: [B] }); await check(env, t);
  assert.deepEqual(await take(sub(2).endpoint), [], "said once");
  t += 60; block(111, t, { work: [A] }); await check(env, t);
  assert.deepEqual(await take(sub(2).endpoint), ["Back to work"]);
});

await it("a phone whose push service says it's gone is forgotten", async () => {
  gone.add(sub(2).endpoint);
  block(112, 99999, { slashes: [[A, B, "again"]] });
  await check(env, 99999);
  assert.equal(env.DB.raw.prepare("SELECT COUNT(*) n FROM subs").get().n, 1);
  assert.equal(env.DB.raw.prepare("SELECT COUNT(*) n FROM watches").get().n, 1);
});

await it("installs and opens are counted once each, and the counts need the token", async () => {
  const id = "11111111-2222-3333-4444-555555555555", id2 = "66666666-2222-3333-4444-555555555555";
  assert.equal((await call("POST", "/api/hello", { id: "nope", standalone: true })).status, 400);
  await call("POST", "/api/hello", { id, standalone: true });
  await call("POST", "/api/hello", { id, standalone: true });
  await call("POST", "/api/hello", { id: id2, standalone: false });
  assert.equal((await call("GET", "/api/stats")).status, 401);
  assert.equal((await call("GET", "/api/stats", null, { Authorization: "Bearer wrong" })).status, 401);
  const s = (await call("GET", "/api/stats", null, { Authorization: "Bearer stats-secret" })).body;
  assert.deepEqual([s.installs, s.home_screen_users_today, s.browser_visitors_7d, s.alert_phones, s.watched_addresses], [1, 1, 1, 1, 1]);
});

await it("subscribing is rate-limited per caller", async () => {
  let last;
  for (let i = 0; i < 21; i++) last = await call("POST", "/api/subscribe", { subscription: sub(100 + i), addresses: [C] }, { ip: "9.9.9.9" });
  assert.equal(last.status, 429);
  assert.equal((await call("POST", "/api/subscribe", { subscription: sub(200), addresses: [C] }, { ip: "8.8.8.8" })).status, 200, "another caller isn't");
});

await it("a nickname goes into its alerts, cleaned and cut to 24 characters", async () => {
  const D = "vibe1" + "d".repeat(36);
  const r = await call("POST", "/api/subscribe", { subscription: sub(300), addresses: [D], names: { [D]: "  Gaming\nPC   upstairs, the one with the long name  " } }, { ip: "7.7.7.7" });
  assert.equal(r.status, 200);
  const name = env.DB.raw.prepare("SELECT name FROM watches WHERE address = ?").get(D).name;
  assert.equal(name, "Gaming PC upstairs, the");
  const t = 200000;
  block(chain.tip + 1, t - 60, { work: [B] }); await check(env, t - 60);
  block(chain.tip + 1, t, { slashes: [[D, B, "a segment didn't match"]] }); await check(env, t);
  const id = Buffer.from(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(sub(300).endpoint))).toString("hex");
  const msgs = (await call("GET", "/api/messages?sub=" + id, null, { ip: "7.7.7.7" })).body.messages;
  assert.equal(msgs[0].title, "Slashed"); assert.ok(msgs[0].body.startsWith("Gaming PC upstairs, the was slashed"), msgs[0].body);
  await call("POST", "/api/subscribe", { subscription: sub(300), addresses: [D], names: {} }, { ip: "7.7.7.7" });  // the name taken away
  assert.equal(env.DB.raw.prepare("SELECT name FROM watches WHERE address = ?").get(D).name, null);
});

await it("a test alert reaches every phone, only with the token", async () => {
  assert.equal((await call("POST", "/api/test", {})).status, 401);
  await take(sub(1).endpoint);  // what it hadn't collected from the steps above (stopped, back, a slash)
  const before = pushes.length;
  const r = await call("POST", "/api/test", {}, { Authorization: "Bearer stats-secret" });
  const phones = env.DB.raw.prepare("SELECT COUNT(*) n FROM subs").get().n;
  assert.equal(r.body.phones, phones); assert.equal(r.body.delivered, phones); assert.equal(pushes.length - before, phones);
  assert.deepEqual(await take(sub(1).endpoint), ["Alerts work"]);
});

await it("unsubscribe forgets the phone", async () => {
  await call("POST", "/api/unsubscribe", { endpoint: sub(1).endpoint });
  assert.equal(env.DB.raw.prepare("SELECT COUNT(*) n FROM subs WHERE endpoint = ?").get(sub(1).endpoint).n, 0);
});

console.log(`\n${passed} passed`);
