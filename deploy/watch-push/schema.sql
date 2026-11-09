-- vibehash.org/watch's alerts and counts (worker.js). Apply with:
--   npx wrangler d1 execute vibehash-watch --remote --file schema.sql
-- A database made before nicknames needs, once:
--   npx wrangler d1 execute vibehash-watch --remote --command "ALTER TABLE watches ADD COLUMN name TEXT"

-- A phone that turned alerts on: its push endpoint (from the browser's push service) and whether it wants payments too.
-- id is the hex SHA-256 of the endpoint: the service worker sends it to collect its messages.
CREATE TABLE IF NOT EXISTS subs (
  id        TEXT PRIMARY KEY,
  endpoint  TEXT NOT NULL,
  payments  INTEGER NOT NULL DEFAULT 1,
  created   INTEGER NOT NULL,
  last_push INTEGER
);

-- The addresses each phone watches, and where each stands, so an alert goes out once and not every minute.
CREATE TABLE IF NOT EXISTS watches (
  sub         TEXT NOT NULL,
  address     TEXT NOT NULL,
  name        TEXT,               -- the nickname the phone gave it ("Gaming PC"), for its alerts; NULL: the address
  last_active INTEGER,            -- the last block time it trained, checked or was paid (seconds)
  stopped     INTEGER NOT NULL DEFAULT 0,  -- "stopped working" sent, and not yet back
  paid_unsent INTEGER NOT NULL DEFAULT 0,  -- syn paid since the last payment alert
  last_paid   INTEGER NOT NULL DEFAULT 0,  -- when the last payment alert went (seconds)
  last_missed INTEGER NOT NULL DEFAULT 0,  -- when the last missed-deadline alert went (seconds)
  PRIMARY KEY (sub, address)
);
CREATE INDEX IF NOT EXISTS watches_by_address ON watches (address);

-- Alerts waiting for a phone to collect them (the push itself carries nothing).
CREATE TABLE IF NOT EXISTS messages (
  sub   TEXT NOT NULL,
  at    INTEGER NOT NULL,
  title TEXT NOT NULL,
  body  TEXT NOT NULL,
  tag   TEXT
);
CREATE INDEX IF NOT EXISTS messages_by_sub ON messages (sub, at);

-- How many people use it: a random id the page makes once, never an address or an IP.
--   kind 'install': its first launch from the home screen; 'open': a day it was opened from the home screen;
--   'visit': a day it was opened in the browser.
CREATE TABLE IF NOT EXISTS hits (
  kind TEXT NOT NULL,
  id   TEXT NOT NULL,
  day  TEXT NOT NULL,
  PRIMARY KEY (kind, id, day)
);

-- Rate limits: a salted hash of the caller's address, never the address itself.
CREATE TABLE IF NOT EXISTS limits (
  key TEXT NOT NULL,
  at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS limits_by_key ON limits (key, at);

-- The last block height checked.
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
