// The push key pair (VAPID) the alerts need, made once:   node keys.mjs
// It prints the public key, for wrangler.jsonc's VAPID_PUBLIC, and the private key, for
// `npx wrangler secret put VAPID_PRIVATE`. Keep the private one only there and in your password manager: anyone with
// it can send pushes in vibehash.org's name to everyone who turned alerts on. A new pair turns every phone's alerts off
// (they subscribed to the old public key), so make it once.
const pair = await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign", "verify"]);
const raw = new Uint8Array(await crypto.subtle.exportKey("raw", pair.publicKey));
const jwk = await crypto.subtle.exportKey("jwk", pair.privateKey);
const b64url = (b) => Buffer.from(b).toString("base64url");
console.log("VAPID_PUBLIC (wrangler.jsonc vars):\n" + b64url(raw) + "\n");
console.log("VAPID_PRIVATE (npx wrangler secret put VAPID_PRIVATE, then paste this one line):\n" + JSON.stringify(jwk));
