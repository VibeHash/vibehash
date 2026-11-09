# vibehash.org/watch's alerts

A Cloudflare Worker (`worker.js`) with a D1 database (`schema.sql`) and a cron every minute. Phones that turned on alerts in vibehash.org/watch get a push when a watched wallet is slashed, misses a deadline, stops working while others earn (and when it's back), and, if they want, a summary of what it was paid at most once an hour. It also counts how many people add the app to their home screen. See the comment at the top of `worker.js` for the routes.

The push carries no data: the phone's service worker (`docs/site/watch-sw.js`) fetches the text from the Worker. Nothing is stored about a person beyond the push endpoint their phone gave and the wallet addresses they chose; the counts use a random id the page makes, and rate limits a salted hash of the caller's address.

## Deploy, once (about 10 minutes)

From this folder, logged in to the Cloudflare account that has vibehash.org (`npx wrangler login`):

1. The database:
   ```
   npx wrangler d1 create vibehash-watch
   ```
   Put the `database_id` it prints into `wrangler.jsonc`, then:
   ```
   npx wrangler d1 execute vibehash-watch --remote --file schema.sql
   ```
2. The push keys:
   ```
   node keys.mjs
   ```
   Put the public key into `wrangler.jsonc` (`VAPID_PUBLIC`). Then `npx wrangler secret put VAPID_PRIVATE` and paste the private line. Save both in the password manager. Make them only once: new keys turn off every phone's alerts.
3. The other secrets:
   ```
   npx wrangler secret put STATS_TOKEN     # any long random string: it opens the counts
   npx wrangler secret put SALT            # any long random string
   ```
4. Deploy:
   ```
   npx wrangler deploy
   ```
   This also sets up `watch-api.vibehash.org` (the page's `ALERTS` address) on the vibehash.org zone.

## The counts

```
curl -H "Authorization: Bearer <STATS_TOKEN>" https://watch-api.vibehash.org/api/stats
```

- `installs`: phones that opened it from the home screen at least once (iPhones don't say when an app is added, so the first home-screen launch is the install)
- `home_screen_users_today` / `_7d`: of those, how many opened it that day or week
- `browser_visitors_7d`: people who used it in a browser tab
- `alert_phones`, `watched_addresses`: alerts turned on

## Check alerts reach phones

With alerts turned on in the app (from the Home Screen), send every such phone a test:

```
curl -X POST -H "Authorization: Bearer <STATS_TOKEN>" https://watch-api.vibehash.org/api/test
```

It answers how many phones it tried and how many the push services took; the phone shows "Alerts work".

## Test

```
node test.mjs
```

It runs the Worker against a real SQLite database (Node 22's `node:sqlite`), a stand-in node and a stand-in push service, and checks every alert, the counts, the limits and the VAPID signature.
