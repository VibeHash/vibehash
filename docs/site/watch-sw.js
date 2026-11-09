// vibehash.org/watch's service worker (registered with scope /watch/): it shows the alerts.
// A push carries nothing; this asks the alerts Worker for what to show (deploy/watch-push). iOS expects every push to
// show a notification, so one always shows, even if the Worker can't be reached.
"use strict";
var ALERTS = "https://watch-api.vibehash.org";

self.addEventListener("install", function () { self.skipWaiting(); });
self.addEventListener("activate", function (e) { e.waitUntil(self.clients.claim()); });

function hex(buf) {
  return Array.prototype.map.call(new Uint8Array(buf), function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
}

self.addEventListener("push", function (event) {
  event.waitUntil(
    self.registration.pushManager.getSubscription()
      .then(function (sub) {
        if (!sub) return [];
        return crypto.subtle.digest("SHA-256", new TextEncoder().encode(sub.endpoint))
          .then(function (d) { return fetch(ALERTS + "/api/messages?sub=" + hex(d), { cache: "no-store" }); })
          .then(function (r) { return r.ok ? r.json() : { messages: [] }; })
          .then(function (j) { return j.messages || []; });
      })
      .catch(function () { return []; })
      .then(function (list) {
        if (!list.length) list = [{ title: "Vibe Hash", body: "Something changed with a wallet you watch. Open the app to see it.", tag: "watch" }];
        // Several at once (a slash and a missed deadline in the same minute): each its own notification, newest last.
        return Promise.all(list.slice(-4).map(function (m) {
          return self.registration.showNotification(m.title, {
            body: m.body, tag: m.tag || undefined, icon: "/watch-180.png", badge: "/watch-180.png", data: { url: "/watch/" },
          });
        }));
      })
  );
});

self.addEventListener("notificationclick", function (event) {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then(function (wins) {
      for (var i = 0; i < wins.length; i++) {
        if (wins[i].url.indexOf("/watch/") >= 0 && "focus" in wins[i]) return wins[i].focus();
      }
      return self.clients.openWindow("/watch/");
    })
  );
});
