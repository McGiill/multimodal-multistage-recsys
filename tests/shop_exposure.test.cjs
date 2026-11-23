// node --test tests/shop_exposure.test.cjs
// 실제 shop.js를 VM에 로드한다. 브라우저 조작 없이 DOM/IntersectionObserver/네트워크 ACK만 모의한다.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../src/serving/shop/shop.js"), "utf8");
const pause = () => new Promise(resolve => setImmediate(resolve));

function setup() {
  const requests = [], observers = [], handlers = {}, navigations = [], toasts = [];
  let nonce = 0;
  class Observer {
    constructor(callback) { this.callback = callback; this.observed = []; observers.push(this); }
    observe(card) { this.observed.push(card); }
    unobserve(card) { this.observed = this.observed.filter(x => x !== card); }
    disconnect() { this.observed = []; }
    emit(entries) { this.callback(entries); }
  }
  const saved = { userId: "U1", sessionId: "S1", member: "U1", hidden: [], recent: [], views: 0, audience: "all" };
  const context = vm.createContext({
    console, Date, URLSearchParams, Uint8Array, setTimeout, clearTimeout, navigations, toasts,
    crypto: { getRandomValues(bytes) { bytes.fill(++nonce); return bytes; } },
    sessionStorage: { getItem() { return JSON.stringify(saved); }, setItem() {} },
    window: { IntersectionObserver: Observer, innerWidth: 1200, innerHeight: 800 },
    IntersectionObserver: Observer,
    document: { hidden: false, visibilityState: "visible", documentElement: { clientWidth: 1200, clientHeight: 800 },
      getElementById() { return {}; }, addEventListener(name, handler) { handlers[name] = handler; } },
    getComputedStyle(element) { return { display: element.hidden ? "none" : "block", visibility: "visible", opacity: "1" }; },
    fetch(url, options) {
      let resolve, reject;
      const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
      requests.push({ url, payload: JSON.parse(options.body),
        resolve: () => resolve({ ok: true, json: async () => ({ status: "accepted" }) }), reject });
      return promise;
    },
  });
  vm.runInContext(source.slice(0, source.indexOf("// ------------------------------------------------------------------ 이벤트")), context);
  vm.runInContext("navigate = value => navigations.push(value); toast = value => toasts.push(value); renderFooter = () => {};", context);
  const click = source.slice(source.indexOf('document.addEventListener("click"'),
    source.indexOf('document.getElementById("search-form")'));
  vm.runInContext(click, context);
  const call = (code, additions = {}) => { Object.assign(context, additions); return vm.runInContext(code, context); };
  function card(pid, flags = {}) {
    const classes = new Set(flags.leaving ? ["leaving"] : []);
    const card = { isConnected: true, hidden: false, dataset: { productId: pid }, ...flags,
      classList: { contains: name => classes.has(name), add: name => classes.add(name) },
      closest: selector => selector === ".card" ? card : null,
      getClientRects: () => card.hidden ? [] : [card.getBoundingClientRect()],
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 }),
    };
    const link = { dataset: { view: pid, source: "home" }, isConnected: true,
      closest: selector => selector === ".card" ? card : selector === "[data-view]" ? link : null };
    card.link = link;
    return card;
  }
  function track(cards, opts = {}) {
    const container = { isConnected: true, querySelectorAll: () => cards };
    call("trackImpressions(container, options)", { container, options: opts });
    return container;
  }
  const entry = (card, ratio = 1, intersecting = true) => ({ target: card, intersectionRatio: ratio, isIntersecting: intersecting });
  return { requests, observers, handlers, call, card, track, entry };
}

test("only intersecting displayed cards count, in DOM order, once", async () => {
  const x = setup(), a = x.card("A"), b = x.card("B"), c = x.card("C", { isConnected: false });
  x.track([a, b, c], { recommendationId: "R1" });
  assert.equal(x.requests.length, 0);
  x.observers[0].emit([x.entry(a, 0, false), x.entry(b, .49), x.entry(c)]);
  assert.equal(x.requests.length, 0);
  x.observers[0].emit([x.entry(b), x.entry(a), x.entry(c)]);
  assert.deepEqual(x.requests[0].payload.product_ids, ["A", "B"]);
  assert.equal(x.requests[0].payload.recommendation_id, "R1");
  x.requests[0].resolve(); await pause();
  x.observers[0].emit([x.entry(a), x.entry(b)]);
  assert.equal(x.requests.length, 1);
});

test("removed page or hidden/leaving card never counts a delayed observer callback", async () => {
  const x = setup(), hidden = x.card("H", { hidden: true }), leaving = x.card("L", { leaving: true });
  const detached = x.card("D"), rejected = x.card("R");
  x.track([hidden, leaving, detached, rejected]);
  detached.isConnected = false;
  x.call('state.hidden.push("R")');
  x.observers[0].emit([hidden, leaving, detached, rejected].map(c => x.entry(c)));
  assert.equal(x.requests.length, 0);
});

test("click waits for exposure acknowledgement and reuses the recorded impression", async () => {
  const x = setup(), card = x.card("A"); x.track([card]);
  let completed = false;
  const clicked = x.call("exposureForClick(link)", { link: card.link }).then(id => { completed = true; return id; });
  assert.equal(x.requests.length, 1); await pause();
  assert.equal(completed, false);
  const id = x.requests[0].payload.impression_id;
  x.requests[0].resolve();
  assert.equal(await clicked, id);
  assert.equal(await x.call("exposureForClick(link)", { link: card.link }), id);
  assert.equal(x.requests.length, 1);
});

test("retry after a lost acknowledgement uses the same idempotency ID", async () => {
  const x = setup(), card = x.card("A"); x.track([card]);
  x.observers[0].emit([x.entry(card)]);
  const id = x.requests[0].payload.impression_id;
  x.requests[0].reject(new Error("Response lost after server committed")); await pause();
  const clicked = x.call("exposureForClick(link)", { link: card.link });
  assert.equal(x.requests.length, 2);
  assert.equal(x.requests[1].payload.impression_id, id);
  x.requests[1].resolve();
  assert.equal(await clicked, id);
});

test("member switch while exposure is pending never records click as the new member", async () => {
  const x = setup(), card = x.card("A"); x.track([card]);
  const clicked = x.handlers.click({ target: card.link, preventDefault() {} });
  assert.equal(x.requests[0].payload.user_id, "U1");
  x.call('state.userId = "U2"; state.sessionId = "S2";');
  x.requests[0].resolve(); await pause();
  const clickEvents = x.requests.filter(r => r.url === "/api/events");
  for (const request of clickEvents) {
    assert.equal(request.payload.user_id, "U1");
    assert.equal(request.payload.session_id, "S1");
    request.resolve();
  }
  await clicked;
  assert.equal(x.call("state.views"), 0);
});
