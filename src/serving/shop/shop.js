/* 통합 서비스(쇼핑몰) 화면. API 서버가 /shop/ 으로 제공한다 (src/serving/main.py).
 *
 * 흐름
 *   회원 선택 → (신규 회원) 취향 조사: 마음에 드는 상품 5개 이상 → 홈 추천
 *   상품 클릭 = 관심 (POST /api/events view) → 다음 추천에 바로 반영 (세션 GRU + Multi-Stage)
 *   검색: 문장 → /api/search, 사진(착용·상품) → /api/search/style
 *   상품 상세: 비슷한 상품(사진 유사도) + 회원 맞춤 추천
 *   홈: 최근 본 상품과 비슷한 상품(사진 유사도, 클릭하면 바로 바뀜) + 회원 맞춤 추천(Multi-Stage)
 * 한 화면 안에서 검색 결과와 추천이 겹치지 않게, 검색·비슷한 상품에 나온 상품은 추천 칸에서 뺀다 (우선순위: 검색 > 추천).
 * 취향 조사는 지금 상품 사진으로 한다. 코디(착용) 사진이 준비되면 SURVEY만 바꾸면 된다.
 */
"use strict";

const CONFIG = {
  demoAt: "2026-04-01T00:00:00+00:00", // 추천 기준 시각 = 로그 데이터가 끝나는 시점 (대시보드 추천 데모와 같음)
  minPicks: 5,                          // = configs/recommend.yaml reranking.cold_start_min_events
  recTopN: 10,
  searchTopK: 20,
  photoTopK: 8,
  similarTopK: 8,
  recentBasis: 3,                       // 홈의 "최근 본 상품과 비슷한 상품": 최근 본 상품 몇 개 기준
  members: ["U00005", "U00020", "U00036", "U00002"], // 시뮬레이터 고객 (조회 이력 있음)
};

const AUDIENCE = { all: "전체", women: "여성", men: "남성", kids: "아동" };
const REASON = { ctcvr: "맞춤 추천", ctr: "맞춤 추천", cvr: "맞춤 추천", mab_exploration: "새로운 발견", new_item: "신상품", popular: "인기", trending: "급상승" };
const COLD_REASONS = new Set(["popular", "trending"]);
const SEARCH_TYPE = { text: "텍스트 검색", image: "이미지 검색", hybrid: "사진 + 문장 검색" };
const BASIS = {
  photo_item: "사진 속 옷과 비슷한 상품", photo_style: "사진 분위기와 어울리는 상품",
  whole_photo: "사진 전체와 비슷한 상품", whole_photo_text: "사진 + 문장으로 찾은 상품",
};
const BOX_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0a0", "#f032e6", "#808000"];

// 취향 조사: 스타일마다 검색어 2개 → 검색 결과 상위 2개씩 (고객층 안에서). 검색은 영어 기준.
const SURVEY = {
  women: [
    { style: "데일리 캐주얼", queries: ["basic cotton t-shirt", "blue denim jeans"] },
    { style: "스트리트", queries: ["oversized hoodie", "cargo trousers"] },
    { style: "미니멀·시크", queries: ["black blazer", "white shirt"] },
    { style: "클래식·프레피", queries: ["knitted cardigan", "pleated skirt"] },
    { style: "로맨틱·페미닌", queries: ["floral dress", "lace blouse"] },
    { style: "워크웨어·빈티지", queries: ["denim jacket", "corduroy trousers"] },
    { style: "스포티·아웃도어", queries: ["sports tights", "padded jacket"] },
  ],
  men: [
    { style: "데일리 캐주얼", queries: ["basic cotton t-shirt", "slim jeans"] },
    { style: "스트리트", queries: ["oversized hoodie", "cargo trousers"] },
    { style: "미니멀·시크", queries: ["black blazer", "white shirt"] },
    { style: "클래식·프레피", queries: ["knitted cardigan", "chino trousers"] },
    { style: "워크웨어·빈티지", queries: ["denim jacket", "corduroy shirt"] },
    { style: "스포티·아웃도어", queries: ["running shorts", "padded jacket"] },
  ],
  kids: [
    { style: "데일리", queries: ["printed t-shirt", "joggers"] },
    { style: "나들이", queries: ["dress", "denim jacket"] },
    { style: "활동복", queries: ["sports top", "padded jacket"] },
    { style: "홈웨어", queries: ["pyjama set", "bodysuit"] },
  ],
};

// ------------------------------------------------------------------ 상태 (방문 단위, 새로고침해도 유지)
const STORE_KEY = "shop-state-v1";
let state = loadState() || freshState("new");
let photo = null; // 사진 검색에 쓴 사진 {dataUrl, base64, width, height, query} (메모리에만)

function freshState(member) {
  const isNew = member === "new";
  return {
    member,
    userId: isNew ? `NEW-${randomHex(6)}` : member, // 데이터에 없는 ID = 이력 없는 신규 회원
    sessionId: `web-${randomHex(12)}`,
    isNew,
    onboarded: !isNew,
    audience: isNew ? null : "all",
    views: 0,     // 이번 방문 클릭 수 (취향 조사 선택 포함)
    recent: [],   // 이번 방문에서 본 상품 (오래된 것 → 최근)
    hidden: [],   // "관심 없음"으로 뺀 상품
  };
}
function loadState() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORE_KEY));
    return saved && Array.isArray(saved.recent) && Array.isArray(saved.hidden) ? saved : null;
  } catch {
    return null;
  }
}
function saveState() {
  try { sessionStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch { /* 저장소를 못 쓰면 이번 화면에서만 유지 */ }
}
function randomHex(n) {
  const bytes = new Uint8Array(Math.ceil(n / 2));
  crypto.getRandomValues(bytes);
  return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("").slice(0, n);
}

// ------------------------------------------------------------------ API
async function api(path, init) {
  let r;
  try {
    r = await fetch(path, init);
  } catch {
    throw new Error("API 서버에 연결할 수 없어요.");
  }
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* 본문이 JSON이 아님 */ }
    const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    err.status = r.status;
    throw err;
  }
  return r.json();
}
const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const infoCache = new Map();
function productInfo(pid) {
  if (!infoCache.has(pid)) infoCache.set(pid, api(`/api/products/${encodeURIComponent(pid)}`).catch(() => null));
  return infoCache.get(pid);
}
const imageUrl = pid => `/api/products/${encodeURIComponent(pid)}/image`;

/** 추천. exclude(같은 화면의 검색 결과 등)와 "관심 없음" 상품은 빼고 recTopN개를 채운다. */
async function recommend(exclude = new Set()) {
  const skip = new Set([...exclude, ...state.hidden]);
  const params = new URLSearchParams({
    user_id: state.userId, session_id: state.sessionId, at: CONFIG.demoAt,
    top_n: String(Math.min(100, CONFIG.recTopN + skip.size)),
  });
  const res = await api(`/api/recommend?${params}`);
  res.shown = res.recommendations.filter(r => !skip.has(r.product_id)).slice(0, CONFIG.recTopN);
  res.isCold = res.recommendations.some(r => COLD_REASONS.has(r.reason));
  return res;
}

/** 상품 클릭 = 관심. 세션 클릭으로 쌓여서 다음 추천에 반영된다 (MAB 탐색 칸이면 클릭 기록). */
async function logView(pid, source) {
  await post("/api/events", {
    event_id: `web-${randomHex(16)}`, user_id: state.userId, session_id: state.sessionId,
    event_type: "view", product_id: pid, timestamp: new Date().toISOString(), source,
  });
  state.views += 1;
  state.recent = [...state.recent.filter(p => p !== pid), pid].slice(-20);
  saveState();
  renderFooter();
}

const styleAudience = () => (state.audience && state.audience !== "all" ? state.audience : "auto");
const searchAudience = () => state.audience || "all";

// ------------------------------------------------------------------ 화면 조각
const $app = document.getElementById("app");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const price = p => (p == null ? "" : `$${Number(p).toFixed(2)}`);

function newPage(html) {
  const page = document.createElement("div");
  page.innerHTML = html;
  $app.replaceChildren(page);
  window.scrollTo(0, 0);
  return page;
}
const slot = (page, name) => page.querySelector(`[data-slot="${name}"]`);

function block(title, name, { caption = "", strip = false } = {}) {
  return `<section class="block">
    <div class="block-head"><div><h2>${esc(title)}</h2>
      <p class="caption" data-slot="${name}-caption">${esc(caption)}</p></div></div>
    <div class="${strip ? "strip" : "grid"}" data-slot="${name}">${skeletons(strip ? 6 : 5)}</div>
  </section>`;
}
function skeletons(n) {
  return Array.from({ length: n }, () => `<div class="card skeleton"><div class="card-media"></div><div class="bar"></div></div>`).join("");
}

function card(p, { source, reason = null, hideable = false } = {}) {
  const pid = p.product_id;
  const meta = [p.category, p.color].filter(Boolean).join(" · ");
  const open = `href="#/product/${encodeURIComponent(pid)}" data-view="${esc(pid)}" data-source="${esc(source)}"`;
  return `<article class="card">
    <a class="card-media" ${open} tabindex="-1" aria-hidden="true">
      <img src="${imageUrl(pid)}" alt="" loading="lazy">
      ${reason ? `<span class="badge badge-${esc(reason)}">${esc(REASON[reason] || reason)}</span>` : ""}
    </a>
    <div class="card-body">
      <a class="card-name" ${open}>${esc(p.name || pid)}</a>
      ${meta ? `<p class="card-meta">${esc(meta)}</p>` : ""}
      ${p.price != null ? `<p class="card-price">${price(p.price)}</p>` : ""}
    </div>
    ${hideable ? `<button class="card-hide" type="button" data-hide="${esc(pid)}" title="관심 없음" aria-label="관심 없음">×</button>` : ""}
  </article>`;
}

async function withInfo(items) {
  const infos = await Promise.all(items.map(i => productInfo(i.product_id)));
  return items.map((item, k) => ({ ...(infos[k] || {}), ...item }));
}

function showError(el, err) {
  if (!el) return;
  const hint = err.status === 503 ? " 추천 모델을 먼저 학습해 주세요 (make docker-train)." : "";
  el.innerHTML = `<p class="error">${esc(err.message)}${esc(hint)}</p>`;
}

function latencyText(res) {
  const l = res.pipeline_latency;
  return `Multi-Stage 추천 ${Math.round(l.total_ms)}ms (후보 ${Math.round(l.candidate_ms)} · 정렬 ${Math.round(l.ranking_ms)} · 재정렬 ${Math.round(l.reranking_ms)})`;
}

/** 추천 칸 채우기. exclude = 같은 화면에 이미 나온 상품. */
async function fillRecommendations(page, exclude, { name = "rec", captionPrefix = "" } = {}) {
  try {
    const res = await recommend(exclude);
    const items = await withInfo(res.shown);
    slot(page, name).innerHTML = items.length
      ? items.map(i => card(i, { source: "home", reason: i.reason, hideable: true })).join("")
      : `<p class="caption">추천할 상품이 없어요.</p>`;
    const parts = [captionPrefix, latencyText(res)];
    if (exclude.size) parts.push("이 화면에 나온 상품은 빼고 추천");
    slot(page, `${name}-caption`).textContent = parts.filter(Boolean).join(" · ");
    return res;
  } catch (err) {
    showError(slot(page, name), err);
    return null;
  }
}

/** 최근 본 상품 몇 개의 "비슷한 상품"을 번갈아 섞는다 (이미 본 상품 제외). 사진 유사도라 클릭하면 바로 바뀐다. */
async function similarToRecent() {
  const basis = state.recent.slice(-CONFIG.recentBasis).reverse();
  const lists = await Promise.all(basis.map(pid =>
    api(`/api/products/${encodeURIComponent(pid)}/similar?top_k=${CONFIG.similarTopK}`).then(r => r.results).catch(() => [])));
  const seen = new Set(state.recent);
  const items = [];
  for (let k = 0; k < CONFIG.similarTopK && items.length < CONFIG.recTopN; k++) {
    for (const list of lists) {
      const item = list[k];
      if (item && !seen.has(item.product_id) && items.length < CONFIG.recTopN) {
        seen.add(item.product_id);
        items.push(item);
      }
    }
  }
  return { basis, items };
}

// ------------------------------------------------------------------ 홈
async function renderHome() {
  const hasRecent = state.recent.length > 0;
  const page = newPage(`
    <div class="hero" data-slot="hero"><h1>${esc(heroTitle())}</h1><p data-slot="hero-sub">추천을 불러오는 중이에요.</p></div>
    <div data-slot="notice"></div>
    ${hasRecent ? block("최근 본 상품과 비슷한 상품", "similar") : ""}
    ${block("회원님을 위한 추천", "rec")}
    <div data-slot="recent-block" hidden>${block(state.views ? "이번 방문에서 본 상품" : "지난번에 본 상품", "recent", { strip: true })}</div>`);
  const shown = new Set();
  if (hasRecent) {
    const { basis, items } = await similarToRecent();
    items.forEach(i => shown.add(i.product_id));
    slot(page, "similar").innerHTML = items.map(i => card(i, { source: "similar" })).join("")
      || `<p class="caption">비슷한 상품을 찾지 못했어요.</p>`;
    slot(page, "similar-caption").textContent = `최근 본 상품 ${basis.length}개와 사진이 비슷한 상품 · 같은 고객층`;
  }
  const res = await fillRecommendations(page, shown);
  if (!res) return;

  const ctx = res.session_context;
  const sub = [];
  if (state.views) sub.push(`이번 방문에서 상품 ${state.views}개를 봤어요.`);
  if (ctx.session_interest) sub.push(`요즘 관심: ${ctx.session_interest}`);
  slot(page, "hero-sub").textContent = sub.join(" ") || "상품을 눌러 보면 관심사에 맞춰 추천이 바로 바뀌어요.";
  if (res.isCold) {
    const left = Math.max(0, CONFIG.minPicks - state.views);
    slot(page, "notice").innerHTML = `<p class="notice warn">아직 취향을 잘 몰라서 인기·급상승 상품을 보여드리고 있어요.${
      state.isNew && left ? ` 상품을 ${left}개 더 보면 맞춤 추천으로 바뀌어요.` : ""}</p>`;
  }
  const recent = [...ctx.recent_clicks].reverse().slice(0, 12);
  if (recent.length) {
    slot(page, "recent-block").hidden = false;
    const items = await withInfo(recent.map(pid => ({ product_id: pid })));
    slot(page, "recent").innerHTML = items.map(i => card({ ...i, category: null, color: null }, { source: "home" })).join("");
  }
}

function heroTitle() {
  if (state.isNew) return state.views ? "취향 조사와 클릭을 반영한 추천이에요" : "처음 오셨네요, 반가워요";
  return `${state.userId}님, 다시 오셨네요`;
}

// ------------------------------------------------------------------ 취향 조사
async function renderSurvey() {
  if (!SURVEY[state.audience]) {
    const page = newPage(`
      <div class="hero"><h1>어떤 옷을 찾으세요?</h1><p>취향 조사를 시작할 고객층을 골라 주세요. 검색할 때도 이 고객층 상품만 보여드려요.</p></div>
      <div class="choice-row">${["women", "men", "kids"].map(a =>
        `<button class="choice" type="button" data-audience="${a}">${AUDIENCE[a]}</button>`).join("")}</div>`);
    page.addEventListener("click", e => {
      const btn = e.target.closest("[data-audience]");
      if (!btn) return;
      state.audience = btn.dataset.audience;
      saveState();
      syncControls();
      renderSurvey();
    });
    return;
  }

  const picks = new Set();
  const page = newPage(`
    <div class="hero"><h1>마음에 드는 상품을 ${CONFIG.minPicks}개 이상 골라 주세요</h1>
      <p>고른 상품을 관심 상품으로 기록해서 첫 추천을 만들어요. 나중에 상품을 둘러보면 추천이 계속 바뀌어요.</p></div>
    <div data-slot="styles">${skeletons(5)}</div>
    <div class="survey-bar">
      <span><strong data-slot="count">0</strong> / ${CONFIG.minPicks}개 선택 ·
        <button class="link-btn" type="button" data-action="audience">${esc(AUDIENCE[state.audience])} 상품 (바꾸기)</button></span>
      <span><button class="link-btn" type="button" data-action="skip">건너뛰기</button>&nbsp;&nbsp;
        <button class="primary" type="button" data-action="done" disabled>추천 받기</button></span>
    </div>`);

  page.addEventListener("click", async e => {
    const pick = e.target.closest("[data-pick]");
    if (pick) {
      const pid = pick.dataset.pick;
      picks.has(pid) ? picks.delete(pid) : picks.add(pid);
      pick.setAttribute("aria-pressed", String(picks.has(pid)));
      slot(page, "count").textContent = picks.size;
      page.querySelector('[data-action="done"]').disabled = picks.size < CONFIG.minPicks;
      return;
    }
    const button = e.target.closest("[data-action]");
    const action = button?.dataset.action;
    if (action === "audience") {
      state.audience = null;
      saveState();
      renderSurvey();
    } else if (action === "skip") {
      state.onboarded = true;
      saveState();
      location.hash = "#/";
    } else if (action === "done") {
      button.disabled = true;
      button.textContent = "추천 만드는 중...";
      try {
        for (const pid of picks) await logView(pid, "onboarding"); // 고른 순서대로 세션 클릭에 쌓는다
        state.onboarded = true;
        saveState();
        location.hash = "#/";
      } catch (err) {
        toast(`기록하지 못했어요: ${err.message}`);
        button.disabled = false;
        button.textContent = "추천 받기";
      }
    }
  });

  try {
    const seen = new Set();
    const groups = await Promise.all(SURVEY[state.audience].map(async s => {
      const lists = await Promise.all(s.queries.map(q =>
        post("/api/search", { query_text: q, top_k: 6, audience: state.audience }).then(r => r.results).catch(() => [])));
      return { style: s.style, lists };
    }));
    const html = groups.map(g => {
      const items = g.lists.flatMap(list => list.filter(i => !seen.has(i.product_id)).slice(0, 2));
      items.forEach(i => seen.add(i.product_id));
      if (!items.length) return "";
      return `<div class="style-group"><h3>${esc(g.style)}</h3><div class="grid">${items.map(i => `
        <button class="pick" type="button" data-pick="${esc(i.product_id)}" aria-pressed="false">
          <span class="card-media"><img src="${imageUrl(i.product_id)}" alt="" loading="lazy"><span class="check">✓</span></span>
          <span class="card-body"><span class="card-name">${esc(i.name)}</span><span class="card-price">${price(i.price)}</span></span>
        </button>`).join("")}</div></div>`;
    }).join("");
    slot(page, "styles").innerHTML = html || `<p class="error">취향 조사 상품을 불러오지 못했어요.</p>`;
  } catch (err) {
    showError(slot(page, "styles"), err);
  }
}

// ------------------------------------------------------------------ 검색 (문장)
async function renderSearch(query) {
  document.getElementById("q").value = query;
  const page = newPage(`
    <div data-slot="notice"></div>
    ${block(`‘${query}’ 검색 결과`, "results")}
    ${block("이런 상품은 어때요", "rec")}`);
  if (/[가-힣]/.test(query)) {
    slot(page, "notice").innerHTML = `<p class="notice warn">검색은 영어 기준으로 만들어서 한국어 검색어는 결과가 정확하지 않을 수 있어요.</p>`;
  }
  let shown = new Set();
  try {
    const res = await post("/api/search", { query_text: query, top_k: CONFIG.searchTopK, audience: searchAudience() });
    shown = new Set(res.results.map(r => r.product_id));
    slot(page, "results").innerHTML = res.results.length
      ? res.results.map(r => card(r, { source: "search" })).join("")
      : `<p class="caption">검색 결과가 없어요. 다른 표현이나 고객층(${esc(AUDIENCE[searchAudience()])})을 바꿔 보세요.</p>`;
    slot(page, "results-caption").textContent =
      `${SEARCH_TYPE[res.search_type] || res.search_type} · ${AUDIENCE[searchAudience()]} · ${Math.round(res.latency_ms)}ms`;
  } catch (err) {
    showError(slot(page, "results"), err);
  }
  await fillRecommendations(page, shown);
}

// ------------------------------------------------------------------ 검색 (사진)
async function renderPhoto() {
  if (!photo) {
    location.replace("#/");
    return;
  }
  const page = newPage(`
    <div class="photo-layout">
      <div class="photo-frame">
        <div class="photo-box" data-slot="photo"><img src="${photo.dataUrl}" alt="검색한 사진"></div>
        <p class="caption" data-slot="photo-caption">사진 속 옷을 찾는 중이에요.</p>
      </div>
      <div data-slot="groups">${block("사진으로 찾은 상품", "loading")}</div>
    </div>
    ${block("이런 상품은 어때요", "rec")}`);
  let shown = new Set();
  try {
    const res = await post("/api/search/style", {
      image_base64: photo.base64, query_text: photo.query || null, top_k: CONFIG.photoTopK, audience: styleAudience(),
    });
    const boxes = [];
    const groups = res.groups.map((g, k) => {
      const color = BOX_COLORS[k % BOX_COLORS.length];
      if (g.box) boxes.push({ box: g.box, label: g.slot_name, color });
      g.results.forEach(r => shown.add(r.product_id));
      return `<section class="block" style="--box:${color}">
        <div class="block-head"><div><h2 class="group-title">${g.box ? `<span class="dot"></span>` : ""}${
          esc(g.slot === "all" ? "사진과 비슷한 상품" : g.slot_name)}</h2>
          <p class="caption">${esc(BASIS[g.basis] || g.basis)} · 후보 ${g.total_count.toLocaleString()}개</p></div></div>
        <div class="grid">${g.results.length ? g.results.map(r => card(r, { source: "search" })).join("")
          : `<p class="caption">조건에 맞는 상품이 없어요.</p>`}</div></section>`;
    });
    slot(page, "groups").innerHTML = groups.join("") || `<p class="caption">사진에서 찾은 상품이 없어요.</p>`;
    slot(page, "photo").insertAdjacentHTML("beforeend", boxes.map(b => {
      const [x1, y1, x2, y2] = b.box;
      const pos = `left:${(x1 / photo.width) * 100}%;top:${(y1 / photo.height) * 100}%;` +
        `width:${((x2 - x1) / photo.width) * 100}%;height:${((y2 - y1) / photo.height) * 100}%;`;
      return `<div class="det-box" style="${pos}--box:${b.color}"><span>${esc(b.label)}</span></div>`;
    }).join(""));
    const kind = res.photo_type === "style" ? `착용 사진 · 옷 ${boxes.length}벌` : "상품 사진";
    const who = res.audience ? ` · ${AUDIENCE[res.audience] || res.audience}` : "";
    slot(page, "photo-caption").textContent =
      `${kind}${who}${photo.query ? ` · 요청 “${photo.query}”` : ""} · ${Math.round(res.latency_ms)}ms`;
  } catch (err) {
    showError(slot(page, "groups"), err);
  }
  await fillRecommendations(page, shown);
}

// ------------------------------------------------------------------ 상품 상세
async function renderProduct(pid) {
  const page = newPage(`
    <div class="detail">
      <div class="card-media"><img src="${imageUrl(pid)}" alt=""></div>
      <div data-slot="info"><div class="skeleton"><div class="bar"></div><div class="bar"></div></div></div>
    </div>
    ${block("비슷한 상품", "similar", { caption: "사진이 비슷한 다른 디자인" })}
    ${block("회원님을 위한 추천", "rec")}`);
  const info = await productInfo(pid);
  slot(page, "info").innerHTML = info
    ? `<p class="caption">${esc(info.category)}</p><h1>${esc(info.name)}</h1>
       <p class="price">${price(info.price)}</p>
       <div class="chips"><span class="chip">색상 ${esc(info.color)}</span></div>
       <p class="caption" style="margin-top:18px">이 상품을 본 기록이 추천에 바로 반영돼요.</p>`
    : `<p class="error">상품 정보를 찾을 수 없어요.</p>`;

  const shown = new Set([pid]);
  try {
    const sim = await api(`/api/products/${encodeURIComponent(pid)}/similar?top_k=${CONFIG.similarTopK}`);
    sim.results.forEach(r => shown.add(r.product_id));
    slot(page, "similar").innerHTML = sim.results.map(r => card(r, { source: "similar" })).join("")
      || `<p class="caption">비슷한 상품이 없어요.</p>`;
    slot(page, "similar-caption").textContent = `사진이 비슷한 다른 디자인 · 같은 고객층 · ${Math.round(sim.latency_ms)}ms`;
  } catch (err) {
    showError(slot(page, "similar"), err);
  }
  await fillRecommendations(page, shown);
}

// ------------------------------------------------------------------ 라우터
const ROUTES = [
  [/^#\/onboarding$/, () => renderSurvey()],
  [/^#\/search\?(.*)$/, m => renderSearch(new URLSearchParams(m[1]).get("q") || "")],
  [/^#\/photo$/, () => renderPhoto()],
  [/^#\/product\/(.+)$/, m => renderProduct(decodeURIComponent(m[1]))],
];

function route() {
  const hash = location.hash || "#/";
  if (state.isNew && !state.onboarded && hash !== "#/onboarding") {
    location.replace("#/onboarding");
    return;
  }
  for (const [pattern, render] of ROUTES) {
    const m = hash.match(pattern);
    if (m) return render(m);
  }
  return renderHome();
}
function navigate(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

// ------------------------------------------------------------------ 상단 바 · 하단 · 알림
function syncControls() {
  const audience = document.getElementById("audience");
  audience.innerHTML = Object.entries(AUDIENCE).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  audience.value = state.audience || "all";
  const member = document.getElementById("member");
  member.innerHTML = [`<option value="new">신규 회원 (취향 조사)</option>`,
    ...CONFIG.members.map(m => `<option value="${m}">회원 ${m}</option>`)].join("");
  member.value = state.member;
  renderFooter();
}

function renderFooter() {
  document.getElementById("footer").innerHTML = [
    `회원 ${esc(state.userId)}`, `방문 ${esc(state.sessionId)}`, `이번 방문 클릭 ${state.views}개`,
    `추천 기준 시각 ${CONFIG.demoAt.slice(0, 10)} (로그 데이터 끝)`,
    `검색·추천 API: <a href="/docs" target="_blank" rel="noopener">/docs</a>`,
  ].map(t => `<span>${t}</span>`).join("");
}

let toastTimer;
function toast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

/** 사진을 긴 변 1024px JPEG로 줄여서 보낸다 (휴대폰 사진은 크고, 박스 좌표는 줄인 사진 기준). */
async function readPhoto(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = () => reject(new Error("사진을 열 수 없어요."));
      i.src = url;
    });
    const scale = Math.min(1, 1024 / Math.max(img.naturalWidth, img.naturalHeight));
    const width = Math.round(img.naturalWidth * scale);
    const height = Math.round(img.naturalHeight * scale);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(img, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    return { dataUrl, base64: dataUrl.split(",")[1], width, height };
  } finally {
    URL.revokeObjectURL(url);
  }
}

// ------------------------------------------------------------------ 이벤트
document.addEventListener("click", async e => {
  const view = e.target.closest("[data-view]");
  if (view) {
    e.preventDefault();
    const pid = view.dataset.view;
    try {
      await logView(pid, view.dataset.source || "home");
    } catch (err) {
      toast(`클릭을 기록하지 못했어요: ${err.message}`);
    }
    navigate(`#/product/${encodeURIComponent(pid)}`);
    return;
  }
  const hide = e.target.closest("[data-hide]");
  if (hide) {
    const pid = hide.dataset.hide;
    if (!state.hidden.includes(pid)) state.hidden.push(pid);
    saveState();
    const cardEl = hide.closest(".card");
    cardEl.classList.add("leaving");
    setTimeout(() => cardEl.remove(), 200);
    toast("이 상품은 앞으로 추천하지 않을게요.");
  }
});

document.getElementById("search-form").addEventListener("submit", e => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (q) navigate(`#/search?${new URLSearchParams({ q })}`);
});

document.getElementById("photo-input").addEventListener("change", async e => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  try {
    photo = { ...(await readPhoto(file)), query: document.getElementById("q").value.trim() };
    navigate("#/photo");
  } catch (err) {
    toast(err.message);
  }
});

document.getElementById("audience").addEventListener("change", e => {
  state.audience = e.target.value;
  saveState();
  route();
});

document.getElementById("member").addEventListener("change", e => {
  state = freshState(e.target.value);
  saveState();
  syncControls();
  navigate(state.isNew ? "#/onboarding" : "#/");
});

document.getElementById("new-visit").addEventListener("click", () => {
  const audience = state.audience;
  state = freshState(state.member);
  if (!state.isNew) state.audience = audience;
  saveState();
  syncControls();
  toast("새 방문을 시작했어요. 이번 방문의 클릭이 비워졌어요.");
  navigate(state.isNew ? "#/onboarding" : "#/");
});

window.addEventListener("hashchange", route);
syncControls();
route();
