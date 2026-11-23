/* Local closet extension. Uses the team's router, member selector and design tokens. */
"use strict";

function closetVisitorId() {
  try {
    let id = localStorage.getItem('closet-visitor-v1');
    if (!/^NEW-[a-f0-9]{12}$/.test(id || '')) {
      id = `NEW-${randomHex(12)}`;
      localStorage.setItem('closet-visitor-v1', id);
    }
    return id;
  } catch { return `NEW-${randomHex(12)}`; }
}

const closetSlots = {top:'상의', bottom:'하의', outer:'아우터', dress:'원피스', shoes:'신발'};
let closetCatalogPromise;
let closetReturnHash = '#/mypage';
const closetSelections = new Map();
const closetCategorySelections = new Map();
const closetAudienceLabels = {all:'전체', men:'남성복', women:'여성복', unisex:'공용', kids:'아동복'};
function closetAudienceOptions(value) {
  return ['men','women','unisex','all'].map(key => `<option value="${key}" ${value === key ? 'selected' : ''}>${closetAudienceLabels[key]}</option>`).join('');
}
function closetModelLabel(context) {
  return context?.enabled ? `<span class="caption model-scope-label">${esc(context.label)}</span>` : '';
}
function closetRegistrationForm(data) {
  if (!data.publisher_model_filter?.enabled) return '';
  const info = data.registration || {};
  const options = (items, value) => Object.entries(items).map(([key, label]) => `<option value="${key}" ${key === (value || 'unknown') ? 'selected' : ''}>${label}</option>`).join('');
  return `<details class="discovery-account"><summary>회원 정보</summary><form data-registration-form>
    <label>성별<select name="gender" aria-label="회원 성별">${options({unknown:'미입력', M:'남성', F:'여성'}, info.gender)}</select></label>
    <label>연령대<select name="age_group" aria-label="회원 연령대">${options({unknown:'미입력', '10s':'10대', '20s':'20대', '30s':'30대', '40s':'40대', '50s':'50대', '60s+':'60대 이상'}, info.age_group)}</select></label>
    <button class="text-btn" type="submit">저장</button><span role="status" data-registration-status></span>
    </form><p>가입 성별에 맞는 모델의 코디를 보여줘요. 미입력 시 전체 모델을 보여줘요.</p></details>`;
}
function bindClosetRegistration(page, uid, refresh) {
  const form = page.querySelector('[data-registration-form]');
  if (!form) return;
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button'); button.disabled = true;
    try {
      await post(closetUrl('registration', {}, uid), {gender:form.elements.gender.value, age_group:form.elements.age_group.value});
      if (page.isConnected && state.userId === uid) await refresh();
    } catch (error) { if (page.isConnected) form.querySelector('[data-registration-status]').textContent = error.message; }
    finally { button.disabled = false; }
  });
}
function closetPreferenceText(preference) {
  if (!preference?.ready) return '';
  if (preference.source === 'saved_outfits') return '저장한 코디 반영';
  if (preference.source === 'saved_and_views') return '저장한 코디와 최근 관심 반영';
  if (preference.source === 'survey_selection') return '선택한 코디를 반영했어요';
  if (preference.source === 'survey_and_views') return '선택한 코디와 최근 관심 반영';
  return preference.source === 'registration_cohort' ? '가입 정보를 참고한 첫 추천' : '최근 본 코디 반영';
}
function closetReturnLink() {
  return `<a class="link-btn" href="${esc(closetReturnHash)}">← ${closetReturnHash.startsWith('#/closet/outfit/') ? '코디 상세' : '내 옷장'}</a>`;
}
function closetCatalog(uid = state.userId) {
  // Membership changes affect demo cohorts; do not share a cached catalog across users.
  return api(closetUrl('catalog', {}, uid));
}
function closetUrl(path, values = {}, uid = state.userId) {
  return '/api/closet/' + path + '?' + new URLSearchParams({user_id: uid, ...values});
}
async function closetUpdate(action, values, uid = state.userId) {
  return post(closetUrl('profile', {}, uid), {action, values});
}
function bindOutfitSave(button, {uid, oid, pid, saved, onChange, isCurrent = () => button.isConnected}) {
  let busy = false;
  const paint = () => {
    button.setAttribute('aria-pressed', String(saved));
    button.textContent = saved ? '♥ 저장됨' : '♡ 코디 저장';
    button.setAttribute('aria-label', saved ? '코디 저장 취소' : '코디 저장');
  };
  paint();
  button.disabled = false;
  button.onclick = async () => {
    if (busy) return;
    busy = true; button.disabled = true;
    const status = button.parentElement.querySelector('.saved-status');
    if (status) status.textContent = '';
    try {
      const result = await post(closetUrl('saved-outfits', {}, uid), {outfit_id:oid, product_id:pid, saved:!saved});
      if (state.userId !== uid) return;
      saved = result.saved;
      if (isCurrent()) { paint(); if (status) status.textContent = saved ? '저장한 코디를 추천에 반영했어요.' : '저장으로 반영된 취향을 해제했어요.'; }
      onChange?.();
    } catch (error) { if (isCurrent() && status) status.textContent = `저장 변경 실패: ${error.message}`; }
    finally { busy = false; if (isCurrent()) button.disabled = false; }
  };
}
function closetName(p, catalog) {
  return catalog?.products.find(item => item.id === p.id)?.display_name || p.name;
}
function closetPhoto(src, name, extra = '') {
  return `<div class="closet-photo ${extra}"><img src="${esc(src)}" alt="${esc(name)}" loading="lazy"></div>`;
}
function closetOutfitLink(pid, mode) {
  return '#/closet/outfits/' + encodeURIComponent(pid) + '?' + new URLSearchParams({mode});
}
function closetError(page, error) {
  if (!page.isConnected) return;
  page.innerHTML = `<a class="link-btn" href="#/mypage">← 마이페이지</a><div class="closet-empty"><h2>지금은 불러올 수 없어요</h2><p>${esc(error.message)}</p><button class="text-btn" data-closet-retry>다시 시도</button></div>`;
}

async function renderCloset() {
  const uid = state.userId;
  const page = newPage('<div class="mypage-heading"><h1>마이페이지</h1></div><p class="caption">옷장을 불러오는 중입니다.</p>');
  try {
    const [catalog, profile] = await Promise.all([closetCatalog(), api(closetUrl('profile', {}, uid))]);
    if (!page.isConnected || state.userId !== uid) return;
    page.className = 'wardrobe-page';
    page.innerHTML = `
      <div class="closet-heading mypage-heading"><h1>마이페이지</h1><div class="mypage-tools"><button class="link-btn" type="button" data-saved-jump>저장한 코디</button><a class="mypage-shop-link" href="#/discover">코디 둘러보기</a></div></div>
      ${closetRegistrationForm(profile)}
      <div class="mypage-shell">
      <div class="mypage-content">
      <section data-mypage-panel="wardrobe" aria-labelledby="mypage-wardrobe-title">
      <div class="mypage-section-heading"><div><h2 id="mypage-wardrobe-title">내 옷장</h2><p class="caption">옷을 고르면 아래에서 코디를 볼 수 있어요.</p></div><div class="wardrobe-scope">${catalog.cohort_enabled ? `<label for="wardrobe-gender">시연 성별</label><select id="wardrobe-gender">${Object.entries({M:'남성', F:'여성', unknown:'전체'}).map(([key, label]) => `<option value="${key}" ${(profile.registration?.gender || 'unknown') === key ? 'selected' : ''}>${label}</option>`).join('')}</select>` : `${closetModelLabel(profile.publisher_model_filter)}<label for="wardrobe-audience">의류 범위</label><select id="wardrobe-audience">${closetAudienceOptions(profile.audience)}</select>`}<button class="mypage-add-button" type="button" id="mypage-add">상품 선택</button></div></div>
      <div class="wardrobe-categories" role="group" aria-label="옷 종류">${Object.entries({all:'전체',...closetSlots}).map(([key,label]) => `<button type="button" data-wardrobe-category="${key}" aria-pressed="false" aria-controls="wardrobe-items" aria-label="${label}">${label}<span>${profile.products.filter(p => key === 'all' || p.slot === key).length}</span></button>`).join('')}</div>
      <div class="closet-toolbar mypage-filter"><span class="caption" data-slot="visible-count" role="status">전체 ${profile.products.length}개</span><div class="mypage-tools"><button type="button" class="link-btn" id="mypage-edit" aria-pressed="false">편집</button></div></div>
      <div class="wardrobe-rack-tools"><button type="button" class="link-btn" data-rack-expand aria-expanded="false">전체 보기</button><button type="button" class="link-btn" data-rack-shift="-1" aria-label="이전 옷">←</button><button type="button" class="link-btn" data-rack-shift="1" aria-label="다음 옷">→</button></div>
      <div class="closet-grid wardrobe-rack" id="wardrobe-items" data-slot="owned" aria-label="보유한 옷"></div>
      <section class="saved-looks-section" aria-labelledby="saved-looks-title"><div class="mypage-section-heading"><h2 id="saved-looks-title" tabindex="-1">저장한 코디 <span data-saved-count></span></h2></div><div data-saved-list><p role="status" class="caption">저장한 코디를 불러오는 중…</p></div></section>
      <details class="closet-add-panel" ${profile.products.length ? '' : 'open'}><summary>시연 상품 선택</summary><p class="caption">준비된 상품에서 옷장을 구성하세요.</p>
      <form id="closet-add-form"><label for="closet-add-select">추가할 상품</label><select id="closet-add-select">${catalog.products.filter(p => !profile.owned.includes(p.id)).map(p => `<option value="${esc(p.id)}">${esc(p.display_name)} · ${esc(p.color || p.category)}</option>`).join('')}</select><button class="primary" ${catalog.products.every(p => profile.owned.includes(p.id)) ? 'disabled' : ''}>옷장에 추가</button></form>
      <button class="link-btn" type="button" id="closet-demo">시연 상품 ${catalog.demo_ids.length}개 담기</button> <button class="link-btn" type="button" data-closet-shoe-demo ${catalog.shoe_demo_ids.length ? '' : 'hidden'}>신발 시연 상품 ${catalog.shoe_demo_ids.length}개 담기</button><button class="link-btn" type="button" data-closet-dress-demo ${catalog.dress_demo_ids.length ? '' : 'hidden'}>원피스 시연 상품 ${catalog.dress_demo_ids.length}개 담기</button><p class="caption">기존 옷장에 시연 상품을 추가합니다. 실제 구매 이력이 아닙니다.</p></details>
      </section>
      </div></div><p class="caption closet-storage">옷장과 코디 조회 기록은 이 PC에 저장됩니다. 회원 선택은 로컬 시연용입니다.</p>`;
    bindClosetRegistration(page, uid, renderCloset);
    let savedRequest = 0;
    let savedLooks = [];
    const savedList = page.querySelector('[data-saved-list]');
    async function refreshSavedLooks() {
      const current = ++savedRequest;
      try {
        const data = await api(closetUrl('saved-outfits', {}, uid));
        if (!page.isConnected || uid !== state.userId || current !== savedRequest) return;
        savedLooks = data.outfits;
        page.querySelector('[data-saved-count]').textContent = data.outfits.length;
        savedList.innerHTML = `${data.hidden_count ? `<p class="caption">현재 성별·추천 범위 밖의 저장 코디 ${data.hidden_count}개는 보관 중이에요.</p>` : ''}<div class="saved-looks-grid">${data.outfits.map((o, i) => `<article><button type="button" class="mypage-outfit-select" data-look-id="${esc(o.id)}" data-saved-open="${i}" aria-haspopup="dialog" aria-label="저장한 ${esc(o.product.display_name)} 코디 보기">${closetPhoto(o.image, o.title, 'outfit-photo')}<div class="closet-card-body"><h3>${esc(o.product.display_name)}</h3><span class="caption">코디·유사상품 보기 ↗</span></div></button><div class="saved-card-actions"><button class="look-save" type="button" data-saved-toggle="${i}"></button><span class="saved-status" role="status"></span></div></article>`).join('')}</div>${!data.outfits.length ? '<p class="saved-empty">마음에 드는 코디에서 ♡ 코디 저장을 눌러보세요.</p>' : ''}`;
        savedList.querySelectorAll('[data-saved-toggle]').forEach(button => {
          const o = savedLooks[Number(button.dataset.savedToggle)];
          bindOutfitSave(button, {uid, oid:o.id, pid:o.product_id, saved:true, onChange:async () => { await refreshSavedRecommendations(); if (page.isConnected && !document.querySelector('.wardrobe-look-dialog')) page.querySelector('#saved-looks-title').focus({preventScroll:true}); }});
        });
      } catch (error) {
        if (!page.isConnected || current !== savedRequest) return;
        savedList.innerHTML = `<p role="alert" class="caption">저장한 코디를 불러오지 못했어요. ${esc(error.message)}</p><button type="button" class="link-btn" data-saved-retry>다시 시도</button>`;
      }
    }
    refreshSavedLooks();
    page.querySelector('[data-saved-jump]').addEventListener('click', () => {
      page.querySelector('.saved-looks-section').scrollIntoView({block:'start', behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
      page.querySelector('#saved-looks-title').focus({preventScroll:true});
    });
    savedList.addEventListener('click', event => {
      if (event.target.closest('[data-saved-retry]')) refreshSavedLooks();
      const button = event.target.closest('[data-saved-open]');
      if (button) {
        const index = Number(button.dataset.savedOpen);
        openWardrobeLook({looks:savedLooks, index, product:savedLooks[index].product, catalog, uid, opener:button, onSavedChange:refreshSavedRecommendations});
      }
    });
    page.querySelector('#mypage-add').addEventListener('click', () => {
      const panel = page.querySelector('.closet-add-panel');
      panel.open = true;
      panel.scrollIntoView({block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
      page.querySelector('#closet-add-select').focus({preventScroll:true});
    });
    let editing = false;
    let selectedId = profile.products.some(p => p.id === closetSelections.get(uid)) ? closetSelections.get(uid) : profile.products.find(p => p.can_recommend)?.id;
    let outfitRequest = 0;
    let outfitLimit = 5;
    let visibleLooks = [];
    let lookProduct = null;
    const outfitPanel = document.createElement('section');
    outfitPanel.id = 'mypage-outfits';
    outfitPanel.className = 'mypage-inline-outfits';
    outfitPanel.setAttribute('aria-label', '선택한 옷의 코디');
    async function refreshSavedRecommendations() {
      await refreshSavedLooks();
      if (page.isConnected && state.userId === uid && selectedId && outfitPanel.isConnected) await selectOutfits(selectedId, false, outfitLimit);
    }
    function placeOutfits() {
      const grid = slot(page, 'owned');
      const cards = [...grid.children].filter(el => el.classList.contains('closet-card'));
      const index = cards.findIndex(el => el.querySelector('[data-closet-select]')?.dataset.closetSelect === selectedId);
      if (index < 0) { outfitPanel.remove(); return; }
      grid.after(outfitPanel);
    }
    async function selectOutfits(pid, scroll = false, limit = 5) {
      outfitLimit = limit;
      selectedId = pid;
      closetSelections.set(uid, pid);
      const request = ++outfitRequest;
      page.querySelectorAll('[data-closet-select]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.closetSelect === pid)));
      outfitPanel.innerHTML = '<p role="status">코디를 불러오는 중입니다.</p>';
      outfitPanel.setAttribute('aria-busy', 'true');
      placeOutfits();
      if (scroll) outfitPanel.scrollIntoView({block:'start', behavior:'instant'});
      try {
        const data = await api(closetUrl('outfits', {product_id:pid, mode:'curated', limit}, uid));
        if (!page.isConnected || state.userId !== uid || request !== outfitRequest) return;
        visibleLooks = data.outfits;
        lookProduct = data.product;
        outfitPanel.innerHTML = `<div class="mypage-section-heading"><div class="wardrobe-selected-item">${closetPhoto(data.product.image, closetName(data.product, catalog))}<div><h2>${esc(closetName(data.product, catalog))} 코디</h2><p class="caption">${esc(closetAudienceLabels[data.audience])} · ${data.outfits.length}${data.has_more ? ' / ' + data.total_count : ''}개${data.outfits.length && data.personalization?.ready ? ' · ' + closetPreferenceText(data.personalization) : ''}</p></div></div><button type="button" class="link-btn" data-outfits-close>접기</button></div>
          <div class="closet-grid outfit-grid">${data.outfits.map((o, index) => `<article class="closet-card"><button type="button" class="mypage-outfit-select" data-look-id="${esc(o.id)}" data-look-open="${index}" aria-haspopup="dialog" aria-label="코디 ${index + 1} · 유사상품 보기">${closetPhoto(o.image, o.title, 'outfit-photo')}<div class="closet-card-body"><span class="look-number">${String(index + 1).padStart(2, '0')}</span><span>${esc(o.style_tag || '코디')}</span><span class="look-open-label">유사상품 보기 ↗</span></div></button></article>`).join('') || `<p class="wardrobe-no-looks">이 옷에 연결된 코디가 없어요. 다른 옷을 골라보세요.</p>`}</div>`;
        if (data.has_more) {
          const more = document.createElement('button');
          more.type = 'button'; more.className = 'link-btn'; more.textContent = '코디 더 보기';
          more.addEventListener('click', () => selectOutfits(pid, false, Math.min(40, limit + 5)));
          outfitPanel.append(more);
        }
      } catch (error) {
        if (!page.isConnected || state.userId !== uid || request !== outfitRequest) return;
        outfitPanel.innerHTML = `<p role="alert">코디를 불러오지 못했어요. ${esc(error.message)}</p><button type="button" class="link-btn" data-outfits-retry>다시 시도</button>`;
      } finally {
        if (request === outfitRequest) outfitPanel.setAttribute('aria-busy', 'false');
      }
    }
    outfitPanel.addEventListener('click', e => {
      const look = e.target.closest('[data-look-open]');
      if (look) openWardrobeLook({looks:visibleLooks, index:Number(look.dataset.lookOpen), product:lookProduct, catalog, uid, opener:look, onSavedChange:refreshSavedRecommendations});
      if (e.target.closest('[data-outfits-retry]')) selectOutfits(selectedId, false);
      if (e.target.closest('[data-outfits-close]')) {
        const button = [...page.querySelectorAll('[data-closet-select]')].find(el => el.dataset.closetSelect === selectedId);
        ++outfitRequest;
        selectedId = null;
        closetSelections.delete(uid);
        outfitPanel.remove();
        button?.setAttribute('aria-pressed', 'false');
        button?.focus();
      }
    });
    slot(page, 'owned').addEventListener('click', e => {
      const button = e.target.closest('[data-closet-select]');
      if (button) selectOutfits(button.dataset.closetSelect);
    });
    page.querySelector('#mypage-edit').addEventListener('click', e => {
      editing = !editing;
      e.currentTarget.textContent = editing ? '편집 완료' : '편집';
      e.currentTarget.setAttribute('aria-pressed', String(editing));
      page.querySelectorAll('[data-closet-remove]').forEach(button => { button.hidden = !editing; });
    });
    function drawOwned(filter = 'all') {
      const products = profile.products.filter(p => filter === 'all' || p.slot === filter);
      closetCategorySelections.set(uid, filter);
      page.querySelectorAll('[data-wardrobe-category]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.wardrobeCategory === filter)));
      if (!products.some(p => p.id === selectedId)) {
        ++outfitRequest;
        selectedId = products.find(p => p.can_recommend)?.id || null;
        closetSelections.delete(uid);
        outfitPanel.remove();
      }
      slot(page, 'visible-count').textContent = `${filter === 'all' ? '전체' : closetSlots[filter]} ${products.length}개`;
      slot(page, 'owned').innerHTML = products.length ? products.map(p => {
        const name = closetName(p, catalog);
        const content = `${closetPhoto(p.image, name)}<div class="closet-card-body"><p class="caption">${esc(closetSlots[p.slot] || p.category)}</p><h3>${esc(name)}</h3></div>`;
        return `<article class="closet-card">${p.can_recommend ? `<button type="button" class="mypage-product-select" data-closet-select="${esc(p.id)}" aria-controls="mypage-outfits" aria-pressed="${p.id === selectedId}" aria-label="${esc(name)} 선택">${content}</button>` : `${content}<p class="caption">코디 검색 준비 중</p>`}
        <button class="link-btn mypage-remove" type="button" data-closet-remove="${esc(p.id)}" ${editing ? '' : 'hidden'}>옷장에서 빼기</button></article>`;
      }).join('') : `<div class="closet-empty"><h3>${filter === 'all' ? '옷장이 비어 있습니다' : `${closetSlots[filter]} 목록이 비어 있습니다`}</h3>${['shoes','dress'].includes(filter) && catalog[filter === 'dress' ? 'dress_demo_ids' : 'shoe_demo_ids'].length ? `<button type="button" class="text-btn" data-closet-${filter === 'dress' ? 'dress' : 'shoe'}-demo>${closetSlots[filter]} 시연 상품 ${catalog[filter === 'dress' ? 'dress_demo_ids' : 'shoe_demo_ids'].length}개 담기</button>` : '<p>다른 종류의 옷을 둘러보거나 옷을 추가해 주세요.</p>'}</div>`;
      const rack = slot(page, 'owned');
      const selectedCard = [...rack.querySelectorAll('[data-closet-select]')].find(el => el.dataset.closetSelect === selectedId);
      rack.scrollLeft = selectedCard ? Math.max(0, selectedCard.offsetLeft - rack.offsetLeft - 8) : 0;
      page.querySelectorAll('[data-rack-shift], [data-rack-expand]').forEach(button => { button.disabled = products.length === 0; });
      if (selectedId && products.some(p => p.id === selectedId)) selectOutfits(selectedId, false);
      else { ++outfitRequest; selectedId = null; closetSelections.delete(uid); outfitPanel.remove(); }
    }
    drawOwned(closetCategorySelections.get(uid) || 'all');
    page.addEventListener('click', async e => {
      const button = e.target.closest('[data-closet-shoe-demo], [data-closet-dress-demo]');
      if (!button) return;
      button.disabled = true;
      try {
        const category = button.hasAttribute('data-closet-dress-demo') ? 'dress' : 'shoes';
        await closetUpdate('add', catalog[category === 'dress' ? 'dress_demo_ids' : 'shoe_demo_ids'], uid);
        closetCategorySelections.set(uid, category);
        if (page.isConnected && state.userId === uid) renderCloset();
      } catch (error) { toast(error.message); button.disabled = false; }
    });
    page.querySelector('#wardrobe-gender')?.addEventListener('change', async e => {
      e.target.disabled = true;
      try {
        await post(closetUrl('registration', {}, uid), {gender:e.target.value, age_group:profile.registration?.age_group || 'unknown'});
        closetSelections.delete(uid);
        closetCategorySelections.set(uid, 'all');
        if (page.isConnected && state.userId === uid) renderCloset();
      } catch (error) { toast(error.message); e.target.disabled = false; }
    });
    page.querySelector('#wardrobe-audience')?.addEventListener('change', async e => {
      e.target.disabled = true;
      try {
        await closetUpdate('audience', [e.target.value], uid);
        if (page.isConnected && state.userId === uid) renderCloset();
      } catch (error) { toast(error.message); e.target.disabled = false; }
    });
    page.querySelector('[data-rack-expand]').addEventListener('click', e => {
      const expanded = slot(page, 'owned').classList.toggle('wardrobe-rack-expanded');
      e.currentTarget.setAttribute('aria-expanded', String(expanded));
      e.currentTarget.textContent = expanded ? '한 줄로 보기' : '전체 보기';
    });
    page.querySelectorAll('[data-rack-shift]').forEach(button => button.addEventListener('click', () => slot(page, 'owned').scrollBy({left:Number(button.dataset.rackShift) * 400, behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'})));
    page.querySelectorAll('[data-wardrobe-category]').forEach(button => button.addEventListener('click', () => drawOwned(button.dataset.wardrobeCategory)));
    page.querySelector('#closet-add-form').addEventListener('submit', async e => {
      e.preventDefault();
      const pid = page.querySelector('#closet-add-select').value;
      if (!pid) return;
      e.submitter.disabled = true;
      try { await closetUpdate('add', [pid], uid); if (page.isConnected && state.userId === uid) renderCloset(); }
      catch (error) { toast(error.message); e.submitter.disabled = false; }
    });
    page.querySelector('#closet-demo').addEventListener('click', async e => {
      e.target.disabled = true;
      try { await closetUpdate('add', catalog.demo_ids, uid); if (page.isConnected && state.userId === uid) renderCloset(); }
      catch (error) { toast(error.message); e.target.disabled = false; }
    });
  } catch (error) { closetError(page, error); }
}

async function renderClosetOutfits(pid, mode) {
  const uid = state.userId;
  const page = newPage('<p class="caption">이 옷을 활용한 코디를 찾고 있어요.</p>');
  try {
    const [data, catalog] = await Promise.all([api(closetUrl('outfits', {product_id: pid, mode}, uid)), closetCatalog()]);
    if (!page.isConnected || state.userId !== uid) return;
    const name = closetName(data.product, catalog);
    page.innerHTML = `<a class="link-btn" href="#/mypage">← 내 옷장</a>
      <div class="closet-anchor">${closetPhoto(data.product.image, name)}<div><p class="eyebrow">OUTFIT IDEAS</p><h1>${esc(name)}로 시작하는 코디</h1><p>내 옷과 비슷한 옷을 활용한 착장입니다.</p><p class="caption">${closetPreferenceText(data.personalization) || '둘러본 코디를 다음 추천에 반영합니다.'}</p></div></div>
      <div class="closet-toolbar"><h2>추천 코디 ${data.outfits.length}개</h2><span class="caption">${data.curated ? '선별 시연' : '전체 검색 후보'}</span></div>
      <div class="closet-grid outfit-grid">${data.outfits.map(o => `<article class="closet-card">${closetPhoto(o.image, o.title, 'outfit-photo')}<div class="closet-card-body"><h3>${esc(o.title)}</h3><p class="caption">${esc(o.style_tag || '스타일 미확인')}</p><p class="closet-reason">${esc(o.preference_reasons.join(' · '))}</p><a class="primary closet-cta" data-closet-outfit="${esc(o.id)}" data-outfit-product="${esc(pid)}" href="#/closet/outfit/${encodeURIComponent(o.id)}?${new URLSearchParams({product_id: pid, mode})}">코디 자세히 보기</a></div></article>`).join('') || '<div class="closet-empty"><h3>현재 기준에 맞는 코디가 없어요</h3><p>다른 옷을 골라보세요. 관련 없는 코디로 채우지 않았어요.</p></div>'}</div>
      <details class="closet-add-panel"><summary>시연 범위</summary><p class="caption">선별 시연은 실제 검색 결과에서 고른 사례입니다. 전체 추천 품질을 의미하지 않습니다.</p>${data.supports_full_search === false ? '' : `<a class="link-btn" href="${closetOutfitLink(pid, data.curated ? 'all' : 'curated')}">${data.curated ? '전체 검색 후보 보기' : '기본 보기로 돌아가기'}</a>`}</details>`;
  } catch (error) { closetError(page, error); }
}

/* Keep the wardrobe and category in place while browsing a look's alternatives. */
function openWardrobeLook({looks, index, product, catalog, uid, opener, onSavedChange}) {
  if (!looks.length || document.querySelector('.wardrobe-look-dialog')) return;
  const dialog = document.createElement('dialog');
  dialog.className = 'wardrobe-look-dialog';
  dialog.setAttribute('aria-labelledby', 'wardrobe-look-title');
  dialog.innerHTML = `<header class="look-dialog-header"><div><p class="caption" data-look-product-name>${esc(closetName(product, catalog))}</p><h2 id="wardrobe-look-title">코디와 유사상품</h2></div><div class="look-dialog-actions"><button type="button" class="look-save" disabled>♡ 코디 저장</button><span class="saved-status" role="status"></span><button type="button" class="look-close" aria-label="닫고 옷장으로 돌아가기" autofocus>닫기 ×</button></div></header>
    <div class="look-dialog-body" aria-busy="true"></div>
    <footer class="look-dialog-footer"><button type="button" class="look-prev" aria-label="이전 코디">← 이전</button><div class="look-thumbnails" aria-label="코디 선택">${looks.map((o, i) => `<button type="button" data-look-index="${i}" aria-label="코디 ${i + 1}" aria-pressed="false"><img src="${esc(o.image)}" alt="" loading="lazy"></button>`).join('')}</div><span class="look-position" role="status"></span><button type="button" class="look-next" aria-label="다음 코디">다음 →</button></footer>`;
  document.body.append(dialog);
  const body = dialog.querySelector('.look-dialog-body');
  const previousOverflow = document.body.style.overflow;
  let request = 0;
  let cleaned = false;
  const viewed = new Set();
  const dismiss = () => dialog.close();
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    ++request;
    document.body.style.overflow = previousOverflow;
    window.removeEventListener('hashchange', dismiss);
    dialog.remove();
    if (opener.isConnected) opener.focus({preventScroll:true});
    else {
      const replacement = [...document.querySelectorAll('[data-look-id]')].find(button => button.dataset.lookId === opener.dataset.lookId && button.hasAttribute('data-saved-open') === opener.hasAttribute('data-saved-open'));
      (replacement || document.querySelector('#saved-looks-title'))?.focus({preventScroll:true});
    }
  };
  dialog.addEventListener('close', cleanup, {once:true});
  window.addEventListener('hashchange', dismiss);
  dialog.querySelector('.look-close').addEventListener('click', dismiss);
  // Native dialog supplies Escape handling, focus trapping and an inert background.
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dismiss();
  });
  dialog.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (event.key === 'ArrowRight' && index < looks.length - 1) { event.preventDefault(); load(index + 1); }
    if (event.key === 'ArrowLeft' && index > 0) { event.preventDefault(); load(index - 1); }
  });
  dialog.querySelector('.look-prev').addEventListener('click', () => load(index - 1));
  dialog.querySelector('.look-next').addEventListener('click', () => load(index + 1));
  dialog.querySelectorAll('[data-look-index]').forEach(button => button.addEventListener('click', () => load(Number(button.dataset.lookIndex))));

  async function load(next) {
    if (next < 0 || next >= looks.length || cleaned) return;
    index = next;
    const current = ++request;
    const look = looks[index];
    product = look.product || product;
    dialog.querySelector('[data-look-product-name]').textContent = closetName(product, catalog);
    const saveButton = dialog.querySelector('.look-save');
    saveButton.disabled = true;
    dialog.querySelector('.saved-status').textContent = '';
    dialog.querySelector('.look-prev').disabled = index === 0;
    dialog.querySelector('.look-next').disabled = index === looks.length - 1;
    dialog.querySelector('.look-position').textContent = `${index + 1} / ${looks.length}`;
    dialog.querySelectorAll('[data-look-index]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.lookIndex) === index)));
    body.setAttribute('aria-busy', 'true');
    body.innerHTML = '<p class="look-loading" role="status">코디를 불러오는 중…</p>';
    try {
      const data = await api(closetUrl('outfit', {product_id:product.id, outfit_id:look.id, mode:'curated'}, uid));
      if (cleaned || current !== request || state.userId !== uid) return;
      bindOutfitSave(saveButton, {uid, oid:look.id, pid:product.id, saved:data.saved, onChange:onSavedChange, isCurrent:() => !cleaned && current === request});
      const groups = data.groups.filter(g => !g.is_anchor && (g.products.length || g.owned_count));
      if (data.shoe_alternatives?.length || data.shoe_alternatives_owned_count) groups.push({label:'신발', products:data.shoe_alternatives || [], owned_count:data.shoe_alternatives_owned_count || 0});
      const alternatives = groups.flatMap(g => g.products);
      const hidden = groups.reduce((total, g) => total + (g.owned_count || 0), 0);
      body.innerHTML = `<div class="look-stage"><img class="look-main-image" src="${esc(data.image)}" alt="${esc(data.title)}"><button type="button" class="look-back-photo" hidden>← 코디 사진으로</button><span class="look-image-caption">코디 ${String(index + 1).padStart(2, '0')}</span></div>
        <section class="look-products" aria-label="함께 입을 유사상품"><div class="look-anchor-summary">${closetPhoto(product.image, closetName(product, catalog))}<div><span class="caption">선택한 내 옷</span><p>${esc(closetName(product, catalog))}</p></div></div>
        <h3>함께 입을 유사상품 <span>${alternatives.length}</span></h3><p class="caption">사진 속 상품과 색·소재·핏이 다를 수 있어요.</p>
        ${groups.map(g => g.products.length ? `<section class="look-product-group"><h4>${esc(g.label)}</h4><div class="look-product-grid">${g.products.map(p => `<button type="button" class="look-product-preview" data-preview-product="${esc(p.id)}" aria-label="${esc(p.name)} 사진 크게 보기" aria-pressed="false">${closetPhoto(p.image, p.name)}<span class="look-product-name">${esc(p.name)}</span><span class="look-product-meta">${esc(p.color || '')}<span>확대 ↗</span></span></button>`).join('')}</div></section>` : '').join('')}
        ${!alternatives.length ? `<div class="look-no-products"><p>${hidden ? '연결된 유사상품을 이미 보유하고 있어요.' : '이 코디에 연결된 유사상품이 아직 없어요.'}</p><button type="button" class="link-btn" data-look-another ${looks.length < 2 ? 'hidden' : ''}>다른 코디 보기 →</button></div>` : ''}
        ${hidden ? `<p class="caption look-owned-excluded">보유한 상품 ${hidden}개 제외</p>` : ''}</section>`;
      body.scrollTop = 0;
      const mainImage = body.querySelector('.look-main-image');
      const backPhoto = body.querySelector('.look-back-photo');
      const caption = body.querySelector('.look-image-caption');
      const resetPhoto = () => {
        mainImage.src = data.image; mainImage.alt = data.title;
        mainImage.classList.remove('showing-product'); backPhoto.hidden = true;
        caption.textContent = `코디 ${String(index + 1).padStart(2, '0')}`;
        body.querySelectorAll('[data-preview-product]').forEach(button => button.setAttribute('aria-pressed', 'false'));
      };
      backPhoto.addEventListener('click', () => {
        const selected = body.querySelector('[data-preview-product][aria-pressed="true"]');
        resetPhoto(); selected?.focus({preventScroll:true});
      });
      body.querySelectorAll('[data-preview-product]').forEach(button => button.addEventListener('click', () => {
        if (button.getAttribute('aria-pressed') === 'true') { resetPhoto(); return; }
        const p = alternatives.find(item => item.id === button.dataset.previewProduct);
        body.querySelectorAll('[data-preview-product]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
        mainImage.src = p.image; mainImage.alt = p.name; mainImage.classList.add('showing-product');
        caption.textContent = p.name; backPhoto.hidden = false;
        body.scrollTo({top:0, behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
      }));
      body.querySelector('[data-look-another]')?.addEventListener('click', () => load((index + 1) % looks.length));
      if (!viewed.has(look.id)) {
        viewed.add(look.id);
        // A real displayed outfit view; never create product clicks/purchases.
        post(closetUrl('events', {}, uid), {event_type:'outfit_view', outfit_id:look.id, product_id:product.id})
          .catch(() => { if (!cleaned) toast('코디는 표시했지만 조회 기록을 저장하지 못했어요.'); });
      }
    } catch (error) {
      if (cleaned || current !== request) return;
      body.innerHTML = `<div class="look-loading"><p role="alert">코디를 불러오지 못했어요.</p><p class="caption">${esc(error.message)}</p><button type="button" class="text-btn" data-look-retry>다시 시도</button></div>`;
      body.querySelector('[data-look-retry]').addEventListener('click', () => load(index));
    } finally {
      if (current === request) body.setAttribute('aria-busy', 'false');
    }
  }
  dialog.showModal();
  document.body.style.overflow = 'hidden';
  load(index);
}

async function renderClosetDetail(oid, params) {
  const uid = state.userId;
  const pid = params.get('product_id');
  const mode = params.get('mode') || 'curated';
  const page = newPage('<p class="caption">코디와 함께 입을 상품을 불러오고 있어요.</p>');
  try {
    const [data, catalog] = await Promise.all([api(closetUrl('outfit', {product_id: pid, outfit_id: oid, mode}, uid)), closetCatalog()]);
    if (!page.isConnected || state.userId !== uid) return;
    const anchor = data.groups.find(g => g.is_anchor);
    const others = data.groups.filter(g => !g.is_anchor && (!data.curated || g.products.length || g.owned_count));
    if (data.shoe_alternatives?.length || data.shoe_alternatives_owned_count) {
      others.push({label:'비슷한 신발', products:data.shoe_alternatives || [], owned_count:data.shoe_alternatives_owned_count || 0});
    }
    const fromDiscovery = params.get('from') === 'discover';
    const back = fromDiscovery ? '#/discover' : params.get('from') === 'mypage' ? '#/mypage' : closetOutfitLink(pid, mode);
    page.innerHTML = `<a class="link-btn" href="${back}">← ${fromDiscovery ? '코디 둘러보기' : params.get('from') === 'mypage' ? '내 옷장' : '코디 목록'}</a><div class="closet-heading"><div><p class="eyebrow">COMPLETE THE LOOK</p><h1>${esc(data.title)}</h1><p>사진 속 다른 옷과 비슷한 상품을 찾아보세요.</p></div></div>
      <div class="closet-detail"><div><div class="closet-look-photo"><img src="${esc(data.image)}" alt="${esc(data.title)}"><div data-slot="region"></div></div>
      <label class="closet-box-toggle"><input type="checkbox" id="closet-show-region"> ${fromDiscovery ? '참고 상품과 비교한 부위 보기' : '내 옷과 매칭한 부위 보기'}</label></div>
      <div><div class="closet-owned-note"><span class="eyebrow">${fromDiscovery ? '함께 참고할 상품' : '선택한 내 옷'}</span><h2>${esc(closetName(data.product, catalog))}</h2>${fromDiscovery ? `${closetPhoto(data.product.image, closetName(data.product, catalog))}<a class="link-btn" data-closet-product href="#/product/${encodeURIComponent(pid)}">상품 살펴보기</a>` : '<p class="caption">옷장에 있는 상품은 추천 목록에서 제외해요.</p>'}</div>
      <h2>함께 입을 유사 상품</h2><p class="caption">사진 속 실제 착용 상품이 아닙니다. 색·소재·핏에 차이가 있을 수 있어요.</p>
      ${others.map(g => `<section class="closet-group"><h3>${esc(g.label)}</h3>${g.owned_count ? `<p class="caption">이미 옷장에 있는 상품 ${g.owned_count}개는 제외했어요.</p>` : ''}<div class="closet-similar-grid">${g.products.map(p => `<article class="closet-card">${closetPhoto(p.image, p.name)}<div class="closet-card-body"><h3>${esc(p.name)}</h3><a class="text-btn closet-cta" data-closet-product href="#/product/${encodeURIComponent(p.id)}">상품 상세</a><button class="link-btn" type="button" data-closet-add="${esc(p.id)}">내 옷장에 추가</button></div></article>`).join('') || '<p class="caption">연결할 유사 상품이 없어요.</p>'}</div></section>`).join('') || '<p class="caption">추가로 연결할 의류 영역이 없어요.</p>'}</div></div>`;
    const saveActions = document.createElement('div');
    saveActions.className = 'look-dialog-actions';
    saveActions.innerHTML = '<button type="button" class="look-save"></button><span class="saved-status" role="status"></span>';
    page.querySelector('.closet-heading').append(saveActions);
    bindOutfitSave(saveActions.querySelector('button'), {uid, oid, pid, saved:data.saved});
    page.querySelector('#closet-show-region').addEventListener('change', e => {
      if (!anchor || !e.target.checked) { slot(page, 'region').innerHTML = ''; return; }
      const [x1,y1,x2,y2] = anchor.box;
      const [w,h] = data.size;
      slot(page, 'region').innerHTML = `<div class="closet-region" style="left:${100*x1/w}%;top:${100*y1/h}%;width:${100*(x2-x1)/w}%;height:${100*(y2-y1)/h}%"><span>내 옷과 매칭한 ${esc(anchor.label)}</span></div>`;
    });
  } catch (error) { closetError(page, error); }
}

document.addEventListener('click', async e => {
  const outfit = e.target.closest('[data-closet-outfit]');
  if (outfit && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey && e.button === 0) {
    e.preventDefault();
    const uid = state.userId, origin = location.hash, target = outfit.getAttribute('href');
    try {
      await post(closetUrl('events', {}, uid), {event_type:'outfit_view', outfit_id:outfit.dataset.closetOutfit, product_id:outfit.dataset.outfitProduct});
    } catch { toast('조회 기록을 저장하지 못했어요. 코디는 계속 볼 수 있습니다.'); }
    if (state.userId === uid && location.hash === origin) location.hash = target;
    return;
  }
  if (e.target.closest('[data-closet-product]')) closetReturnHash = location.hash;
  const button = e.target.closest('[data-closet-add], [data-closet-remove], [data-closet-retry]');
  if (!button) return;
  if (button.hasAttribute('data-closet-retry')) { route(); return; }
  const uid = state.userId;
  const hash = location.hash;
  const add = button.hasAttribute('data-closet-add');
  button.disabled = true;
  try {
    await closetUpdate(add ? 'add' : 'remove', [add ? button.dataset.closetAdd : button.dataset.closetRemove], uid);
    if (state.userId !== uid || location.hash !== hash) return;
    toast(add ? '내 옷장에 추가했어요.' : '옷장에서 뺐어요.');
    if (hash.startsWith('#/mypage') || hash.startsWith('#/closet/')) route();
    else button.textContent = '옷장에 추가됨';
  } catch (error) { toast(error.message); button.disabled = false; }
});
