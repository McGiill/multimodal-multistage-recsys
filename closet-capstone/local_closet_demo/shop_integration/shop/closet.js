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
      <div class="closet-heading mypage-heading"><h1>마이페이지</h1><a class="mypage-shop-link" href="#/discover">코디 둘러보기</a></div>
      ${closetRegistrationForm(profile)}
      <div class="mypage-shell">
      <div class="mypage-content">
      <section data-mypage-panel="wardrobe" aria-labelledby="mypage-wardrobe-title">
      <div class="mypage-section-heading"><div><h2 id="mypage-wardrobe-title">내 옷장</h2></div><div class="wardrobe-scope">${closetModelLabel(profile.publisher_model_filter)}<label for="wardrobe-audience">의류 범위</label><select id="wardrobe-audience">${closetAudienceOptions(profile.audience)}</select><button class="mypage-add-button" type="button" id="mypage-add">옷 추가</button></div></div>
      <div class="wardrobe-categories" role="group" aria-label="옷 종류">${Object.entries({all:'전체',...closetSlots}).map(([key,label]) => `<button type="button" data-wardrobe-category="${key}" aria-pressed="false" aria-controls="wardrobe-items" aria-label="${label}">${label}<span>${profile.products.filter(p => key === 'all' || p.slot === key).length}</span></button>`).join('')}</div>
      <div class="closet-toolbar mypage-filter"><span class="caption" data-slot="visible-count" role="status">전체 ${profile.products.length}개</span><div class="mypage-tools"><button type="button" class="link-btn" id="mypage-edit" aria-pressed="false">편집</button></div></div>
      <div class="wardrobe-rack-tools"><button type="button" class="link-btn" data-rack-expand aria-expanded="false">전체 보기</button><button type="button" class="link-btn" data-rack-shift="-1" aria-label="이전 옷">←</button><button type="button" class="link-btn" data-rack-shift="1" aria-label="다음 옷">→</button></div>
      <div class="closet-grid wardrobe-rack" id="wardrobe-items" data-slot="owned" aria-label="보유한 옷"></div>
      <details class="closet-add-panel" ${profile.products.length ? '' : 'open'}><summary>옷장에 상품 추가</summary><p class="caption">기존 상품 목록에서 보유한 옷을 선택하세요. 관심 표시와 보유 여부는 따로 저장해요.</p>
      <form id="closet-add-form"><label for="closet-add-select">추가할 상품</label><select id="closet-add-select">${catalog.products.filter(p => !profile.owned.includes(p.id)).map(p => `<option value="${esc(p.id)}">${esc(p.display_name)} · ${esc(p.color || p.category)}</option>`).join('')}</select><button class="primary" ${catalog.products.every(p => profile.owned.includes(p.id)) ? 'disabled' : ''}>옷장에 추가</button></form>
      <button class="link-btn" type="button" id="closet-demo">시연 상품 ${catalog.demo_ids.length}개 담기</button> <button class="link-btn" type="button" data-closet-shoe-demo ${catalog.shoe_demo_ids.length ? '' : 'hidden'}>신발 시연 상품 ${catalog.shoe_demo_ids.length}개 담기</button><button class="link-btn" type="button" data-closet-dress-demo ${catalog.dress_demo_ids.length ? '' : 'hidden'}>원피스 시연 상품 ${catalog.dress_demo_ids.length}개 담기</button><p class="caption">기존 옷장에 시연 상품을 추가합니다. 실제 구매 이력이 아닙니다.</p></details>
      </section>
      </div></div><p class="caption closet-storage">옷장과 코디 조회 기록은 이 PC에 저장됩니다. 회원 선택은 로컬 시연용입니다.</p>`;
    bindClosetRegistration(page, uid, renderCloset);
    page.querySelector('#mypage-add').addEventListener('click', () => {
      const panel = page.querySelector('.closet-add-panel');
      panel.open = true;
      panel.scrollIntoView({block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
      page.querySelector('#closet-add-select').focus({preventScroll:true});
    });
    let editing = false;
    let selectedId = profile.owned.includes(closetSelections.get(uid)) ? closetSelections.get(uid) : profile.products.find(p => p.can_recommend)?.id;
    let outfitRequest = 0;
    const outfitPanel = document.createElement('section');
    outfitPanel.id = 'mypage-outfits';
    outfitPanel.className = 'mypage-inline-outfits';
    outfitPanel.setAttribute('aria-label', '선택한 옷의 코디');
    function placeOutfits() {
      const grid = slot(page, 'owned');
      const cards = [...grid.children].filter(el => el.classList.contains('closet-card'));
      const index = cards.findIndex(el => el.querySelector('[data-closet-select]')?.dataset.closetSelect === selectedId);
      if (index < 0) { outfitPanel.remove(); return; }
      grid.after(outfitPanel);
    }
    async function selectOutfits(pid, scroll = false, limit = 5) {
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
        outfitPanel.innerHTML = `<div class="mypage-section-heading"><div><h2>${esc(closetName(data.product, catalog))} 코디</h2><p class="caption">${esc(closetAudienceLabels[data.audience])} · ${data.outfits.length}${data.has_more ? ' / ' + data.total_count : ''}개${data.outfits.length && data.personalization?.ready ? ' · ' + closetPreferenceText(data.personalization) : ''}</p></div><button type="button" class="link-btn" data-outfits-close>접기</button></div>
          <div class="closet-grid outfit-grid">${data.outfits.map(o => `<article class="closet-card"><a class="mypage-outfit-select" data-closet-outfit="${esc(o.id)}" data-outfit-product="${esc(pid)}" href="#/closet/outfit/${encodeURIComponent(o.id)}?${new URLSearchParams({product_id:pid, mode:'curated', from:'mypage'})}" aria-label="${esc(o.title)} 자세히 보기">${closetPhoto(o.image, o.title, 'outfit-photo')}<div class="closet-card-body"><h3>${esc(o.title)}</h3><p class="caption">${esc([o.style_tag, {menswear:'남성복', womenswear:'여성복', unisex:'공용'}[o.clothing_audience]].filter(Boolean).join(' · '))}</p></div></a></article>`).join('') || `<p class="wardrobe-no-looks">현재 모델·의류 범위에 맞는 코디는 아직 없어요. 다른 옷을 골라보세요.</p>`}</div>`;
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
      slot(page, 'owned').scrollLeft = 0;
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
    page.querySelector('#wardrobe-audience').addEventListener('change', async e => {
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
