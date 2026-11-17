/* First visit is browsing, not a mandatory survey. Reuse known shopping scope. */
async function renderDiscovery() {
  const uid = state.userId;
  const page = newPage('<p class="caption">코디를 불러오는 중입니다.</p>');
  try {
    const data = await api(closetUrl('discover', {}, uid));
    if (!page.isConnected || state.userId !== uid) return;
    state.audience = data.audience;
    state.onboarded = true;
    saveState();
    syncControls();
    page.innerHTML = `<div class="discovery-heading"><div><p class="eyebrow">DAILY EDIT</p><h1>오늘은 어떻게 입을까요?</h1><p class="caption">마음이 가는 코디부터 둘러보세요. 보는 동안 추천도 조금씩 맞춰집니다.</p></div><a class="discovery-closet" href="#/mypage">${data.owned_count ? '내 옷으로 코디 찾기' : '내 옷장 시작하기'} <span aria-hidden="true">↗</span></a></div>
      ${closetRegistrationForm(data)}<div class="discovery-toolbar">${closetModelLabel(data.publisher_model_filter)}<div class="discovery-filters" role="group" aria-label="쇼핑 대상">${['all','women','men','unisex'].map(value => `<button type="button" data-discovery-audience="${value}" aria-pressed="${data.audience === value}">${closetAudienceLabels[value]}</button>`).join('')}</div><span class="caption">${closetPreferenceText(data.personalization) || '여러 분위기의 코디를 모았어요'}</span><a class="link-btn" href="#/onboarding">취향 다시 고르기</a></div>
      <div class="discovery-grid">${data.outfits.map(o => `<article><a class="discovery-look" data-closet-outfit="${esc(o.id)}" data-outfit-product="${esc(o.product_id)}" href="#/closet/outfit/${encodeURIComponent(o.id)}?${new URLSearchParams({product_id:o.product_id, mode:'curated', from:'discover'})}" aria-label="${esc(o.title)} 자세히 보기">${closetPhoto(o.image,o.title)}<div class="discovery-look-caption"><h2>${esc(o.title)}</h2><span>${esc(o.style)}</span></div></a></article>`).join('') || '<div class="closet-empty"><h2>이 범위의 코디는 아직 준비 중이에요</h2><p>다른 의류 범위를 선택해 주세요.</p></div>'}</div>`;
    bindClosetRegistration(page, uid, renderDiscovery);
    page.querySelectorAll('[data-discovery-audience]').forEach(button => button.addEventListener('click', async () => {
      const value = button.dataset.discoveryAudience;
      page.querySelectorAll('[data-discovery-audience]').forEach(el => { el.disabled = true; });
      try {
        await closetUpdate('audience', [value], uid);
        if (!page.isConnected || state.userId !== uid) return;
        state.audience = value;
        saveState();
        syncControls();
        renderDiscovery();
      } catch (error) {
        toast(error.message);
        page.querySelectorAll('[data-discovery-audience]').forEach(el => { el.disabled = false; });
      }
    }));
  } catch (error) { closetError(page, error); }
  return page;
}
