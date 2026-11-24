/* Explicit outfit choices are separate from passive views and owned clothing. */
async function renderFirstVisit() {
  const uid = state.userId;
  const page = newPage('<p class="caption">취향을 불러오는 중입니다.</p>');
  try {
    const data = await api(closetUrl('survey', {}, uid));
    if (!page.isConnected || state.userId !== uid) return;
    return data.survey.completed ? renderDiscovery() : renderMoodboard(data);
  } catch (error) { closetError(page, error); }
}

async function renderMoodboard(loaded) {
  const uid = state.userId;
  const background = await renderDiscovery();
  if (!background.isConnected || state.userId !== uid) return;
  const page = document.createElement('dialog');
  page.className = 'moodboard';
  page.setAttribute('aria-label', '코디 취향 선택');
  $app.append(page);
  try {
    const data = loaded || await api(closetUrl('survey', {}, uid));
    if (!page.isConnected || state.userId !== uid) return;
    const lookup = new Map(data.outfits.map(o => [o.id, o]));
    const draftKey = `closet-moodboard-v1:${uid}`;
    let draft;
    try { draft = JSON.parse(sessionStorage.getItem(draftKey)); } catch {}
    const picks = new Set((Array.isArray(draft?.selected) ? draft.selected : data.survey.selected).filter(id => lookup.has(id)).slice(0, data.max_picks));
    let order = Array.isArray(draft?.order) ? [...new Set(draft.order)].filter(id => lookup.has(id)) : [];
    if (!order.length) {
      order = [...lookup.keys()];
      // A fresh order for each survey draft; no permanent first-card advantage.
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(crypto.getRandomValues(new Uint32Array(1))[0] / 4294967296 * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    for (const id of lookup.keys()) if (!order.includes(id)) order.push(id);
    let visible = Math.max(25, ...[...picks].map(id => order.indexOf(id) + 1));
    let busy = false;
    page.innerHTML = `<div class="moodboard-top">${closetModelLabel(data.publisher_model_filter)}<label class="moodboard-audience">의류 범위 <select data-mood-audience>${closetAudienceOptions(data.audience)}</select></label><button type="button" class="moodboard-close" data-mood-close aria-label="취향 선택 닫기"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg></button></div><div class="moodboard-scroll"><div class="moodboard-wall" aria-label="취향을 고를 코디 사진"></div>
      <div class="moodboard-more"><button type="button" data-mood-more>다른 코디 더 보기 <span aria-hidden="true">↓</span></button></div></div>
      <div class="moodboard-dock"><div class="moodboard-dock-inner"><div class="moodboard-count" role="status" aria-live="polite"><strong data-mood-count>0</strong><span>개 선택</span></div><div class="moodboard-selected" aria-label="선택한 코디"></div><div class="moodboard-actions"><button type="button" class="moodboard-skip" data-mood-skip>건너뛰기</button><button type="button" class="moodboard-done" data-mood-done>계속 <span aria-hidden="true">→</span></button></div></div><p class="moodboard-status" role="status" data-mood-status></p></div>`;
    const wall = page.querySelector('.moodboard-wall');
    const status = page.querySelector('[data-mood-status]');
    const audienceSelect = page.querySelector('[data-mood-audience]');
    audienceSelect.addEventListener('change', async () => {
      if (busy) return;
      busy = true;
      audienceSelect.disabled = true;
      try {
        await closetUpdate('audience', [audienceSelect.value], uid);
        if (page.isConnected && state.userId === uid) renderMoodboard();
      } catch (error) { busy = false; audienceSelect.disabled = false; status.textContent = error.message; }
    });
    function saveDraft() {
      try { sessionStorage.setItem(draftKey, JSON.stringify({selected:[...picks], order})); } catch {}
    }
    function drawWall() {
      wall.innerHTML = order.slice(0, visible).map((id, index) => {
        const item = lookup.get(id);
        return `<button type="button" class="moodboard-photo" data-mood-pick="${esc(id)}" aria-label="코디 ${index + 1} 선택" aria-pressed="${picks.has(id)}"><img src="${esc(item.image)}" alt="${esc(item.title)}" loading="${index < 10 ? 'eager' : 'lazy'}" draggable="false"><span class="moodboard-check" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="m5 12 4 4 10-10"/></svg></span><span class="moodboard-unavailable" hidden>사진을 불러올 수 없어요</span></button>`;
      }).join('');
      if (!order.length) wall.innerHTML = '<p class="caption">이 범위의 코디는 아직 준비 중이에요.</p>';
      wall.querySelectorAll('img').forEach(img => img.addEventListener('error', () => {
        const button = img.closest('button');
        button.disabled = true;
        button.dataset.unavailable = 'true';
        button.querySelector('.moodboard-unavailable').hidden = false;
        img.style.visibility = 'hidden';
        picks.delete(button.dataset.moodPick);
        updateSelection();
      }));
      page.querySelector('[data-mood-more]').hidden = visible >= order.length;
    }
    function updateSelection() {
      page.querySelector('[data-mood-count]').textContent = picks.size;
      page.querySelector('.moodboard-count').hidden = picks.size === 0;
      wall.querySelectorAll('[data-mood-pick]').forEach(button => button.setAttribute('aria-pressed', String(picks.has(button.dataset.moodPick))));
      page.querySelector('.moodboard-selected').innerHTML = [...picks].map(id => `<button type="button" data-mood-remove="${esc(id)}" aria-label="선택한 코디 ${order.indexOf(id) + 1} 취소"><img src="${esc(lookup.get(id).image)}" alt=""><span aria-hidden="true">×</span></button>`).join('');
      page.querySelector('[data-mood-done]').disabled = busy;
      saveDraft();
    }
    page.addEventListener('click', async e => {
      if (busy) return;
      const pick = e.target.closest('[data-mood-pick]');
      const remove = e.target.closest('[data-mood-remove]');
      if (pick || remove) {
        const id = pick ? pick.dataset.moodPick : remove.dataset.moodRemove;
        if (picks.has(id)) picks.delete(id);
        else if (picks.size < data.max_picks) picks.add(id);
        else { status.textContent = '최대 5장까지 선택할 수 있어요. 선택한 사진을 누르면 해제됩니다.'; return; }
        status.textContent = '';
        updateSelection();
        if (remove) [...wall.querySelectorAll('[data-mood-pick]')].find(b => b.dataset.moodPick === id)?.focus({preventScroll:true});
      }
      if (e.target.closest('[data-mood-more]')) {
        visible += 25;
        drawWall();
      }
      const skip = e.target.closest('[data-mood-skip], [data-mood-close]');
      const done = e.target.closest('[data-mood-done]');
      if (!skip && !done) return;
      busy = true;
      audienceSelect.disabled = true;
      page.querySelectorAll('button').forEach(button => { button.disabled = true; });
      status.textContent = '저장하는 중입니다.';
      try {
        await post(closetUrl('survey', {}, uid), {selected:skip ? [] : [...picks], skipped:!!skip, clear:!skip && picks.size === 0});
        try { sessionStorage.removeItem(draftKey); } catch {}
        if (!page.isConnected || state.userId !== uid) return;
        state.onboarded = true;
        saveState();
        navigate('#/discover');
      } catch (error) {
        busy = false;
        audienceSelect.disabled = false;
        page.querySelectorAll('button').forEach(button => { button.disabled = button.dataset.unavailable === 'true'; });
        updateSelection();
        status.textContent = `저장하지 못했어요. ${error.message} 다시 눌러주세요.`;
      }
    });
    drawWall();
    updateSelection();
    page.addEventListener('cancel', e => {
      e.preventDefault();
      if (!busy) page.querySelector('[data-mood-close]').click();
    });
    page.showModal();
    page.querySelector('[data-mood-close]').focus({preventScroll:true});
  } catch (error) {
    page.remove();
    toast(error.message);
    navigate('#/discover');
  }
}
