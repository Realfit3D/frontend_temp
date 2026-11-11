/* library.js — 影片資料庫的查詢邏輯（不碰畫面，可單獨測試）
 *
 * 讀取兩個檔案（都在同一個資料夾 base 底下）：
 *   manifest.json  有哪些影片（由 build_index.py 產生）
 *   catalog.json   衣服、動作、身高體重區間的顯示名稱
 *
 * 使用方式：
 *   const lib = await Library.load('library/');
 *   lib.genders(sel) / lib.heights(sel) / lib.weights(sel)
 *   lib.uppers(sel)  / lib.lowers(sel)  / lib.motions(sel)   → [{ id, label, disabled, ... }]
 *   lib.opacities(sel)                                        → [{ id, value, label, disabled }]（沒有透明影片時是空陣列）
 *   lib.videoUrl(sel)                                         → 影片網址或 null
 *   lib.normalize(sel)                                        → 清掉目前已不可選的欄位
 *
 * sel = { gender, height, weight, upper, lower, motion, opacity }  （沒選的欄位是 null 或 ''）
 *   opacity 是不透明度的字串（例如 '0.5'）；沒選或 null 一律當 1（不透明）。
 *   只採用 opacity_mode 為 v3 的透明影片；direct / optical_depth 是替代做法，不給使用者選。
 *   衣服、動作、身形的選項只由「不透明影片」決定，不受透明度影響；透明度的選項則要看其他欄位都選好後有沒有影片。
 *   height / weight 是「區間 id」（例如 '150-160'），影片實際用的是該區間的 value（中間值，例如 155）。
 *
 * 一個選項 disabled = 在「它所依賴的、已經選好的欄位」限制下，沒有任何一支影片符合（依賴關係見 DEPENDS_ON）。
 */
(function (root) {
  'use strict';

  const GENDERS = [{ id: 'female', label: '女性' }, { id: 'male', label: '男性' }];

  function rangesFrom(catalogList, values, unit) {
    // catalog 有定義區間就用它；沒有的話，用影片實際有的數值自己產生一個 id=數值 的清單
    if (Array.isArray(catalogList) && catalogList.length) {
      return catalogList.map((r) => ({ id: String(r.id), label: r.label || String(r.id), value: r.value }));
    }
    return [...new Set(values)].sort((a, b) => a - b).map((v) => ({ id: String(v), label: v + ' ' + unit, value: v }));
  }

  function build(manifest, catalog, base) {
    base = base || '';
    catalog = catalog || {};
    const garments = catalog.garments || {};
    const motionsCat = catalog.motions || {};

    // ---- 「身形不適合此衣服」規則（catalog.fit_rules）----
    //   [{ garment, reason, bodies: [{ gender, height, weight? }] }]；height / weight 是區間的中間值（例如 155、45），
    //   weight 省略 = 該身高的所有體重。符合規則的影片不會出現在選項裡，衣服卡片會改顯示原因。
    const fitRules = Array.isArray(catalog.fit_rules) ? catalog.fit_rules : [];
    function unfitReason(garment, gender, height, weight) {
      for (const r of fitRules) {
        if (r.garment !== garment) continue;
        const hit = (r.bodies || []).some((b) =>
          (b.gender == null || b.gender === gender) &&
          (b.height == null || b.height === height) &&
          (b.weight == null || b.weight === weight));
        if (hit) return r.reason || '此身形不適合';
      }
      return null;
    }

    // ---- 把 manifest 的每支影片整理成 { url, gender, height, weight, motion, upper, lower } ----
    const videos = [];
    const allVideos = [];
    (manifest.videos || []).forEach((v) => {
      let upper = null, lower = null;
      (v.garments || []).forEach((g, i) => {
        const slot = (garments[g] && garments[g].slot) || (i === 0 ? 'lower' : 'upper'); // catalog 沒寫就依順序：下身在前
        if (slot === 'upper') upper = g; else if (slot === 'lower') lower = g;
      });
      if (!upper || !lower) return;                      // 不是「一件上身＋一件下身」就略過
      allVideos.push({ upper, lower });                  // 過濾前出現過的衣服（讓被規則排除的衣服仍列在畫面上）
      if (unfitReason(upper, v.gender, v.height, v.weight) || unfitReason(lower, v.gender, v.height, v.weight)) return;  // 身形不適合
      const opacity = v.opacity == null ? 1 : Number(v.opacity);   // 舊影片沒有這個欄位 = 不透明
      const mode = v.opacity_mode || 'direct';
      if (!(opacity >= 1) && mode !== 'v3') return;      // 透明影片只採用 v3
      // make_site.py 會在上線用的 manifest 裡給每支影片一個內容雜湊 v；網址加上 ?v= 後，換過的影片不會被手機的舊快取擋住
      videos.push({ url: base + v.path + (v.v ? '?v=' + encodeURIComponent(v.v) : ''), gender: v.gender, height: v.height, weight: v.weight, motion: v.motion, upper, lower,
                    opacity, mode, targets: v.opacity_garments || [] });
    });

    const opaqueVideos = videos.filter((v) => v.opacity >= 1);       // 其他選項只由不透明影片決定
    const heightRanges = rangesFrom(catalog.heights, opaqueVideos.map((v) => v.height), 'cm');
    const weightRanges = rangesFrom(catalog.weights, opaqueVideos.map((v) => v.weight), 'kg');
    const valueOf = (ranges, id) => { const r = ranges.find((x) => x.id === String(id)); return r ? r.value : undefined; };

    // ---- 篩選：sel 裡有填的欄位都要符合（ignore 指定的欄位不參與篩選）----
    function matching(sel, ignore) {
      const hv = sel.height ? valueOf(heightRanges, sel.height) : undefined;
      const wv = sel.weight ? valueOf(weightRanges, sel.weight) : undefined;
      const ov = sel.opacity ? Number(sel.opacity) : 1;     // 沒選透明度 = 不透明
      return videos.filter((v) =>
        (ignore === 'opacity' || v.opacity === ov) &&
        (ignore === 'gender' || !sel.gender || v.gender === sel.gender) &&
        (ignore === 'height' || !sel.height || v.height === hv) &&
        (ignore === 'weight' || !sel.weight || v.weight === wv) &&
        (ignore === 'upper' || !sel.upper || v.upper === sel.upper) &&
        (ignore === 'lower' || !sel.lower || v.lower === sel.lower) &&
        (ignore === 'motion' || !sel.motion || v.motion === sel.motion));
    }
    // 每個欄位的選項只受「它依賴的欄位」限制，避免出現「換不動」的情況：
    //   身形依頁面順序：性別（只要有該性別的影片就能選）→ 身高（受性別限制）→ 體重（受性別與身高限制），
    //   所以切換性別時，不合的身高體重會被清掉，不會出現「改不回去」；
    //   衣服受身形與另一件衣服限制，但不受動作限制
    //   （換衣服時動作會自動換成可用的）；動作受全部限制。
    const DEPENDS_ON = {
      gender: [],
      height: ['gender'],
      weight: ['gender', 'height'],
      upper: ['gender', 'height', 'weight', 'lower'],
      lower: ['gender', 'height', 'weight', 'upper'],
      motion: ['gender', 'height', 'weight', 'upper', 'lower'],
      opacity: ['gender', 'height', 'weight', 'upper', 'lower', 'motion']
    };
    const has = (sel, field, pred) => {
      const scoped = {};
      DEPENDS_ON[field].forEach((k) => { scoped[k] = sel[k]; });
      return matching(scoped, field).some(pred);
    };

    function garmentItems(slot, field, sel) {
      const ids = Object.keys(garments).filter((g) => garments[g].slot === slot);
      const used = new Set(opaqueVideos.map((v) => v[field]));
      const bodyKnown = sel.gender && sel.height && sel.weight;
      const hv = bodyKnown ? valueOf(heightRanges, sel.height) : undefined;
      const wv = bodyKnown ? valueOf(weightRanges, sel.weight) : undefined;
      // 衣服只要對「某些身形」有影片就會列出；若這個身形被規則排除，就標成 unfit 並附原因（不可選）。
      const everUsed = new Set(allVideos.map((v) => v[field]));
      return ids.filter((g) => used.has(g) || everUsed.has(g)).map((g) => {
        const reason = bodyKnown ? unfitReason(g, sel.gender, hv, wv) : null;
        return {
          id: g, label: garments[g].label || g, thumb: base + 'thumbnails/' + g + '.png',
          disabled: !!reason || !has(sel, field, (v) => v[field] === g),
          unfit: !!reason, reason: reason || ''
        };
      });
    }

    const lib = {
      base, videos, heightRanges, weightRanges,

      genders(sel) {
        return GENDERS.map((g) => ({ id: g.id, label: g.label, disabled: !has(sel, 'gender', (v) => v.gender === g.id) }));
      },
      heights(sel) {
        const used = new Set(opaqueVideos.map((v) => v.height));
        return heightRanges.filter((r) => used.has(r.value)).map((r) => ({
          id: r.id, label: r.label, disabled: !has(sel, 'height', (v) => v.height === r.value) }));
      },
      weights(sel) {
        const used = new Set(opaqueVideos.map((v) => v.weight));
        return weightRanges.filter((r) => used.has(r.value)).map((r) => ({
          id: r.id, label: r.label, disabled: !has(sel, 'weight', (v) => v.weight === r.value) }));
      },
      uppers(sel) { return garmentItems('upper', 'upper', sel); },
      lowers(sel) { return garmentItems('lower', 'lower', sel); },
      motions(sel) {
        const used = new Set(opaqueVideos.map((v) => v.motion));
        const ids = Object.keys(motionsCat).filter((m) => used.has(m));
        [...used].forEach((m) => { if (!ids.includes(m)) ids.push(m); });   // catalog 沒列的動作也要能用（顯示代號）
        return ids.map((m) => ({
          id: m, label: (motionsCat[m] && motionsCat[m].label) || m, disabled: !has(sel, 'motion', (v) => v.motion === m) }));
      },

      /** 不透明度的刻度（由大到小，1 = 不透明）。沒有任何透明影片時回傳空陣列，畫面就不顯示拉桿。 */
      opacities(sel) {
        const values = [...new Set(videos.map((v) => v.opacity))];
        if (!values.some((x) => x < 1)) return [];
        const fromCatalog = (catalog.opacities || []);
        return values.sort((a, b) => b - a).map((val) => {
          const c = fromCatalog.find((x) => Number(x.value) === val);
          return {
            id: String(val), value: val,
            label: (c && c.label) || (Math.round(val * 100) + '%'),
            disabled: !has(sel, 'opacity', (v) => v.opacity === val)
          };
        });
      },

      labelOf(kind, id) {
        const list = { height: heightRanges, weight: weightRanges }[kind];
        if (list) { const r = list.find((x) => x.id === String(id)); return r ? r.label : String(id); }
        if (kind === 'gender') { const g = GENDERS.find((x) => x.id === id); return g ? g.label : String(id); }
        if (kind === 'opacity') return Math.round(Number(id) * 100) + '%';
        if (kind === 'motion') return (motionsCat[id] && motionsCat[id].label) || String(id);
        return (garments[id] && garments[id].label) || String(id);
      },

      /** 選好之後的影片網址；欄位沒選齊或沒有這支影片就回傳 null。 */
      videoUrl(sel) {
        if (!(sel.gender && sel.height && sel.weight && sel.upper && sel.lower && sel.motion)) return null;
        const m = matching(sel);
        if (!m.length) return null;
        m.sort((a, b) => (a.mode === 'direct' ? 0 : 1) - (b.mode === 'direct' ? 0 : 1) || a.targets.length - b.targets.length);
        return m[0].url;
      },

      /** 清掉已經不可選的欄位（回傳新物件，不改原本的）。
       *  依優先順序逐項檢查：性別 > 身高 > 體重 > 上身 > 下身 > 動作。
       *  每一項只對照「優先順序比它高、而且已確認有效」的欄位，所以衝突時留下先選的、清掉後面的。 */
      normalize(sel) {
        const order = [['gender', 'genders'], ['height', 'heights'], ['weight', 'weights'],
                       ['upper', 'uppers'], ['lower', 'lowers'], ['motion', 'motions'], ['opacity', 'opacities']];
        const out = { gender: null, height: null, weight: null, upper: null, lower: null, motion: null, opacity: null };
        order.forEach(([k, list]) => {
          if (!sel[k]) return;
          const item = lib[list](out).find((x) => x.id === sel[k]);
          if (item && !item.disabled) out[k] = sel[k];
        });
        return out;
      },

      /** 目前條件下第一個可選的動作 id（沒有就 null）。 */
      firstMotion(sel) {
        const m = lib.motions(sel).find((x) => !x.disabled);
        return m ? m.id : null;
      }
    };
    return lib;
  }

  async function fetchJson(url) {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
    return r.json();
  }

  const Library = {
    build,
    async load(base) {
      base = base || '';
      const [manifest, catalog] = await Promise.all([fetchJson(base + 'manifest.json'), fetchJson(base + 'catalog.json')]);
      return build(manifest, catalog, base);
    }
  };

  root.Library = Library;
  if (typeof module !== 'undefined' && module.exports) module.exports = Library;
})(typeof window !== 'undefined' ? window : globalThis);
