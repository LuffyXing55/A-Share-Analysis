/* A股30强看好度 —— 前端。数据来源：本地服务 /api/live（盘中实时）或静态 data/dashboard.json（GitHub Pages）。
   布局：左边榜单、右边所选股票的详情（K 线 + 看好分 K 线 + 原因 / 海外 / 新闻），一屏内完成；窄屏时详情全屏覆盖。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const state = {
    dash: null, chart: null, bt: null, btTech: null, live: false, timer: null,
    list: "main", sort: { main: "total", tech: "total" }, sel: { main: null, tech: null },
    range: 66, showParts: false, dtab: "reasons", more: "watch",
  };

  const fmtPct = (x, d = 2) => (x === null || x === undefined || isNaN(x)) ? "—" : (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
  const fmtNum = (x, d = 2) => (x === null || x === undefined || isNaN(x)) ? "—" : Number(x).toFixed(d);
  const cls = (x) => (x > 0 ? "up" : x < 0 ? "down" : "flat");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const isNum = (x) => typeof x === "number" && isFinite(x);
  const narrow = () => window.matchMedia("(max-width: 900px)").matches;

  async function getJSON(url) {
    const r = await fetch(url + (url.includes("?") ? "&" : "?") + "t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error(url + " " + r.status);
    return r.json();
  }

  async function load() {
    try {
      state.dash = await getJSON("/api/live");
      state.live = true;
    } catch (e) {
      state.dash = await getJSON("data/dashboard.json");
      state.live = false;
    }
    const base = state.live ? "/data/" : "data/";
    const [chart, bt, btTech] = await Promise.all(["chart.json", "backtest.json", "backtest_tech.json"].map(
      (f) => getJSON(base + f).catch(() => null)));
    state.chart = chart; state.bt = bt; state.btTech = btTech;
    render();
    if (state.live && !state.timer) state.timer = setInterval(async () => {
      try {
        state.dash = await getJSON("/api/live");
        if (!state.chart || (state.chart.data_date || "") < (state.dash.data_date || "")) {
          state.chart = await getJSON("/data/chart.json").catch(() => state.chart);
        }
        render();
      } catch (e) { /* 保留上一轮 */ }
    }, 30000);
  }

  // ------------------------------------------------------------------ 总渲染
  function render() {
    const d = state.dash;
    $("#subline").textContent = `数据日期 ${d.data_date} · 更新于 ${d.generated_at} · 本月组合 ${d.month}（${d.pick_date} 收盘后选出）` +
      (d.picks_note && /不完整|补选/.test(d.picks_note) ? ` · ⚠ ${d.picks_note}` : "");
    const mb = $("#mode-badge");
    mb.textContent = state.live ? "● 盘中实时（本地）" : "日更（GitHub）";
    mb.className = "mode" + (state.live ? " live" : "");
    renderMarket(d.market || {});
    const hasTech = !!(d.tech && d.tech.picks && d.tech.picks.length);
    $('#list-tabs [data-list="tech"]').classList.toggle("hidden", !hasTech);
    if (!hasTech && state.list === "tech") state.list = "main";
    renderList();
    renderDetail();
    renderWatch(d.watch || {});
    renderNews(d.news || []);
    renderModel();
    renderTechBT();
    fitHeader();
  }

  function fitHeader() {
    document.documentElement.style.setProperty("--hdr", $("#top").offsetHeight + "px");
  }

  function tile(n, v, c, digits, bp) {
    const chg = (bp !== undefined && bp !== null) ? `${bp > 0 ? "+" : ""}${Number(bp).toFixed(1)}bp` : fmtPct(c);
    const dir = (bp !== undefined && bp !== null) ? bp : c;
    return `<div class="tile"><span class="n">${esc(n)}</span><span class="v ${cls(dir)}">${fmtNum(v, digits ?? 2)}</span><span class="c ${cls(dir)}">${chg}</span></div>`;
  }

  function renderMarket(m) {
    $("#index-tiles").innerHTML = (m.indices || []).map((x) => tile(x.name, x.price, x.pct)).join("") +
      (m.breadth ? `<div class="tile"><span class="n">涨停/跌停/炸板</span><span class="v">${m.breadth.zt ?? "—"} / ${m.breadth.dt ?? "—"} / ${m.breadth.zb ?? "—"}</span></div>` : "");
    $("#global-tiles").innerHTML = (m.global || []).map((x) => tile(x.name, x.price, x.pct, x.price > 1000 ? 0 : 2, x.bp)).join("");
    const r = m.regime;
    $("#regime").innerHTML = (r ? `<b>市场状态：${esc(r.level)}</b> · ${esc(r.text)}` : "") + (m.time ? ` · 行情时间 ${esc(m.time)}` : "");
  }

  // ------------------------------------------------------------------ 左：榜单
  function curList() {
    const d = state.dash;
    return state.list === "tech" ? ((d.tech || {}).picks || []) : (d.picks || []);
  }

  function findPick(code, which) {
    const d = state.dash;
    const l = which === "tech" ? ((d.tech || {}).picks || []) : (d.picks || []);
    return l.find((x) => x.code === code);
  }

  function renderList() {
    const d = state.dash, which = state.list, key = state.sort[which];
    const list = curList();
    $$("#list-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.list === which));
    $$("#sort-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.sort === key));
    $('#list-tabs [data-list="main"]').textContent = `本月30强`;
    if (d.tech) $('#list-tabs [data-list="tech"]').textContent = `科技周榜 ${d.tech.week ? d.tech.week.slice(5) : ""}`;
    if (which === "tech") {
      const t = d.tech || {}, w = t.weights || { value: 0.5, short: 0.5 };
      $("#list-hint").textContent = `${t.pick_date || ""} 收盘后选出，每周最后一个交易日换一次；分数是科技股池（${t.universe_n ?? "—"} 只）内的分位，综合 = ${w.value}×价值 + ${w.short}×短线。` +
        (t.note && /不完整|补选/.test(t.note) ? ` ⚠ ${t.note}` : "");
    } else {
      const w = d.weights || { value: 0.6, short: 0.4 };
      $("#list-hint").textContent = `综合 = ${w.value}×价值 + ${w.short}×短线，分数是全市场约 4000 只股票中的分位（0–100）。点击股票查看 K 线、看好分走势与原因` +
        (state.live ? "；盘中每 1 分钟重算。" : "。");
    }
    const rows = [...list].sort((a, b) => (b[key] ?? -1) - (a[key] ?? -1));
    const tv = rows.map((p) => p.total).filter(isNum);
    const tmin = Math.min(...tv), tmax = Math.max(...tv);
    if (!state.sel[which] || !list.find((p) => p.code === state.sel[which])) state.sel[which] = rows.length ? rows[0].code : null;
    $("#list-body").innerHTML = rows.map((p, i) => {
      const ch = (p.prev_rank && p.rank) ? p.prev_rank - p.rank : null;
      const chTxt = ch === null ? '<span class="flat">新</span>' : ch > 0 ? `<span class="up">↑${ch}</span>` : ch < 0 ? `<span class="down">↓${-ch}</span>` : '<span class="flat">–</span>';
      return `<tr data-code="${esc(p.code)}" class="${p.code === state.sel[which] ? "sel" : ""}">
        <td>${key === "total" ? p.rank : i + 1}</td><td>${chTxt}</td>
        <td class="l"><b>${esc(p.name)}</b><span class="code">${esc(p.code.slice(2))}</span></td>
        <td>${fmtNum(p.total, 1)}<span class="bar" style="width:${tmax > tmin ? 3 + 27 * ((p.total ?? tmin) - tmin) / (tmax - tmin) : 15}px"></span></td>
        <td>${fmtNum(p.value, 1)}</td><td>${fmtNum(p.short, 1)}</td>
        <td class="${cls(p.pct_today)}">${fmtPct(p.pct_today)}</td></tr>`;
    }).join("");
    $$("#list-body tr").forEach((tr) => tr.addEventListener("click", () => {
      select(tr.dataset.code, true);
      if (!narrow()) $(".list-scroll").focus({ preventScroll: true });   // 之后可以用 ↑ ↓ 切换
    }));
  }

  function select(code, open) {
    state.sel[state.list] = code;
    $$("#list-body tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.code === code));
    // 窄屏：先显示详情层再画图，否则 K 线按 0 宽度绘制
    if (open && narrow()) { document.body.classList.add("detail-open"); $("#detail").scrollTop = 0; }
    renderDetail();
  }

  // ------------------------------------------------------------------ 右：详情
  function signalChips(p) {
    const sig = [];
    if (p.signals && Math.abs(p.signals.overseas || 0) >= 0.5) sig.push(`<span class="chip ${p.signals.overseas > 0 ? "pos" : "neg"}">海外${p.signals.overseas > 0 ? "利好" : "利空"} ${p.signals.overseas_points > 0 ? "+" : ""}${fmtNum(p.signals.overseas_points, 1)}分</span>`);
    if (p.signals && Math.abs(p.signals.news || 0) >= 0.2) sig.push(`<span class="chip ${p.signals.news > 0 ? "pos" : "neg"}">新闻${p.signals.news > 0 ? "偏正" : "偏负"}</span>`);
    if (p.is_st) sig.push('<span class="chip neg">ST</span>');
    if (p.suspended) sig.push('<span class="chip">停牌</span>');
    return sig.join("");
  }

  function renderDetail() {
    const which = state.list, code = state.sel[which];
    const p = code ? findPick(code, which) : null;
    if (!p) { $("#d-name").textContent = "—"; $("#kline").innerHTML = ""; return; }
    const src = which === "tech" ? state.dash.tech : state.dash;
    $("#d-name").innerHTML = `${esc(p.name)} <span class="code">${esc(p.code.slice(2))}</span> <span class="chip">${esc(p.industry || "")}</span>`;
    const pe = isNum(p.pe_ttm) ? (p.pe_ttm > 0 ? fmtNum(p.pe_ttm, 1) : "亏损") : "—";
    $("#d-meta").innerHTML = `现价 <b>${fmtNum(p.price, 2)}</b> <span class="${cls(p.pct_today)}">${fmtPct(p.pct_today)}</span> · ` +
      `入选以来 <span class="${cls(p.ret_since_pick)}">${fmtPct(p.ret_since_pick)}</span>（${esc(src.pick_date || "")} 选出）· ` +
      `PE(TTM) ${pe} · 市值 ${fmtNum(p.total_mv_yi, 0)} 亿 ${signalChips(p)}`;
    $("#d-scores").innerHTML = [["综合", p.total, "total"], ["价值", p.value, ""], ["短线", p.short, ""]]
      .map(([n, v, c]) => `<div class="sc ${c}"><div class="n">${n}</div><div class="v">${fmtNum(v, 1)}</div></div>`).join("");
    drawKline($("#kline"), buildSeries(p.code, which), { n: state.range, pickDate: src.pick_date, name: p.name, which });
    const scope = which === "tech" ? "科技股池" : "全市场股票池";
    $("#d-reasons").innerHTML = reasonsHTML(p.reasons_value, "价值投资分（中期，约 1 个月）的主要加分 / 减分项") +
      reasonsHTML(p.reasons_short, "短线分（约 1 周）的主要加分 / 减分项") +
      `<p class="hint">「前x%」表示该指标在全市场股票池中的位置（分数本身是${scope}内的分位）。条形长度为该项对打分的贡献。</p>`;
    const chains = (p.chains || []).map((c) => `<div class="card"><div class="t">${esc(c.name)} <span class="chip">${esc(c.strength || "")}</span></div>
      <div class="s">${esc(c.logic || "")}</div>
      <div class="s">海外标的：${(c.overseas || []).map((o) => `${esc(o.name || o.symbol)} ${o.suspect ? '<span class="flat">数据异常，已忽略</span>' : `<span class="${cls(o.pct)}">${fmtPct(o.pct)}</span>`}`).join("，") || "—"}</div>
      ${c.beta !== undefined && c.beta !== null ? `<div class="s">历史联动：海外涨 1% → 该股次日平均 ${fmtPct(c.beta / 100, 2)}（相关系数 ${fmtNum(c.corr, 2)}，${esc(c.sample || "")}）</div>` : ""}</div>`).join("");
    $("#d-chains").innerHTML = chains || '<p class="hint">该股没有明确的海外产业链映射。</p>';
    $("#d-news").innerHTML = (p.news || []).map(newsLi).join("") || '<li class="hint">近期没有匹配到相关新闻。</li>';
    showDTab(state.dtab);
  }

  function showDTab(t) {
    state.dtab = t;
    $$("#d-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
    ["reasons", "chains", "news"].forEach((k) => $("#d-" + k).classList.toggle("hidden", k !== t));
  }

  function reasonsHTML(list, title) {
    if (!list || !list.length) return "";
    const mx = Math.max(...list.map((r) => Math.abs(r.contrib)), 1e-6);
    return `<h3>${esc(title)}</h3>` + list.map((r) => {
      const w = Math.abs(r.contrib) / mx * 50;
      const left = r.contrib >= 0 ? 50 : 50 - w;
      return `<div class="reason"><span>${esc(r.name)}</span><div class="track"><div class="fill" style="left:${left}%;width:${w}%;background:${r.contrib >= 0 ? "var(--up)" : "var(--down)"}"></div></div><span class="${cls(r.contrib)}">${(r.pct !== undefined && r.pct !== null) ? "前" + Math.max(1, Math.round(100 - r.pct)) + "%" : ""}</span></div>`;
    }).join("");
  }

  // ------------------------------------------------------------------ K 线数据
  function buildSeries(code, which) {
    const ch = state.chart || {};
    const kl = ch.kline || {};
    const dates = [...(kl.dates || [])];
    const px = [...(((kl.codes || {})[code]) || dates.map(() => null))];
    const p = findPick(code, which) || {};
    const src = which === "tech" ? (state.dash.tech || {}) : state.dash;
    // chart.json 比看好分记录旧（例如页面跨日一直开着）：补上这些日期，只有看好分、没有 K 线
    const lastK = dates.length ? dates[dates.length - 1] : "";
    const extra = [...new Set(((src.history || {})[code] || []).map((x) => String(x[0]).slice(0, 10)))]
      .filter((dt) => dt > lastK && !(p.today_bar && p.today_bar.date === dt)).sort();
    extra.forEach((dt) => { dates.push(dt); px.push(null); });
    const tb = p.today_bar;
    if (tb && isNum(tb.c) && (!dates.length || tb.date > dates[dates.length - 1])) {   // 本地盘中：今天的 K 线
      let lc = null;
      for (let j = px.length - 1; j >= 0; j--) if (px[j]) { lc = px[j][3]; break; }
      if (isNum(tb.pc) && tb.pc > 0 && lc && Math.abs(tb.pc / lc - 1) > 1e-3) {
        const r = tb.pc / lc;                // 今天除权除息：历史前复权到今天的口径
        for (let j = 0; j < px.length; j++) if (px[j]) px[j] = [px[j][0] * r, px[j][1] * r, px[j][2] * r, px[j][3] * r, px[j][4]];
      }
      dates.push(tb.date);
      px.push([isNum(tb.o) && tb.o > 0 ? tb.o : tb.c, isNum(tb.h) && tb.h > 0 ? tb.h : tb.c, isNum(tb.l) && tb.l > 0 ? tb.l : tb.c, tb.c,
        isNum(tb.v) ? Math.round(tb.v) : null]);
    }
    // 每日收盘看好分：先用回算的历史，再用网站当时实际记录的（含海外 / 新闻调整）覆盖
    const closeScore = {};
    const sh = ((ch.score_hist || {})[which]) || {};
    Object.keys(sh).forEach((dt) => { const v = sh[dt] && sh[dt][code]; if (v && isNum(v[0])) closeScore[dt] = { t: v[0], v: v[1], s: v[2], src: "回算" }; });
    ((src.history || {})[code] || []).forEach((x) => {
      if (!String(x[0]).includes("盘中") && isNum(x[1])) closeScore[String(x[0]).slice(0, 10)] = { t: x[1], v: x[2], s: x[3], src: "当日记录" };
    });
    const sc = [];
    let prev = null;
    dates.forEach((dt) => {
      const st = p.score_today;
      if (st && st.date === dt && isNum(st.c)) {
        sc.push({ o: st.o, h: st.h, l: st.l, c: st.c, v: p.value, s: p.short, src: "盘中" });
        prev = st.c; return;
      }
      const cur = closeScore[dt];
      if (!cur) { sc.push(null); return; }
      const o = prev ?? cur.t;
      sc.push({ o, c: cur.t, h: Math.max(o, cur.t), l: Math.min(o, cur.t), v: cur.v, s: cur.s, src: cur.src });
      prev = cur.t;
    });
    return { dates, px, sc };
  }

  function ma(px, k) {
    // 最近 k 个有成交日的收盘均价；停牌日本身不画
    const buf = [];
    let s = 0;
    return px.map((b) => {
      if (!b) return null;
      buf.push(b[3]); s += b[3];
      if (buf.length > k) s -= buf.shift();
      return buf.length === k ? s / k : null;
    });
  }

  // ------------------------------------------------------------------ K 线绘制（SVG）
  function drawKline(el, S, opts) {
    const N = S.dates.length;
    if (!el.clientWidth) return;            // 详情不可见（窄屏未打开）：打开时再画
    if (!N || !S.px.some(Boolean)) {
      el.innerHTML = '<p class="hint">K 线数据还没有生成（下一次收盘计算后出现）。</p>';
      $("#kline-legend").innerHTML = "";
      return;
    }
    const n = Math.min(opts.n, N), off = N - n;
    const m5 = ma(S.px, 5).slice(off), m20 = ma(S.px, 20).slice(off);
    const px = S.px.slice(off), sc = S.sc.slice(off), dates = S.dates.slice(off);
    const W = Math.max(320, el.clientWidth || 640), H = Math.max(260, el.clientHeight || 380);
    const padL = 4, padR = 50, top = 16, axisH = 16, gap = 18;
    const avail = H - top - axisH - gap * 2;
    const hP = avail * 0.55, hV = avail * 0.13, hS = avail * 0.32;
    const yP0 = top, yV0 = yP0 + hP + gap, yS0 = yV0 + hV + gap;
    const plotW = W - padL - padR, step = plotW / n, cw = Math.max(1, Math.min(12, step * 0.68));
    const X = (i) => padL + step * (i + 0.5);
    // 价格坐标
    let lo = Infinity, hi = -Infinity;
    px.forEach((k, i) => { if (k) { lo = Math.min(lo, k[2]); hi = Math.max(hi, k[1]); } [m5[i], m20[i]].forEach((v) => { if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const pp = (hi - lo) * 0.05; lo -= pp; hi += pp;
    const YP = (v) => yP0 + hP * (1 - (v - lo) / (hi - lo));
    const vmax = Math.max(1, ...px.map((k) => (k && isNum(k[4]) ? k[4] : 0)));
    const YV = (v) => yV0 + hV * (1 - v / vmax);
    // 看好分坐标
    let slo = Infinity, shi = -Infinity;
    sc.forEach((s) => {
      if (!s) return;
      slo = Math.min(slo, s.l); shi = Math.max(shi, s.h);
      if (opts.showParts !== false && state.showParts) [s.v, s.s].forEach((v) => { if (isNum(v)) { slo = Math.min(slo, v); shi = Math.max(shi, v); } });
    });
    const hasScore = isFinite(slo);
    if (!hasScore) { slo = 0; shi = 100; }
    if (shi - slo < 4) { const mid = (shi + slo) / 2; slo = mid - 2; shi = mid + 2; }
    const sp = (shi - slo) * 0.08; slo = Math.max(0, slo - sp); shi = Math.min(100, shi + sp);
    const YS = (v) => yS0 + hS * (1 - (v - slo) / (shi - slo));
    const css = getComputedStyle(document.documentElement);
    const C = (k) => css.getPropertyValue(k).trim();
    const cUp = C("--k-up"), cDn = C("--k-down"), sUp = C("--s-up"), sDn = C("--s-down"), line = C("--line"), muted = C("--muted");
    let g = `<svg viewBox="0 0 ${W} ${H}">`;
    // 网格与右侧坐标
    const ticks = (y0, h, a, b, k, fmt) => {
      let s = "";
      for (let t = 0; t <= k; t++) {
        const v = a + (b - a) * t / k, y = y0 + h * (1 - t / k);
        s += `<line x1="${padL}" x2="${W - padR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${line}" stroke-width="1"/>`;
        s += `<text x="${W - padR + 4}" y="${(y + 4).toFixed(1)}" font-size="10" fill="${muted}">${fmt(v)}</text>`;
      }
      return s;
    };
    const pd = hi < 10 ? 2 : hi < 100 ? 2 : 1;
    g += ticks(yP0, hP, lo, hi, 4, (v) => v.toFixed(pd));
    g += ticks(yS0, hS, slo, shi, 3, (v) => v.toFixed(shi - slo < 10 ? 1 : 0));
    g += `<text x="${W - padR + 4}" y="${(yV0 + 9).toFixed(1)}" font-size="10" fill="${muted}">${(vmax / 1e4).toFixed(vmax >= 1e5 ? 0 : 1)}万手</text>`;
    // 分区标题
    g += `<text x="${padL + 2}" y="${yP0 - 4}" font-size="11" fill="${muted}">股价（前复权，红涨绿跌）</text>`;
    g += `<text x="${padL + 2}" y="${yV0 - 4}" font-size="11" fill="${muted}">成交量</text>`;
    g += `<text x="${padL + 2}" y="${yS0 - 4}" font-size="11" fill="${muted}">看好分 · 综合（蓝升橙降）</text>`;
    // 入选日标记
    const pi = dates.indexOf(opts.pickDate);
    if (pi >= 0) {
      const x = X(pi).toFixed(1);
      g += `<line x1="${x}" x2="${x}" y1="${yP0}" y2="${yS0 + hS}" stroke="${C("--accent")}" stroke-dasharray="3,3" stroke-width="1"/>`;
      const right = X(pi) > padL + plotW - 30;
      g += `<text x="${(right ? +x - 3 : +x + 3).toFixed(1)}" y="${yP0 + 10}" font-size="10" fill="${C("--accent")}" text-anchor="${right ? "end" : "start"}">入选</text>`;
    }
    // 股价 K 线 + 成交量
    px.forEach((k, i) => {
      if (!k) return;
      const [o, h, l, c, v] = k, up = c >= o, col = up ? cUp : cDn, x = X(i);
      g += `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${YP(h).toFixed(1)}" y2="${YP(l).toFixed(1)}" stroke="${col}" stroke-width="1"/>`;
      const y1 = YP(Math.max(o, c)), y2 = YP(Math.min(o, c));
      g += `<rect x="${(x - cw / 2).toFixed(1)}" y="${y1.toFixed(1)}" width="${cw.toFixed(1)}" height="${Math.max(1, y2 - y1).toFixed(1)}" fill="${col}"/>`;
      if (isNum(v)) g += `<rect x="${(x - cw / 2).toFixed(1)}" y="${YV(v).toFixed(1)}" width="${cw.toFixed(1)}" height="${Math.max(0.5, yV0 + hV - YV(v)).toFixed(1)}" fill="${col}" opacity=".55"/>`;
    });
    const poly = (arr, Y, col, w, dash) => {
      let s = "", seg = [];
      const flush = () => { if (seg.length > 1) s += `<polyline fill="none" stroke="${col}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ""} points="${seg.join(" ")}"/>`; seg = []; };
      arr.forEach((v, i) => { if (isNum(v)) seg.push(`${X(i).toFixed(1)},${Y(v).toFixed(1)}`); else flush(); });
      flush();
      return s;
    };
    g += poly(m5, YP, C("--ma5"), 1);
    g += poly(m20, YP, C("--ma20"), 1);
    // 看好分 K 线
    if (state.showParts) {
      g += poly(sc.map((s) => (s ? s.v : null)), YS, C("--p-value"), 1.2, "4,2");
      g += poly(sc.map((s) => (s ? s.s : null)), YS, C("--p-short"), 1.2, "2,2");
    }
    sc.forEach((s, i) => {
      if (!s) return;
      const up = s.c >= s.o, col = up ? sUp : sDn, x = X(i);
      if (s.h > Math.max(s.o, s.c) + 1e-9 || s.l < Math.min(s.o, s.c) - 1e-9)
        g += `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${YS(s.h).toFixed(1)}" y2="${YS(s.l).toFixed(1)}" stroke="${col}" stroke-width="1"/>`;
      const y1 = YS(Math.max(s.o, s.c)), y2 = YS(Math.min(s.o, s.c));
      g += `<rect x="${(x - cw / 2).toFixed(1)}" y="${y1.toFixed(1)}" width="${cw.toFixed(1)}" height="${Math.max(1, y2 - y1).toFixed(1)}" fill="${col}" ${s.src === "回算" ? 'opacity=".7"' : ""}/>`;
    });
    if (!hasScore) g += `<text x="${padL + plotW / 2}" y="${yS0 + hS / 2}" font-size="11" fill="${muted}" text-anchor="middle">暂无看好分历史</text>`;
    // 日期轴
    const nl = Math.max(2, Math.min(6, n, Math.floor(plotW / 85)));
    for (let t = 0; t < nl; t++) {
      const i = Math.round((n - 1) * (nl === 1 ? 1 : t / (nl - 1)));
      g += `<text x="${X(i).toFixed(1)}" y="${H - 3}" font-size="10" fill="${muted}" text-anchor="${t === 0 ? "start" : t === nl - 1 ? "end" : "middle"}">${esc(dates[i].slice(2))}</text>`;
    }
    g += `<line id="kx" x1="0" x2="0" y1="${yP0}" y2="${yS0 + hS}" stroke="${muted}" stroke-width="1" stroke-dasharray="2,2" visibility="hidden"/>`;
    g += `<rect id="kov" x="${padL}" y="0" width="${plotW}" height="${H}" fill="transparent"/>`;
    g += "</svg><div class=\"tip hidden\" id=\"ktip\"></div>";
    el.innerHTML = g;

    const legend = (i) => {
      const k = px[i], s = sc[i];
      let prevC = null;
      for (let j = off + i - 1; j >= 0; j--) if (S.px[j]) { prevC = S.px[j][3]; break; }   // 上一个有成交日的收盘
      const chg = k && prevC ? k[3] / prevC - 1 : null;
      return `<span><b>${esc(dates[i])}</b></span>` +
        (k ? `<span>收 <b class="${cls(chg)}">${fmtNum(k[3], 2)}</b> <span class="${cls(chg)}">${fmtPct(chg)}</span></span>` : "<span>停牌</span>") +
        `<span><i class="ln" style="background:${C("--ma5")}"></i>MA5 ${fmtNum(m5[i], 2)}</span><span><i class="ln" style="background:${C("--ma20")}"></i>MA20 ${fmtNum(m20[i], 2)}</span>` +
        (s ? `<span><i style="background:${s.c >= s.o ? sUp : sDn}"></i>看好分 ${fmtNum(s.c, 1)}${s.src === "回算" ? "（回算·样本内）" : s.src === "盘中" ? "（盘中）" : ""}</span>` : "") +
        (state.showParts ? `<span><i class="ln" style="background:${C("--p-value")}"></i>价值 ${fmtNum(s && s.v, 1)}</span><span><i class="ln" style="background:${C("--p-short")}"></i>短线 ${fmtNum(s && s.s, 1)}</span>` : "");
    };
    const lg = $("#kline-legend");
    lg.innerHTML = legend(n - 1);
    const ov = el.querySelector("#kov"), kx = el.querySelector("#kx"), tip = el.querySelector("#ktip");
    const move = (ev) => {
      const rect = el.querySelector("svg").getBoundingClientRect();
      const cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
      const i = Math.max(0, Math.min(n - 1, Math.floor((cx * W / rect.width - padL) / step)));
      const x = X(i);
      kx.setAttribute("x1", x); kx.setAttribute("x2", x); kx.setAttribute("visibility", "visible");
      lg.innerHTML = legend(i);
      const k = px[i], s = sc[i];
      tip.innerHTML = `<div class="h">${esc(dates[i])}</div>` +
        (k ? `<div><span class="k">开</span>${fmtNum(k[0], 2)}　<span class="k">高</span>${fmtNum(k[1], 2)}</div><div><span class="k">低</span>${fmtNum(k[2], 2)}　<span class="k">收</span>${fmtNum(k[3], 2)}</div>` +
          `<div><span class="k">成交</span>${isNum(k[4]) ? (k[4] / 1e4).toFixed(2) + " 万手" : "—"}</div>` : "<div>停牌 / 无数据</div>") +
        (s ? `<div style="margin-top:3px"><span class="k">看好分</span><b style="color:${s.c >= s.o ? sUp : sDn}">${fmtNum(s.c, 1)}</b>（前 ${fmtNum(s.o, 1)}${s.h !== s.l && s.src === "盘中" ? `，盘中 ${fmtNum(s.l, 1)}~${fmtNum(s.h, 1)}` : ""}）</div>` +
          `<div><span class="k">价值</span>${fmtNum(s.v, 1)}　<span class="k">短线</span>${fmtNum(s.s, 1)}</div>` +
          (s.src === "回算" ? '<div class="hint" style="margin:0;white-space:normal;max-width:220px">回算·样本内：网站上线前的分数，用现在的模型补算（模型训练时见过之后的走势，不是当时的真实预测），不含估值因子和海外/新闻调整，仅供参考</div>' : "") : "");
      tip.classList.remove("hidden");
      const px_ = x * rect.width / W;
      const tw = tip.offsetWidth;
      const cand = px_ + 14 + tw > rect.width ? px_ - tw - 14 : px_ + 14;
      tip.style.left = Math.max(0, Math.min(rect.width - tw, cand)) + "px";
      tip.style.top = "18px";
    };
    const leave = () => { kx.setAttribute("visibility", "hidden"); tip.classList.add("hidden"); lg.innerHTML = legend(n - 1); };
    ov.addEventListener("mousemove", move);
    let t0 = null;
    ov.addEventListener("touchstart", (e) => { t0 = { x: e.touches[0].clientX, y: e.touches[0].clientY, h: null }; move(e); }, { passive: true });
    ov.addEventListener("touchmove", (e) => {
      const t = e.touches[0];
      if (t0 && t0.h === null) {
        const dx = Math.abs(t.clientX - t0.x), dy = Math.abs(t.clientY - t0.y);
        if (dx + dy > 6) t0.h = dx > dy;
      }
      if (t0 && t0.h === false) return;      // 竖向滑动：交给页面滚动
      move(e); e.preventDefault();
    }, { passive: false });
    let lastTouch = 0;
    ov.addEventListener("touchstart", () => { lastTouch = Date.now(); }, { passive: true });
    // 触屏点按后浏览器会补发一次 mouseleave，忽略它，由 touchend 的计时来收起提示
    ov.addEventListener("mouseleave", () => { if (Date.now() - lastTouch > 2000) leave(); });
    ov.addEventListener("touchend", () => setTimeout(leave, 1500));
  }

  // ------------------------------------------------------------------ 下方：其余内容
  function newsLi(n) {
    const s = n.sentiment || 0;
    const tag = s > 0.15 ? "pos" : s < -0.15 ? "neg" : "";
    const allp = [...(state.dash.picks || []), ...((state.dash.tech || {}).picks || [])];
    const picks = (n.picks || []).map((c) => {
      const p = allp.find((x) => x.code === c);
      return p ? `<span class="chip">${esc(p.name)}</span>` : "";
    }).join("");
    const ev = (n.events || []).map((e) => `<span class="chip">${esc(e)}</span>`).join("");
    const title = n.url ? `<a href="${esc(n.url)}" target="_blank" rel="noopener">${esc(n.title)}</a>` : esc(n.title);
    return `<li><span class="dot ${tag}"></span>${title}<div class="meta">${esc(n.time)} · ${esc(n.source)} · 情绪 ${fmtNum(s, 2)} ${ev}${picks}</div></li>`;
  }

  function pickChips(codes) {
    const all = [...(state.dash.picks || []), ...((state.dash.tech || {}).picks || [])];
    return (codes || []).map((c) => {
      const p = all.find((x) => x.code === c);
      return p ? `<span class="chip">${esc(p.name)}</span>` : "";
    }).join("");
  }

  function renderWatch(w) {
    $("#signals").innerHTML = (w.signals || []).map((s) => `<div class="card sig ${s.level >= 3 ? "l3" : ""}"><div class="when">${esc(s.when || "实时信号")}</div>
      <div class="t">${esc(s.title)}</div><div class="s">${esc(s.text)}</div><div>${pickChips(s.picks)}</div></div>`).join("");
    $("#events").innerHTML = (w.events || []).map((e) => `<div class="card"><div class="when">${esc(e.date)}${e.end ? " ~ " + esc(e.end) : ""} ${esc(e.time || "")} · ${esc(e.category)} · ${esc(e.confidence || "")}<span class="stars">${"★".repeat(e.importance || 1)}</span></div>
      <div class="t">${esc(e.title)}</div>${e.scenarios ? `<div class="s">${esc(e.scenarios)}</div>` : ""}
      <div class="s">${(e.industries || []).map((x) => `<span class="chip">${esc(x)}</span>`).join("")}</div>
      ${e.picks && e.picks.length ? `<div class="s">可能影响的入选股：${pickChips(e.picks)}</div>` : ""}</div>`).join("");
    $("#themes").innerHTML = (w.themes || []).map((t) => `<div class="theme"><b>${esc(t.title)}</b>${esc(t.text)}${t.picks && t.picks.length ? `<div>${pickChips(t.picks)}</div>` : ""}</div>`).join("");
  }

  function renderNews(list) {
    $("#news").innerHTML = list.slice(0, 40).map(newsLi).join("") || '<li class="hint">暂无快讯</li>';
  }

  function line(series, opts = {}) {
    // 简单折线图（回测净值）。series: [{name, values:[...], color}]
    const W = 600, H = opts.h || 160, P = 26;
    const all = series.flatMap((s) => s.values.filter((v) => v !== null && !isNaN(v)));
    if (!all.length) return '<p class="hint">暂无数据</p>';
    let lo = Math.min(...all), hi = Math.max(...all);
    if (hi === lo) { hi += 1; lo -= 1; }
    const n = Math.max(...series.map((s) => s.values.length));
    const X = (i) => P + (W - 2 * P) * (n <= 1 ? 0.5 : i / (n - 1));
    const Y = (v) => H - P - (H - 2 * P) * (v - lo) / (hi - lo);
    let out = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
    for (let k = 0; k <= 3; k++) {
      const v = lo + (hi - lo) * k / 3, y = Y(v);
      out += `<line x1="${P}" x2="${W - P}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-width="1"/>`;
      const dg = opts.pct ? (hi - lo < 0.2 ? 1 : 0) : (hi - lo < 5 ? 1 : 0);
      const lv = opts.pct ? (v - 1) * 100 : v;
      const txt = (Math.abs(lv) < 0.5 * Math.pow(10, -dg) ? 0 : lv).toFixed(dg) + (opts.pct ? "%" : "");
      out += `<text x="2" y="${y + 4}" font-size="10" fill="var(--muted)">${txt}</text>`;
    }
    series.forEach((s) => {
      const pts = s.values.map((v, i) => (v === null || isNaN(v)) ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean);
      out += `<polyline fill="none" stroke="${s.color}" stroke-width="${s.w || 1.8}" points="${pts.join(" ")}"/>`;
    });
    if (opts.labels && opts.labels.length) {
      const L = opts.labels;
      [0, Math.floor((L.length - 1) / 2), L.length - 1].forEach((i) => {
        out += `<text x="${X(i)}" y="${H - 6}" font-size="10" fill="var(--muted)" text-anchor="middle">${esc(L[i])}</text>`;
      });
    }
    out += "</svg>";
    const legend = series.map((s) => `<span style="color:${s.color}">━</span> ${esc(s.name)}`).join("　");
    return out + `<div class="legend">${legend}</div>`;
  }

  function metricsHTML(m) {
    if (!m.length) return "";
    const keys = Object.keys(m[0]).filter((k) => k !== "区间");
    return `<tr><th class="l">区间</th>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr>` +
      m.map((r) => `<tr><td class="l">${esc(r["区间"])}</td>${keys.map((k) => `<td>${typeof r[k] === "number" ? (Math.abs(r[k]) < 5 && !/比|夏普|卡玛/.test(k) ? fmtPct(r[k], 1) : fmtNum(r[k], 2)) : esc(r[k] ?? "—")}</td>`).join("")}</tr>`).join("");
  }

  function renderModel() {
    const bt = state.bt;
    if (bt && bt.nav) {
      $("#bt-chart").innerHTML = line([
        { name: bt.nav.strategy_name || "策略", values: bt.nav.strategy, color: "var(--up)" },
        { name: bt.nav.bench_name || "基准", values: bt.nav.bench, color: "var(--muted)", w: 1.4 },
        ...(bt.nav.bench2 ? [{ name: bt.nav.bench2_name || "基准2", values: bt.nav.bench2, color: "var(--accent)", w: 1.2 }] : []),
      ], { h: 260, labels: bt.nav.dates, pct: true });
      $("#bt-note").textContent = bt.note || "";
      $("#bt-metrics").innerHTML = metricsHTML(bt.metrics || []);
    }
    const tr = state.dash.track || [];
    $("#track").innerHTML = tr.length ? `<tr><th class="l">月份</th><th>选股日</th><th>组合收益</th><th>全A等权</th><th>超额</th><th>状态</th></tr>` +
      tr.map((r) => `<tr><td class="l">${esc(r.month)}</td><td>${esc(r.pick_date)}</td><td class="${cls(r.ret)}">${fmtPct(r.ret)}</td><td class="${cls(r.bench)}">${fmtPct(r.bench)}</td><td class="${cls(r.excess)}">${fmtPct(r.excess)}</td><td>${esc(r.status || "")}</td></tr>`).join("")
      : '<tr><td class="hint">实盘跟踪从第一个完整月份开始记录。</td></tr>';
    $("#method").innerHTML = state.dash.method || (bt && bt.method) || "";
  }

  function renderTechBT() {
    const t = state.dash.tech;
    $('#more-tabs [data-more="techbt"]').classList.toggle("hidden", !t);
    if (!t) return;
    const w = t.weights || { value: 0.5, short: 0.5 };
    $("#tech-hint").textContent = `科技股周榜独立于 30 强：只在申万一级「${(t.industries || ["电子", "计算机", "通信", "传媒"]).join(" / ")}」的科技股池（${t.universe_n ?? "—"} 只）里选，` +
      `每周最后一个交易日收盘后选出下周 15 只，每天重新打分排名。综合分 = ${w.value}×价值分 + ${w.short}×短线分（科技股池内分位）。`;
    const bt = state.btTech;
    if (bt && bt.nav) {
      $("#tech-bt-chart").innerHTML = line([
        { name: bt.nav.strategy_name, values: bt.nav.strategy, color: "var(--up)" },
        { name: bt.nav.bench_name, values: bt.nav.bench, color: "var(--muted)", w: 1.4 },
      ], { labels: bt.nav.dates, pct: true });
      $("#tech-bt-metrics").innerHTML = metricsHTML(bt.metrics || []);
      $("#tech-bt-note").textContent = bt.note || "";
    }
    const tr = t.track || [];
    $("#tech-track").innerHTML = tr.length ? `<tr><th class="l">周</th><th>选出日</th><th>名单收益</th><th>科技股池等权</th><th>超额</th><th>状态</th></tr>` +
      tr.slice(-12).reverse().map((r) => `<tr><td class="l">${esc(r.week)}</td><td>${esc(r.pick_date)}</td><td class="${cls(r.ret)}">${fmtPct(r.ret)}</td><td class="${cls(r.bench)}">${fmtPct(r.bench)}</td><td class="${cls(r.excess)}">${fmtPct(r.excess)}</td><td>${esc(r.status || "")}</td></tr>`).join("")
      : '<tr><td class="hint">每周跟踪从第一周名单开始记录。</td></tr>';
  }

  // ------------------------------------------------------------------ 交互
  $$("#list-tabs button").forEach((b) => b.addEventListener("click", () => { state.list = b.dataset.list; renderList(); renderDetail(); }));
  $$("#sort-tabs button").forEach((b) => b.addEventListener("click", () => { state.sort[state.list] = b.dataset.sort; renderList(); }));
  $$("#range-tabs button").forEach((b) => b.addEventListener("click", () => {
    state.range = +b.dataset.n;
    $$("#range-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    renderDetail();
  }));
  $("#show-parts").addEventListener("change", (e) => { state.showParts = e.target.checked; renderDetail(); });
  $$("#d-tabs button").forEach((b) => b.addEventListener("click", () => showDTab(b.dataset.tab)));
  $$("#more-tabs button").forEach((b) => b.addEventListener("click", () => {
    state.more = b.dataset.more;
    $$("#more-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".more-tab").forEach((x) => x.classList.toggle("hidden", x.id !== state.more));
  }));
  $("#d-back").addEventListener("click", () => document.body.classList.remove("detail-open"));
  const listEl = $(".list-scroll");
  listEl.addEventListener("keydown", (e) => {
    if (!state.dash || !/^Arrow(Up|Down)$/.test(e.key)) return;
    const rows = $$("#list-body tr");
    const i = rows.findIndex((r) => r.dataset.code === state.sel[state.list]);
    const j = Math.max(0, Math.min(rows.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)));
    if (rows[j] && j !== i) { e.preventDefault(); select(rows[j].dataset.code, false); rows[j].scrollIntoView({ block: "nearest" }); }
  });
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { if (state.dash) { fitHeader(); renderDetail(); } }, 150); });
  load().catch((e) => { $("#subline").textContent = "数据加载失败：" + e.message; });
})();
