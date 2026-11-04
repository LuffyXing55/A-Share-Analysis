/* A股30强看好度 —— 前端。数据来源：本地服务 /api/live（盘中实时）或静态 data/dashboard.json（GitHub Pages）。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const state = { dash: null, bt: null, sort: "total", sortTech: "total", live: false, timer: null };

  const fmtPct = (x, d = 2) => (x === null || x === undefined || isNaN(x)) ? "—" : (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%";
  const fmtNum = (x, d = 2) => (x === null || x === undefined || isNaN(x)) ? "—" : Number(x).toFixed(d);
  const cls = (x) => (x > 0 ? "up" : x < 0 ? "down" : "flat");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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
    if (!state.bt) {
      try { state.bt = await getJSON(state.live ? "/data/backtest.json" : "data/backtest.json"); } catch (e) { state.bt = null; }
    }
    render();
    if (state.live && !state.timer) state.timer = setInterval(async () => {
      try { state.dash = await getJSON("/api/live"); render(); } catch (e) { /* 保留上一轮 */ }
    }, 30000);
  }

  function render() {
    const d = state.dash;
    $("#subline").textContent = `数据日期 ${d.data_date} · 更新于 ${d.generated_at} · 本月组合 ${d.month}（${d.pick_date} 收盘后选出）` +
      (d.picks_note && d.picks_note.includes("不完整") ? ` · ⚠ ${d.picks_note}` : "");
    const mb = $("#mode-badge");
    mb.textContent = state.live ? "● 盘中实时（本地）" : "日更（GitHub）";
    mb.className = "mode" + (state.live ? " live" : "");
    renderMarket(d.market || {});
    renderPicks();
    renderTech();
    renderWatch(d.watch || {});
    renderNews(d.news || []);
    renderModel();
  }

  function tile(n, v, c, digits, bp) {
    const chg = (bp !== undefined && bp !== null) ? `${bp > 0 ? "+" : ""}${Number(bp).toFixed(1)}bp` : fmtPct(c);
    const dir = (bp !== undefined && bp !== null) ? bp : c;
    return `<div class="tile"><div class="n">${esc(n)}</div><div class="v ${cls(dir)}">${fmtNum(v, digits ?? 2)}</div><div class="c ${cls(dir)}">${chg}</div></div>`;
  }

  function renderMarket(m) {
    $("#index-tiles").innerHTML = (m.indices || []).map((x) => tile(x.name, x.price, x.pct)).join("") +
      (m.breadth ? `<div class="tile"><div class="n">涨停 / 跌停 / 炸板</div><div class="v">${m.breadth.zt ?? "—"} / ${m.breadth.dt ?? "—"} / ${m.breadth.zb ?? "—"}</div><div class="c flat">${esc(m.breadth.note || "")}</div></div>` : "");
    $("#global-tiles").innerHTML = (m.global || []).map((x) => tile(x.name, x.price, x.pct, x.price > 1000 ? 0 : 2, x.bp)).join("");
    $("#market-time").textContent = m.time ? "行情时间 " + m.time : "";
    const r = m.regime;
    $("#regime").innerHTML = r ? `<b>市场状态：${esc(r.level)}</b> · ${esc(r.text)}` : "";
  }

  function rowsHTML(list, key) {
    const rows = [...(list || [])].sort((a, b) => (b[key] ?? -1) - (a[key] ?? -1));
    const tv = rows.map((p) => p.total).filter((v) => v !== null && v !== undefined);
    const tmin = Math.min(...tv), tmax = Math.max(...tv);
    return rows.map((p, i) => {
      const ch = (p.prev_rank && p.rank) ? p.prev_rank - p.rank : null;
      const chTxt = ch === null ? '<span class="flat">新</span>' : ch > 0 ? `<span class="up">↑${ch}</span>` : ch < 0 ? `<span class="down">↓${-ch}</span>` : '<span class="flat">–</span>';
      const sig = [];
      if (p.signals && Math.abs(p.signals.overseas || 0) >= 0.5) sig.push(`<span class="chip ${p.signals.overseas > 0 ? "pos" : "neg"}">海外${p.signals.overseas > 0 ? "利好" : "利空"}</span>`);
      if (p.signals && Math.abs(p.signals.news || 0) >= 0.2) sig.push(`<span class="chip ${p.signals.news > 0 ? "pos" : "neg"}">新闻${p.signals.news > 0 ? "偏正" : "偏负"}</span>`);
      if (p.is_st) sig.push('<span class="chip neg">ST</span>');
      if (p.suspended) sig.push('<span class="chip">停牌</span>');
      return `<tr data-code="${esc(p.code)}">
        <td>${key === "total" ? p.rank : i + 1}</td><td>${chTxt}</td>
        <td class="l"><b>${esc(p.name)}</b><span class="code">${esc(p.code.slice(2))}</span></td>
        <td class="l hide-s">${esc(p.industry || "")}</td>
        <td><span class="score">${fmtNum(p.total, 1)}</span><span class="bar" style="width:${tmax > tmin ? 4 + 46 * ((p.total ?? tmin) - tmin) / (tmax - tmin) : 25}px"></span></td>
        <td>${fmtNum(p.value, 1)}</td><td>${fmtNum(p.short, 1)}</td>
        <td class="${cls(p.pct_today)}">${fmtPct(p.pct_today)}</td>
        <td class="hide-s ${cls(p.ret_since_pick)}">${fmtPct(p.ret_since_pick)}</td>
        <td class="hide-s">${p.pe_ttm !== null && p.pe_ttm !== undefined ? (p.pe_ttm > 0 ? fmtNum(p.pe_ttm, 1) : "亏损") : "—"}</td>
        <td class="hide-s">${fmtNum(p.total_mv_yi, 0)}</td>
        <td class="hide-s l">${sig.join("")}</td></tr>`;
    }).join("");
  }

  function renderPicks() {
    const d = state.dash;
    const w = d.weights || { value: 0.6, short: 0.4 };
    $("#picks-hint").textContent = `综合分 = ${w.value}×价值投资分 + ${w.short}×短线分。分数是在全市场约 4000 只股票中的分位（0–100），入选股都在前列，所以差距体现在小数位；条形长度按 30 只之间的相对高低显示。点击任意一行查看原因、海外联动与新闻。` +
      (state.live ? " 本地模式下盘中每 1 分钟重算一次（页面每 30 秒拉取），价格变化会实时影响分数和排名。" : "");
    $("#picks-body").innerHTML = rowsHTML(d.picks, state.sort);
    document.querySelectorAll("#picks-body tr").forEach((tr) => tr.addEventListener("click", () => showDetail(tr.dataset.code, "main")));
  }

  function renderTech() {
    const t = state.dash.tech;
    if (!t || !t.picks || !t.picks.length) { $("#tech").classList.add("hidden"); return; }
    $("#tech").classList.remove("hidden");
    const w = t.weights || { value: 0.5, short: 0.5 };
    $("#tech-title").textContent = `科技股周榜 · ${t.picks.length} 只（${t.week}）`;
    $("#tech-hint").textContent = `独立于 30 强：只在申万一级「${(t.industries || ["电子", "计算机", "通信", "传媒"]).join(" / ")}」的科技股池（${t.universe_n ?? "—"} 只）里选，` +
      `每周最后一个交易日收盘后选出下周名单（${t.pick_date} 收盘后选出），每天重新打分排名。综合分 = ${w.value}×价值分 + ${w.short}×短线分，` +
      `分数是在科技股池内的分位。` + (t.note && t.note.includes("不完整") ? ` ⚠ ${t.note}` : "");
    $("#tech-body").innerHTML = rowsHTML(t.picks, state.sortTech);
    document.querySelectorAll("#tech-body tr").forEach((tr) => tr.addEventListener("click", () => showDetail(tr.dataset.code, "tech")));
    const tr = t.track || [];
    $("#tech-track").innerHTML = tr.length ? `<tr><th class="l">周</th><th>选出日</th><th>名单收益</th><th>科技股池等权</th><th>超额</th><th>状态</th></tr>` +
      tr.slice(-12).reverse().map((r) => `<tr><td class="l">${esc(r.week)}</td><td>${esc(r.pick_date)}</td><td class="${cls(r.ret)}">${fmtPct(r.ret)}</td><td class="${cls(r.bench)}">${fmtPct(r.bench)}</td><td class="${cls(r.excess)}">${fmtPct(r.excess)}</td><td>${esc(r.status || "")}</td></tr>`).join("")
      : '<tr><td class="hint">每周跟踪从第一周名单开始记录。</td></tr>';
  }

  function line(series, opts = {}) {
    // series: [{name, values:[...], color}], x: labels
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
      out += `<text x="2" y="${y + 4}" font-size="10" fill="var(--muted)">${opts.pct ? ((v - 1) * 100).toFixed(0) + "%" : v.toFixed(0)}</text>`;
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

  function reasonsHTML(list, title) {
    if (!list || !list.length) return "";
    const mx = Math.max(...list.map((r) => Math.abs(r.contrib)), 1e-6);
    return `<div class="hint">${esc(title)}</div>` + list.map((r) => {
      const w = Math.abs(r.contrib) / mx * 50;
      const left = r.contrib >= 0 ? 50 : 50 - w;
      return `<div class="reason"><span>${esc(r.name)}</span><div class="track"><div class="fill" style="left:${left}%;width:${w}%;background:${r.contrib >= 0 ? "var(--up)" : "var(--down)"}"></div></div><span class="${cls(r.contrib)}">${(r.pct !== undefined && r.pct !== null) ? "前" + Math.max(1, Math.round(100 - r.pct)) + "%" : ""}</span></div>`;
    }).join("");
  }

  function showDetail(code, which) {
    const d = state.dash;
    const src = which === "tech" && d.tech ? d.tech : d;
    const p = (src.picks || []).find((x) => x.code === code);
    if (!p) return;
    $("#d-hist-title").textContent = which === "tech" ? "看好分走势（本周，科技股池内分位）" : "看好分走势（本月）";
    $("#detail").classList.remove("hidden");
    $("#d-title").textContent = `${p.name}（${p.code.slice(2)}） · ${p.industry || ""} · 综合 ${fmtNum(p.total, 0)} / 价值 ${fmtNum(p.value, 0)} / 短线 ${fmtNum(p.short, 0)}`;
    const h = (src.history || {})[code] || [];
    $("#d-chart").innerHTML = line([
      { name: "综合分", values: h.map((x) => x[1]), color: "var(--accent)" },
      { name: "价值分", values: h.map((x) => x[2]), color: "var(--warn)", w: 1.2 },
      { name: "短线分", values: h.map((x) => x[3]), color: "var(--up)", w: 1.2 },
    ], { labels: h.map((x) => x[0].slice(5)) });
    $("#d-reasons").innerHTML = reasonsHTML(p.reasons_value, "价值投资分（中期，约 1 个月）的主要加分 / 减分项") +
      reasonsHTML(p.reasons_short, "短线分（约 1 周）的主要加分 / 减分项") +
      `<p class="hint">「前x%」表示该指标在全市场股票池中的位置。条形长度为该项对模型打分的贡献（LightGBM 特征贡献）。</p>`;
    const chains = (p.chains || []).map((c) => `<div class="card"><div class="t">${esc(c.name)} <span class="chip">${esc(c.strength || "")}</span></div>
      <div class="s">${esc(c.logic || "")}</div>
      <div class="s">海外标的：${(c.overseas || []).map((o) => `${esc(o.name || o.symbol)} <span class="${cls(o.pct)}">${fmtPct(o.pct)}</span>`).join("，") || "—"}</div>
      ${c.beta !== undefined && c.beta !== null ? `<div class="s">历史联动：海外涨 1% → 该股次日平均 ${fmtPct(c.beta / 100, 2)}（相关系数 ${fmtNum(c.corr, 2)}，${esc(c.sample || "")}）</div>` : ""}</div>`).join("");
    $("#d-chains").innerHTML = chains || '<p class="hint">该股没有明确的海外产业链映射。</p>';
    $("#d-news").innerHTML = (p.news || []).map(newsLi).join("") || '<li class="hint">近期没有匹配到相关新闻。</li>';
    $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }

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
    $("#signals").innerHTML = (w.signals || []).map((s) => `<div class="card sig"><div class="when">${esc(s.when || "实时信号")}</div>
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

  function renderModel() {
    const bt = state.bt;
    if (bt && bt.nav) {
      $("#bt-chart").innerHTML = line([
        { name: bt.nav.strategy_name || "策略", values: bt.nav.strategy, color: "var(--up)" },
        { name: bt.nav.bench_name || "基准", values: bt.nav.bench, color: "var(--muted)", w: 1.4 },
        ...(bt.nav.bench2 ? [{ name: bt.nav.bench2_name || "基准2", values: bt.nav.bench2, color: "var(--accent)", w: 1.2 }] : []),
      ], { h: 260, labels: bt.nav.dates, pct: true });
      $("#bt-note").textContent = bt.note || "";
      const m = bt.metrics || [];
      if (m.length) {
        const keys = Object.keys(m[0]).filter((k) => k !== "区间");
        $("#bt-metrics").innerHTML = `<tr><th class="l">区间</th>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr>` +
          m.map((r) => `<tr><td class="l">${esc(r["区间"])}</td>${keys.map((k) => `<td>${typeof r[k] === "number" ? (Math.abs(r[k]) < 5 && !/比|夏普|卡玛/.test(k) ? fmtPct(r[k], 1) : fmtNum(r[k], 2)) : esc(r[k] ?? "—")}</td>`).join("")}</tr>`).join("");
      }
    }
    const tr = state.dash.track || [];
    $("#track").innerHTML = tr.length ? `<tr><th class="l">月份</th><th>选股日</th><th>组合收益</th><th>全A等权</th><th>超额</th><th>状态</th></tr>` +
      tr.map((r) => `<tr><td class="l">${esc(r.month)}</td><td>${esc(r.pick_date)}</td><td class="${cls(r.ret)}">${fmtPct(r.ret)}</td><td class="${cls(r.bench)}">${fmtPct(r.bench)}</td><td class="${cls(r.excess)}">${fmtPct(r.excess)}</td><td>${esc(r.status || "")}</td></tr>`).join("")
      : '<tr><td class="hint">实盘跟踪从第一个完整月份开始记录。</td></tr>';
    $("#method").innerHTML = state.dash.method || (bt && bt.method) || "";
  }

  document.querySelectorAll("#sort-tabs-tech button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#sort-tabs-tech button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    state.sortTech = b.dataset.sort;
    renderTech();
  }));
  document.querySelectorAll("#sort-tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#sort-tabs button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    state.sort = b.dataset.sort;
    renderPicks();
  }));
  $("#d-close").addEventListener("click", () => $("#detail").classList.add("hidden"));
  load().catch((e) => { $("#subline").textContent = "数据加载失败：" + e.message; });
})();
