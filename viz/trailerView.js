// One source of truth: the video's clock. Every click seeks the video; every highlight follows it.
// Hovering while paused previews a shot; leaving the mark goes back to the shot under the playhead.
const PAD = { l: 56, r: 16 };
const GROUP_LABEL = {
    position: "position in film", dialogue: "dialogue", scale: "shot scale",
    content: "content", editing: "editing", clip: "look"
};
const vid = document.getElementById("vid"), tip = d3.select("#tip");
let D = null, starts = [], tx, fx, Y, W = 1200;
let cur = -1;                    // shot under the playhead
let preview = null;              // hover while paused: {trailer?, film?}
let focusFilm = null;            // a clicked film shot that isn't in the trailer
let selectedFilm = null;
let expanded = null;             // feature group opened in the "why" bars
let allGroups = false;           // the "why" bars show the top 3 groups unless opened
const API = location.origin;   // the server that serves this page

// --------------------------------------------------
// Processing overlay
// --------------------------------------------------

function showLoading(msg = "Processing your trailer...") {
    const el = document.getElementById("loading-overlay");
    if (!el) return;
    el.hidden = false;
    if (msg) updateLoading(msg);
    // Force a frame so opacity transitions smoothly from 0 -> 1
    requestAnimationFrame(() => el.classList.add("visible"));
}

function updateLoading(message) {
    document.querySelector("#loading-message").textContent = message;
}

function hideLoading() {
    const el = document.getElementById("loading-overlay");
    if (!el) return;
    el.classList.remove("visible");
}

let filmTitles = new Map();

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtT = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const pad4 = i => String(i).padStart(4, "0");
const tImg = (i, k = 1) =>
    `${API}/api/trailer-frame/${D.id}/${i}/${k}`;

const fImg = (i, k = 1) =>
    `${API}/api/film-frame/${D.id}/${i}/${k}`;

const framePreloadCache = new Map();

function preloadFrame(src) {
    if (framePreloadCache.has(src)) {
        return framePreloadCache.get(src);
    }

    const promise = new Promise(resolve => {
        const image = new Image();

        image.onload = () => resolve(image);
        image.onerror = () => resolve(null);

        image.src = src;
    });

    framePreloadCache.set(src, promise);

    // Keep the browser-side cache bounded.
    if (framePreloadCache.size > 100) {
        const oldest = framePreloadCache.keys().next().value;
        framePreloadCache.delete(oldest);
    }

    return promise;
}

const img = src => `
  <img
    src="${src}"
    alt="Frame"
    loading="lazy"
    decoding="async"
    onerror="
      this.style.display='none';
      this.insertAdjacentHTML(
        'afterend',
        '<div class=empty>Frame unavailable</div>'
      );
    "
  >
`;
const shotAt = t => Math.max(0, d3.bisectRight(starts, t) - 1);
const usersOf = j => D.shots.map((s, i) => [s, i]).filter(d => d[0][3] === j).map(d => d[1]);

function showTip(ev, html) {
    tip.html(html).style("display", "block");
    const x = Math.min(ev.clientX + 14, innerWidth - 400), y = Math.min(ev.clientY + 14, innerHeight - 200);
    tip.style("left", x + "px").style("top", y + "px");
}
const hideTip = () => tip.style("display", "none");

async function init() {
    await loadClusters();

    const index = await d3.json("/api/films");
    index.sort((a, b) => d3.ascending(a.title, b.title));

    const input = document.getElementById("search");
    const dropdown = document.getElementById("search-results");

    function renderDropdown(query = "") {
        const q = query.trim().toLowerCase();
        const matches = q
            ? index.filter(d => d.title.toLowerCase().includes(q))
            : index;

        dropdown.innerHTML = "";
        matches.slice(0, 100).forEach(d => {
            const item = document.createElement("div");
            item.className = "search-item";
            item.textContent = d.title;
            item.title = d.title; // full title on hover
            item.addEventListener("mousedown", (e) => {
                e.preventDefault(); // fires before input blur
                input.value = d.title;
                dropdown.classList.remove("open");
                load(d.id);
            });
            dropdown.appendChild(item);
        });

        dropdown.classList.toggle("open", matches.length > 0);
    }

    input.addEventListener("focus", () => renderDropdown(input.value));
    input.addEventListener("input", () => renderDropdown(input.value));
    input.addEventListener("blur", () => dropdown.classList.remove("open"));

    load(new URLSearchParams(location.search).get("id") || index[0].id);
}

let clusters = [];

async function loadClusters() {
    try {
        clusters = await d3.csv("../data/features/clusters.csv");
    } catch (err) {
        console.warn("Could not load clusters.csv:", err);
        clusters = [];
    }
}

function getYoutubeId(id) {
    const imdbId = String(id).trim();

    const row = clusters.find(
        d => String(d.imdb_id).trim() === imdbId
    );

    return row?.youtube_id?.trim() || null;
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function ensureViewerData(id) {
    const response = await fetch(`${API}/api/trailer/${id}`);

    if (!response.ok) {
        throw new Error(`Preparation request failed: ${response.status}`);
    }

    let result = await response.json();

    while (result.status !== "ready") {

        if (result.status?.startsWith("error:")) {
            throw new Error(result.status);
        }

        updateLoading(
            result.status || "Processing film and generating features..."
        );

        await sleep(1000);

        const statusResponse = await fetch(
            `${API}/api/trailer/${id}/status`
        );

        if (!statusResponse.ok) {
            throw new Error("Could not check preparation status");
        }

        result = await statusResponse.json();
    }

    updateLoading("Loading visualization...");

    const data = await d3.json(`data/${id}.json`);

    if (!data) {
        throw new Error("Visualization data was not generated");
    }

    return data;
}

let loadVersion = 0;

async function load(id) {
    const version = ++loadVersion;
    const minDelay = new Promise(resolve => setTimeout(resolve, 600));
    showLoading("Loading film data...");
    const t0 = +new URLSearchParams(location.search).get("t") || 0;

    try {
        // Wait for both the data AND at least 600ms so the transition feels intentional
        const [data] = await Promise.all([ensureViewerData(id), minDelay]);

        // Ignore stale requests.
        if (version !== loadVersion) return;

        D = data;
        starts = D.shots.map(s => s[0]);

        cur = -1;
        preview = null;
        focusFilm = null;
        selectedFilm = null;
        expanded = null;

        history.replaceState(null, "", t0 > 0 ? `?id=${id}&t=${t0}` : `?id=${id}`);

        const nMatched = d3.sum(D.shots, s => s[2]);
        const nUsed = d3.sum(D.film.in_trailer);

        d3.select("#title").text(D.title.replace(/\s*\(\d{4}\)$/, ""));

        d3.select("#meta").text(
            `${D.year} · ` +
            `${(D.genres || []).slice(0, 2).join(", ")} · ` +
            `${D.shots.length} trailer shots, ` +
            `${Math.round(100 * nMatched / D.shots.length)}% matched · ` +
            `${nUsed}/${D.film.n} film shots used`
        );

        drawAll();

        cur = 0;
        render();

        // Initialize and load YouTube after drawAll() has built the tx/fx scales
        const youtubeId = getYoutubeId(id);

        if (youtubeId) {
            await loadYouTubeVideo(youtubeId, t0);
        } else {
            follow(t0);
        }

    } catch (err) {
        console.error("Could not load viewer data:", id, err);

        updateLoading("Something went wrong: " + err.message);

        document.getElementById("retry-loading").hidden = false;

        return;
    }

    hideLoading();
}

let ytMedia = null;
let captionsOn = true;

const pp = document.getElementById("pp");
const clock = document.getElementById("clock");
const ccBtn = document.getElementById("cc");
const muteBtn = document.getElementById("mute");

// Current YouTube playback state (using YTMedia's <video>-like API)
function isPlaying() {
    return ytMedia ? !ytMedia.paused : false;
}

function currentTime() {
    return ytMedia ? ytMedia.currentTime : 0;
}

function duration() {
    const d = ytMedia ? ytMedia.duration : NaN;
    return Number.isFinite(d) && d > 0 ? d : (D?.duration || 0);
}

function ensureYTMedia() {
    if (ytMedia) return ytMedia;

    vid.innerHTML = '<div id="youtube-player"></div>';
    ytMedia = new YTMedia("youtube-player");

    ytMedia.addEventListener("play", () => {
        preview = null;
        focusFilm = null;
        render();
        syncBar();
    });

    ytMedia.addEventListener("pause", () => {
        syncBar();
        follow(currentTime());
    });

    ytMedia.addEventListener("timeupdate", () => {
        syncBar();
        if (!ytMedia.paused) {
            follow(currentTime());
        }
    });

    ytMedia.addEventListener("seeked", () => {
        syncBar();
        follow(currentTime());
    });

    ytMedia.addEventListener("loadedmetadata", () => {
        syncBar();
        if (captionsOn && ytMedia?.p?.loadModule) {
            ytMedia.p.loadModule("captions");
            ytMedia.p.setOption("captions", "track", { languageCode: "en" });
        }
        if (D && D.duration) {
            console.log("YouTube duration:", duration(), "Timeline duration:", D.duration);
        }
    });

    ytMedia.addEventListener("error", () => {
        console.error("YouTube playback error for video:", ytMedia.videoId);
    });

    return ytMedia;
}

async function loadYouTubeVideo(videoId, startAt = 0) {
    const media = ensureYTMedia();
    await media.load(videoId, startAt);
    syncBar();
    follow(startAt);
}

// Seeking
function seek(t) {
    if (!ytMedia) return;

    t = Math.max(0, Math.min(duration() - 0.05, t));

    focusFilm = null;
    preview = null;

    ytMedia.currentTime = t + 0.01;
    follow(t + 0.01);
}

// Playback
function toggle() {
    if (!ytMedia) return;

    if (isPlaying()) {
        ytMedia.pause();
    } else {
        ytMedia.play();
    }
}

// Shot navigation
function stepShot(dir) {
    if (!D || !ytMedia) return;

    const t = currentTime();
    const i = shotAt(t);

    if (dir < 0 && t - starts[i] > 0.6) {
        return seek(starts[i] + 0.01);
    }

    const next = Math.max(
        0,
        Math.min(starts.length - 1, i + dir)
    );

    seek(starts[next] + 0.01);
}

// Playback controls
pp.addEventListener("click", toggle);

document.getElementById("prev")
    .addEventListener("click", () => stepShot(-1));

document.getElementById("next")
    .addEventListener("click", () => stepShot(1));

// Synchronise controls and clock
function syncBar() {
    if (!ytMedia) return;

    const playing = isPlaying();

    pp.textContent = playing ? "❚❚" : "▶";
    pp.setAttribute("aria-label", playing ? "Pause" : "Play");

    clock.textContent =
        `${fmtT(currentTime())} / ${fmtT(duration())}`;

    muteBtn.setAttribute(
        "aria-pressed",
        !ytMedia.muted
    );
}

// Captions
function toggleCC() {
    captionsOn = !captionsOn;

    ccBtn.setAttribute("aria-pressed", captionsOn);

    if (!ytMedia?.p) return;

    if (captionsOn) {
        ytMedia.p.loadModule("captions");
        ytMedia.p.setOption("captions", "track", {
            languageCode: "en"
        });
    } else {
        ytMedia.p.unloadModule("captions");
    }
}

ccBtn.addEventListener("click", toggleCC);

// Mute
function toggleMute() {
    if (!ytMedia) return;

    ytMedia.muted = !ytMedia.muted;

    syncBar();
}

muteBtn.addEventListener("click", toggleMute);

// Keyboard shortcuts
document.querySelectorAll(".controls button")
    .forEach(b => {
        b.addEventListener("mousedown", e => e.preventDefault());
    });

addEventListener("keydown", ev => {
    if (
        ev.target.tagName === "INPUT" ||
        ev.metaKey ||
        ev.ctrlKey ||
        ev.altKey ||
        !D
    ) return;

    const k = ev.key;

    if (
        ev.target.tagName === "BUTTON" &&
        (k === " " || k === "Enter")
    ) return;

    if (k === " " || k === "k") {
        ev.preventDefault();
        toggle();
    }

    else if (k === "ArrowRight" || k === "ArrowLeft") {
        ev.preventDefault();
        stepShot(k === "ArrowRight" ? 1 : -1);
    }

    else if (k === "Home") {
        ev.preventDefault();
        seek(0);
    }

    else if (k === "c") {
        toggleCC();
    }

    else if (k === "m") {
        toggleMute();
    }

    else if (k === "Escape") {
        preview = null;
        focusFilm = null;
        render();
    }
});

// Resize
let resizeT;

addEventListener("resize", () => {
    clearTimeout(resizeT);

    resizeT = setTimeout(() => {
        if (D) {
            drawAll();
            follow(currentTime());
        }
    }, 150);
});

function follow(t) {
    if (!D || typeof tx !== "function") return;

    cur = shotAt(t);

    // Update the playhead position
    d3.select("#playhead")
        .attr("x1", tx(t))
        .attr("x2", tx(t));

    // Update the clock
    if (ytMedia) {
        clock.textContent = `${fmtT(t)} / ${fmtT(duration())}`;
    }

    // Update the side panel and highlights
    render();
}

// ---- what is shown: preview > clicked film shot > shot under the playhead ----
function current() {
    if (preview) return preview;

    if (selectedFilm !== null) {
        return { film: selectedFilm };
    }

    if (focusFilm !== null) {
        return { film: focusFilm };
    }

    return cur >= 0
        ? { trailer: cur, film: D.shots[cur][3] }
        : null;
}
const same = (a, b) => a === b || (a && b && a.trailer === b.trailer && a.film === b.film);
function hover(h) {
    if (!isPlaying() && !same(h, preview)) {
        preview = h;
        render();
    }
}

function unhover() { hideTip(); if (preview) { preview = null; render(); } }
function render() { const h = current(); highlight(h); drawNow(h); }

function highlight(h) {
    const t = h && h.trailer !== undefined ? h.trailer : null, f = h && h.film >= 0 ? h.film : null;
    const onLink = i => t !== null ? i === t : D.shots[i][3] === f;
    const links = d3.select("#main .links").selectAll("path");
    const none = !h || (f === null && t !== null);  // an unmatched trailer shot has no link to single out
    d3.selectAll("#order .dot").classed("cur", ([, i]) => t !== null && i === t);
    d3.selectAll("#order .dot.cur").raise();
    if (none) links.classed("hi", false);
    else {
        links.classed("hi", ([, i]) => onLink(i));
        links.filter(".hi").raise();  // draw the highlighted link on top of the others
    }
    const rows = d3.select("#tbl tbody").selectAll("tr").classed("cur", (_, i) => i === t);
    if (t !== null && document.getElementById("tbl").open) {
        const r = rows.nodes()[t], box = r && r.closest(".tbl-wrap");
        if (box && (r.offsetTop < box.scrollTop + 24 || r.offsetTop > box.scrollTop + box.clientHeight - 24)) box.scrollTop = r.offsetTop - box.clientHeight / 2;
    }
    const th = d3.select("#trailer-hi"), fh = d3.select("#film-hi");
    if (t === null) th.style("display", "none");
    else { const s = D.shots[t]; th.style("display", null).attr("x", tx(s[0]) - 2).attr("width", Math.max(5, tx(s[1]) - tx(s[0]) + 4)); }
    if (f === null) fh.style("display", "none");
    else { const a = fx(D.film.start[f]), b = fx(D.film.end[f]); fh.style("display", null).attr("x", a - 2).attr("width", Math.max(5, b - a + 4)); }
}

// the shot under the playhead (or the one hovered / clicked): its frame next to the film frame it matched

function drawNow(h) {
    const box = d3.select("#now");

    if (!h) {
        box.html("");
        return;
    }

    const f = D.film;
    const t = h.trailer !== undefined ? h.trailer : null;
    const j = h.film >= 0 ? h.film : null;

    const pos = j !== null
        ? `${Math.round(100 * f.story_pos[j])}% into the film`
        : "";

    let head;

    if (t !== null) {
        const s = D.shots[t];
        head = `<b>${j !== null ? `Film shot ${j + 1}` : `Unmatched trailer shot ${t + 1}`}</b>
      <span>${fmtT(s[0])}–${fmtT(s[1])}</span>
      <span class="tag">${j !== null ? "matched" : "no match"}</span>
      ${pos ? `<span class="pos">${pos}</span>` : ""}`;
    } else {
        const users = usersOf(j);
        head = `<b>Source film · Shot ${j + 1}</b>
      <span class="tag">${users.length ? "in the trailer" : "not in the trailer"}</span>
      ${pos ? `<span class="pos">${pos}</span>` : ""}`;
    }

    // Only show the film frame. The trailer is already visible in the player.
    const frames = j !== null
        ? `<figure>${img(fImg(j))}</figure>`
        : `<div class="empty">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="2" y="4" width="20" height="16" rx="2.5"></rect>
              <line x1="7" y1="4" x2="7" y2="20"></line>
              <line x1="17" y1="4" x2="17" y2="20"></line>
              <line x1="2" y1="9" x2="7" y2="9"></line>
              <line x1="2" y1="15" x2="7" y2="15"></line>
              <line x1="17" y1="9" x2="22" y2="9"></line>
              <line x1="17" y1="15" x2="22" y2="15"></line>
              <line x1="9.5" y1="9.5" x2="14.5" y2="14.5"></line>
              <line x1="14.5" y1="9.5" x2="9.5" y2="14.5"></line>
            </svg>
            <div class="empty-title">No corresponding film shot</div>
            <div class="empty-sub">This segment appears only in the trailer—such as a rating card, studio bumper, title graphic, or alternate cut.</div>
          </div>`;

    box.html(`
    <div class="head">${head}</div>
    <div class="body">
      <div class="frames">${frames}</div>
      <div class="why">
        <div class="why-h">
          <span>What moved the model's score, vs the average shot</span>
          <span>blue: toward the trailer · red: away</span>
        </div>
        <svg id="shap"></svg>
      </div>
    </div>
  `);

    shapFor = j;
    drawShap(j);
}

// the reasons chart is sized to the panel; redraw it whenever the panel's width settles or changes
let shapFor = null, panelW = 0;
new ResizeObserver(([e]) => {
    const w = Math.round(e.contentRect.width);
    if (w !== panelW) { panelW = w; if (D) drawShap(shapFor); }
}).observe(document.getElementById("now"));

function drawShap(i) {
    const f = D.film, svg = d3.select("#shap");
    if (i === null || !f.shap || f.trailerness[i] == null) { d3.select("#now .why").remove(); return; }
    svg.selectAll("*").remove();
    const w = document.getElementById("now").clientWidth - 30;  // the panel's inner width (padding 14px each side)
    // a few reasons read better than all of them: top 3 groups, the rest behind "more"
    let rows = f.shap_groups.map((g, k) => ({ key: g, label: GROUP_LABEL[g], v: f.shap[i][k], group: true }));
    rows.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const hidden = rows.length - 3;
    if (!allGroups && hidden > 0) rows = rows.slice(0, 3);
    const detail = f.detail[i];
    if (expanded && detail) {
        const at = rows.findIndex(r => r.key === expanded);
        const feats = f.detail_features.map((n, k) => ({ n, val: detail[k][0], v: detail[k][1] }))
            .filter(d => groupOf(d.n) === expanded).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 6)
            .map(d => ({ key: d.n, label: `  ${d.n} = ${d.val}`, v: d.v, group: false }));
        rows.splice(at + 1, 0, ...feats);
    }
    const rh = 20, H = rows.length * rh + 4;
    svg.attr("width", w).attr("height", H).attr("viewBox", `0 0 ${w} ${H}`);
    const m = d3.max(rows, r => Math.abs(r.v)) || 1, x = d3.scaleLinear([-m, m], [Math.min(150, w * 0.42) + 40, w - 40]);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", H).attr("stroke", "var(--axis)");
    if (rows.length > 1) svg.append("line").attr("x1", 0).attr("x2", w).attr("y1", rh + 1).attr("y2", rh + 1).attr("stroke", "var(--grid)");
    const g = svg.selectAll("g.shap-row").data(rows).join("g").attr("class", "shap-row")
        .attr("transform", (_, k) => `translate(0,${k * rh})`)
        .on("click", (ev, r) => { if (r.group && r.key !== "clip" && detail) { expanded = expanded === r.key ? null : r.key; drawShap(i); } });
    // only the top reason is reliable across retrained models (76% vs 55% for the top-3 set, shap_stability.json):
    // it gets the weight, the rest read as secondary
    const top = rows[0];
    g.append("text").attr("y", 14).style("font-weight", r => r === top ? 600 : 400)
        .style("fill", r => r === top ? null : "var(--ink-2)")
        .text(r => r.group ? `${r.key === expanded ? "▾" : r.key === "clip" || !detail ? " " : "▸"} ${r.label}` : r.label);
    g.append("path").attr("d", r => bar(x(0), x(r.v), 4, 12))
        .attr("fill", r => r.v >= 0 ? "var(--pos)" : "var(--neg)").attr("opacity", r => r.group ? 1 : 0.7);
    // values are log-odds contributions; shown as what they do to the odds (×1.5 = 50% better odds)
    g.append("text").attr("class", "val").attr("y", 14).attr("x", r => r.v >= 0 ? x(r.v) + 4 : x(r.v) - 4)
        .attr("text-anchor", r => r.v >= 0 ? "start" : "end").text(r => `×${d3.format(".2f")(Math.exp(r.v))}`);
    if (hidden > 0) {
        svg.attr("height", H + 18).attr("viewBox", `0 0 ${w} ${H + 18}`);
        svg.append("text").attr("class", "lbl").attr("y", H + 12).style("cursor", "pointer")
            .text(allGroups ? "▴ fewer" : `▾ ${hidden} more`).on("click", () => { allGroups = !allGroups; drawShap(i); });
    }
}
// a horizontal bar from x0 (the baseline, square) to x1 (the data end, rounded 4px)
function bar(x0, x1, y, h) {
    const d = x1 - x0, w = Math.abs(d), r = Math.min(4, w / 2, h / 2), k = Math.sign(d) || 1;
    if (w < 1) return `M${x0},${y}h${k}v${h}h${-k}Z`;
    return `M${x0},${y}H${x1 - k * r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - k * r},${y + h}H${x0}Z`;
}
function groupOf(n) {
    if (n.startsWith("c_")) return "content";
    if (n.startsWith("scale_")) return "scale";
    if (n.startsWith("dlg_")) return "dialogue";
    if (["story_pos", "act", "synopsis_pos"].includes(n)) return "position";
    return "editing";
}

function drawAll() { drawMain(); drawOrder(); drawTable(); render(); }

// the same shots as a table (the accessible twin of the timeline); a row click seeks there
function drawTable() {
    const f = D.film;
    d3.select("#tbl tbody").selectAll("tr").data(D.shots).join("tr")
        .html((s, i) => `<td>${i + 1}</td><td>${fmtT(s[0])}</td><td>${fmtT(s[1])}</td><td>${s[2] ? "yes" : "no"}</td>` +
            `<td>${s[3] >= 0 ? s[3] : "–"}</td><td>${s[3] >= 0 ? Math.round(100 * f.story_pos[s[3]]) + "%" : "–"}</td><td>${s[4] != null ? s[4].toFixed(2) : "–"}</td>`)
        .on("click", (ev, s) => {
            preview = null;
            selectedFilm = s[3] >= 0 ? s[3] : null;

            seek(s[0] + 0.01);
            render();
        });
}

// ---- timeline: trailer on top (the scrubber), film below, links between ----
function drawMain() {
    W = Math.max(560, document.getElementById("main-wrap").clientWidth);
    const f = D.film;
    Y = { tcolor: [6, 62], tmatch: [65, 72], tAxis: 76, link: [100, 250], used: [252, 262], barcode: [264, 320] };
    Y.fAxis = Y.barcode[1] + 4;
    const H = Y.fAxis + 26;
    const svg = d3.select("#main").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    tx = d3.scaleLinear([0, D.duration], [PAD.l, W - PAD.r]);
    fx = d3.scaleLinear([0, 1], [PAD.l, W - PAD.r]);
    const label = (y, t) => svg.append("text").attr("class", "lbl").attr("x", PAD.l - 8).attr("y", y).attr("text-anchor", "end").text(t);

    const centre = r => (r[0] + r[1]) / 2 + 4;
    label(centre(Y.tcolor), "trailer");
    label(centre(Y.barcode), "film");

    const tcol = D.shot_colors || [];
    svg.append("g").selectAll("rect").data(D.shots).join("rect")
        .attr("x", s => tx(s[0])).attr("width", s => Math.max(0.5, tx(s[1]) - tx(s[0])))
        .attr("y", Y.tcolor[0]).attr("height", Y.tcolor[1] - Y.tcolor[0]).attr("fill", (_, i) => tcol[i] || "var(--unmatched)");
    svg.append("g").attr("class", "tcells").selectAll("rect").data(D.shots).join("rect")
        .attr("x", s => tx(s[0]) + 1).attr("width", s => Math.max(1, tx(s[1]) - tx(s[0]) - 2))
        .attr("y", Y.tmatch[0]).attr("height", Y.tmatch[1] - Y.tmatch[0])
        .attr("fill", s => s[2] ? "var(--matched)" : "none");  // no match is left empty
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Y.tAxis})`)
        .call(d3.axisBottom(tx).ticks(Math.round(W / 110)).tickFormat(fmtT).tickSizeOuter(0));

    // links: trailer shot -> where it is in the film
    const y0 = Y.tAxis + 18, mid = (y0 + Y.link[1]) / 2;
    svg.append("g").attr("class", "links").selectAll("path").data(D.shots.map((s, i) => [s, i]).filter(d => d[0][3] >= 0))
        .join("path").attr("class", "link")
        .attr("d", ([s]) => { const a = tx((s[0] + s[1]) / 2), b = fx(f.story_pos[s[3]]); return `M${a},${y0} C${a},${mid} ${b},${mid} ${b},${Y.link[1]}`; });

    // film: shots used by the trailer, then the color barcode
    const shots = d3.range(f.n);
    svg.append("g").selectAll("rect").data(shots.filter(i => f.in_trailer[i])).join("rect")
        .attr("x", i => fx(f.story_pos[i]) - 1).attr("width", 2).attr("y", Y.used[0]).attr("height", Y.used[1] - Y.used[0]).attr("fill", "var(--used)");
    if (f.color) {
        svg.append("g").selectAll("rect").data(shots).join("rect")
            .attr("x", i => fx(f.start[i])).attr("width", i => Math.max(0.5, fx(f.end[i]) - fx(f.start[i])))
            .attr("y", Y.barcode[0]).attr("height", Y.barcode[1] - Y.barcode[0]).attr("fill", i => f.color[i] || "var(--unmatched)");
    }
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${Y.fAxis})`)
        .call(d3.axisBottom(fx).ticks(10).tickFormat(d3.format(".0%")).tickSizeOuter(0));

    svg.append("rect").attr("id", "trailer-hi").attr("class", "outline").attr("rx", 1)
        .attr("y", Y.tcolor[0] - 3).attr("height", Y.tmatch[1] - Y.tcolor[0] + 6).style("display", "none");
    svg.append("rect").attr("id", "film-hi").attr("class", "outline").attr("rx", 1)
        .attr("y", Y.used[0] - 3).attr("height", Y.barcode[1] - Y.used[0] + 6).style("display", "none");
    svg.append("line").attr("id", "playhead").attr("y1", Y.tcolor[0]).attr("y2", Y.tmatch[1]);

    // the trailer band is the scrubber: press or drag anywhere on it to seek
    let scrubbing = false;
    const t = ev => tx.invert(d3.pointer(ev, svg.node())[0]);
    svg.append("rect").attr("class", "scrub").attr("x", PAD.l).attr("width", W - PAD.l - PAD.r)
        .attr("y", Y.tcolor[0]).attr("height", Y.tmatch[1] - Y.tcolor[0]).attr("fill", "transparent")
        .on("pointerdown", function (ev) { scrubbing = true; this.setPointerCapture(ev.pointerId); hideTip(); preview = null; seek(t(ev)); })
        .on("pointerup pointercancel", () => { scrubbing = false; })
        .on("pointermove", ev => {
            const tt = t(ev);

            if (scrubbing) return seek(tt);

            const i = shotAt(tt);
            const s = D.shots[i];

            hover({ trailer: i, film: s[3] });
            hideTip();
        })
        .on("pointerleave", unhover);

    // film shots: click one the trailer uses to jump there (again for its next use); others just show
    svg.append("g").selectAll("rect").data(shots).join("rect").attr("class", "hit")
        .attr("x", i => fx(f.start[i])).attr("width", i => Math.max(4, fx(f.end[i]) - fx(f.start[i])))
        .attr("y", Y.used[0]).attr("height", Y.barcode[1] - Y.used[0]).attr("fill", "transparent")
        .on("pointermove", (ev, i) => { hover({ film: i }); showTip(ev, img(fImg(i)) + `<div>Film shot ${i} · ${Math.round(100 * f.story_pos[i])}%</div>`); })
        .on("pointerleave", unhover)
        .on("click", (ev, i) => {
            const users = usersOf(i);
            hideTip();

            focusFilm = i;
            preview = null;
            render();

            if (users.length) {
                const nxt = users.find(u => u > cur) ?? users[0];
                seek(starts[nxt] + 0.01);
            }
        });
}

// scatter helpers: a visible dot plus a 24px transparent hit target
function scatter(svg, data, cls, cx, cy, on) {
    svg.append("g").selectAll("circle").data(data).join("circle").attr("class", cls).attr("r", 4.5).attr("cx", cx).attr("cy", cy);
    svg.append("g").selectAll("circle").data(data).join("circle").attr("class", "hit").attr("r", 12).attr("fill", "transparent")
        .attr("cx", cx).attr("cy", cy).on("pointermove", on.move).on("pointerleave", on.leave).on("click", on.click);
}
function axes(svg, x, y, w, H, m, xf, xl, yl) {
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(xf).tickSizeOuter(0));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")).tickSizeOuter(0));
    svg.append("text").attr("class", "lbl").attr("x", w - m.r).attr("y", H - 4).attr("text-anchor", "end").text(xl);
    svg.append("text").attr("class", "lbl").attr("x", m.l).attr("y", 10).attr("text-anchor", "middle").text(yl);
}

function drawOrder() {
    const w = document.getElementById("order").parentNode.clientWidth - 28, H = 300, m = { l: 44, r: 12, t: 20, b: 36 };
    const svg = d3.select("#order").attr("width", w).attr("height", H).attr("viewBox", `0 0 ${w} ${H}`);
    svg.selectAll("*").remove();
    const x = d3.scaleLinear([0, D.duration], [m.l, w - m.r]), y = d3.scaleLinear([0, 1], [H - m.b, m.t]);
    svg.append("line").attr("class", "diag").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(D.duration)).attr("y2", y(1));
    svg.append("text").attr("class", "lbl").attr("x", x(D.duration) - 4).attr("y", y(1) + 14).attr("text-anchor", "end").text("in film order");
    const pts = D.shots.map((s, i) => [s, i]).filter(d => d[0][3] >= 0);
    scatter(svg, pts, "dot", ([s]) => x((s[0] + s[1]) / 2), ([s]) => y(D.film.story_pos[s[3]]), {
        move: (ev, [s, i]) => { hover({ trailer: i, film: s[3] }); showTip(ev, `<div class="pair">${img(tImg(i))}${img(fImg(s[3]))}</div><div>Shot ${i + 1}</div>`); },
        leave: unhover,
        click: (ev, [s]) => { preview = null; hideTip(); seek(s[0] + 0.01); },
    });
    axes(svg, x, y, w, H, m, fmtT, "trailer time", "film position");
}

init();