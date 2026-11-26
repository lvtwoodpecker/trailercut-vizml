const margin = { top: 30, right: 30, bottom: 30, left: 30 };
const width = 800 - margin.left - margin.right;
const height = 540 - margin.top - margin.bottom;

const svg = d3.select("#scatterplot")
    .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`);

// Background click rect to reset trailer selection when clicking empty space
svg.append("rect")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("fill", "transparent")
    .on("click", () => {
        resetSelection();
    });

const plotArea = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// Interactive pan/zoom surface inside plotArea
const zoomSurface = plotArea.append("rect")
    .attr("class", "zoom-surface")
    .attr("width", width)
    .attr("height", height)
    .attr("fill", "transparent")
    .style("cursor", "grab")
    .on("click", (event) => {
        // Ignore clicks right after a drag gesture
        if (event.defaultPrevented) return;
        resetSelection();
    });

const dotsGroup = plotArea.append("g")
    .attr("class", "dots-group");

const workspace = d3.select(".workspace");
const tooltip = d3.select("#tooltip");
const trailerMap = new Map();
let globalData = [];

let currentModality = "all";
let currentSpace = "res";
let currentGenre = "all";
let activeLegendKey = null;
let selectedImdbId = null;

// Base scales & current zoom transform (semantic zoom state)
let baseXScale = d3.scaleLinear().range([0, width]);
let baseYScale = d3.scaleLinear().range([height, 0]);
let currentTransform = d3.zoomIdentity;

let currentColorBy = "cluster";
const genreColorScale = d3.scaleOrdinal(d3.schemeTableau10);
const decadeColorScale = d3.scaleOrdinal(d3.schemeSet2);

function getClusterLabel(modality, clusterId, space = currentSpace) {
    const doc = modalityDocs[modality] || modalityDocs.all;
    const spaceLabels = doc.labels?.[space] || doc.labels?.res || doc.labels?.base || {};
    return spaceLabels[clusterId] ?? `Cluster ${clusterId}`;
}

function getDecadeLabel(d) {
    if (typeof d.decade === "string" && d.decade.trim() !== "" && d.decade !== "nan") {
        const trimmed = d.decade.trim();
        if (/^\d{4}s$/i.test(trimmed)) return trimmed;
        const num = Number(trimmed);
        if (Number.isFinite(num) && num > 0) {
            return num >= 1000 && num < 10 ? `${num}s` : `${Math.floor(num / 10) * 10}s`;
        }
    }
    if (Number.isFinite(d.decade) && d.decade > 0) {
        return d.decade >= 1000 ? `${Math.floor(d.decade / 10) * 10}s` : `${d.decade}s`;
    }
    if (Number.isFinite(d.year) && d.year > 0) {
        return `${Math.floor(d.year / 10) * 10}s`;
    }
    return "Unknown";
}

function getColorCategory(d) {
    if (currentColorBy === "genre") {
        const g = d.primary_genre && d.primary_genre !== "nan" ? d.primary_genre : "Unknown";
        return { key: g, label: g };
    }
    if (currentColorBy === "decade") {
        const dec = getDecadeLabel(d);
        return { key: dec, label: dec };
    }
    const doc = modalityDocs[currentModality] || modalityDocs.all;
    const c = getClusterValue(d);
    return {
        key: c,
        label: getClusterLabel(currentModality, c, currentSpace)
    };
}

function getDotColor(d) {
    const { key } = getColorCategory(d);
    if (currentColorBy === "genre") return genreColorScale(key);
    if (currentColorBy === "decade") return decadeColorScale(key);
    return colorScale(key);
}

const zoomBehavior = d3.zoom()
    .scaleExtent([1, 12])
    .translateExtent([[0, 0], [width, height]])
    .extent([[0, 0], [width, height]])
    .on("start", () => {
        zoomSurface.style("cursor", "grabbing");
    })
    .on("zoom", (event) => {
        currentTransform = event.transform;
        applySemanticZoom();
    })
    .on("end", () => {
        zoomSurface.style("cursor", "grab");
    });

zoomSurface.call(zoomBehavior);

// Axis labels: UMAP axes mean nothing by themselves, but in most views one feature runs almost straight
// along one of them. Each view is rotated (UMAP's rotation and flip are arbitrary) so x follows the
// feature it lines up with best, then an axis gets an arrow label only when a feature's rank correlation
// with it is at least AXIS_MIN_RHO. ?rotate=0 keeps the raw layout.
const AXIS_MIN_RHO = 0.6;
const ROTATE = new URLSearchParams(location.search).get("rotate") !== "0";
const CONCEPTS = ["explosion", "gun", "fight", "chase", "running", "kiss", "crying", "scream", "laugh", "crowd",
    "night", "day_outdoor", "creature", "space", "water", "city", "interior", "text_card", "logo", "blood"];
const FEATURE_GROUPS = {
    pacing: ["n_shots", "duration", "cuts_per_min", "shot_len_median", "shot_len_cv", "pace_q1", "pace_q2", "pace_q3", "pace_q4", "pace_accel"],
    audio: ["rms_mean_db", "rms_dynamic_range", "quiet_frac", "onset_rate", "centroid_mean", "rms_q1", "rms_q2", "rms_q3", "rms_q4"],
    story: ["match_rate", "story_mean", "story_median", "story_max", "story_std", "first_act_share", "final_act_share", "order_corr"],
    dialogue: ["dlg_words_per_shot", "dlg_sentiment_mean", "dlg_sentiment_std", "dlg_neg_share"],
    visual: ["scale_width_mean", ...CONCEPTS.map(c => `c_${c}_mean`)],
    speech: ["speech_share", "words_per_min", "lines_found", "sync_rate"]
};
FEATURE_GROUPS.all = Object.values(FEATURE_GROUPS).flat();
// what "more of this feature" looks like, in plain words
const AXIS_TEXT = {
    n_shots: "more shots", duration: "longer trailer", cuts_per_min: "faster cutting", shot_len_median: "longer shots",
    shot_len_cv: "more uneven cutting", pace_q1: "faster opening", pace_q2: "faster 2nd quarter", pace_q3: "faster 3rd quarter",
    pace_q4: "faster finish", pace_accel: "speeds up toward the end",
    rms_mean_db: "louder", rms_dynamic_range: "bigger loud/quiet swings", quiet_frac: "more quiet moments",
    onset_rate: "more hits and beats", centroid_mean: "brighter sound", rms_q1: "louder opening", rms_q2: "louder 2nd quarter",
    rms_q3: "louder 3rd quarter", rms_q4: "louder finish",
    match_rate: "more film footage", story_mean: "later in the film", story_median: "later in the film",
    story_max: "reaches deeper into the film", story_std: "spread across the film", first_act_share: "more first act",
    final_act_share: "more final act", order_corr: "keeps the film's order",
    dlg_words_per_shot: "more words per shot", dlg_sentiment_mean: "more upbeat lines", dlg_sentiment_std: "more mixed-mood lines",
    dlg_neg_share: "more negative lines",
    scale_width_mean: "wider shots", speech_share: "more talking", words_per_min: "faster talk", lines_found: "more film lines",
    sync_rate: "lines play over their own scene",
    c_day_outdoor_mean: "more daytime outdoors", c_night_mean: "more night scenes", c_text_card_mean: "more title cards",
    c_space_mean: "more outer space", c_kiss_mean: "more kissing", c_crying_mean: "more crying", c_scream_mean: "more screaming",
    c_laugh_mean: "more laughing", c_running_mean: "more running", c_interior_mean: "more interiors", c_city_mean: "more city"
};
const axisText = f => AXIS_TEXT[f] ?? `more ${f.replace(/^c_|_mean$/g, "").replace(/_/g, " ")}${f.startsWith("c_") ? "s" : ""}`;
const axisInfo = {};   // "space_modality" -> {x, y}: {feature, rho} or null

function ranks(v) {
    // average ranks over the finite values; NaN stays NaN
    const idx = v.map((x, i) => i).filter(i => Number.isFinite(v[i])).sort((a, b) => v[a] - v[b]);
    const r = new Array(v.length).fill(NaN);
    for (let i = 0; i < idx.length;) {
        let j = i;
        while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++;
        for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2;
        i = j + 1;
    }
    return r;
}

function pearson(a, b) {
    let n = 0, sa = 0, sb = 0;
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) { n++; sa += a[i]; sb += b[i]; }
    if (n < 10) return 0;
    const ma = sa / n, mb = sb / n;
    let ab = 0, aa = 0, bb = 0;
    for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) {
        const da = a[i] - ma, db = b[i] - mb; ab += da * db; aa += da * da; bb += db * db;
    }
    return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}

// the feature that best follows this coordinate (Spearman), as {feature, rho}
function bestFeature(coord, featRanks) {
    const rc = ranks(coord);
    let best = { feature: null, rho: 0 };
    for (const [f, rf] of featRanks) {
        const rho = pearson(rc, rf);
        if (Math.abs(rho) > Math.abs(best.rho)) best = { feature: f, rho };
    }
    return best;
}

function orientMaps(data) {
    const allRanks = new Map(FEATURE_GROUPS.all.filter(f => f in data[0]).map(f => [f, ranks(data.map(d => +d[f]))]));
    for (const m of Object.keys(FEATURE_GROUPS)) {
        const featRanks = [...allRanks].filter(([f]) => FEATURE_GROUPS[m].includes(f));
        for (const sp of ["base", "res"]) {
            const kx = `umap_${sp}_${m}_x`, ky = `umap_${sp}_${m}_y`;
            if (!data.every(d => Number.isFinite(d[kx]) && Number.isFinite(d[ky]))) continue;
            let x = data.map(d => d[kx]), y = data.map(d => d[ky]);
            if (ROTATE) {
                // try every angle (2 degree steps), keep the one whose x axis follows a feature most closely
                let bestA = 0, bestRho = 0;
                for (let deg = 0; deg < 180; deg += 2) {
                    const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
                    const rho = bestFeature(x.map((xi, i) => xi * c + y[i] * s), featRanks).rho;
                    if (Math.abs(rho) > Math.abs(bestRho)) { bestA = a; bestRho = rho; }
                }
                const c = Math.cos(bestA), s = Math.sin(bestA);
                const sx = Math.sign(bestRho) || 1;   // flip so the feature grows to the right
                [x, y] = [x.map((xi, i) => sx * (xi * c + y[i] * s)), y.map((yi, i) => sx * (-x[i] * s + yi * c))];
                if (bestFeature(y, featRanks).rho < 0) y = y.map(v => -v);   // and grows upward
                data.forEach((d, i) => { d[kx] = x[i]; d[ky] = y[i]; });
            }
            const keep = b => Math.abs(b.rho) >= AXIS_MIN_RHO ? b : null;
            axisInfo[`${sp}_${m}`] = { x: keep(bestFeature(x, featRanks)), y: keep(bestFeature(y, featRanks)) };
        }
    }
}

const axisLabels = svg.append("g").attr("class", "axis-labels");

function drawAxisLabels() {
    // single-group maps only: on Everything one feature out of ~60 would oversell what the map shows
    const info = currentModality === "all" ? {} : axisInfo[`${currentSpace}_${currentModality}`] || {};
    const left = margin.left, right = margin.left + width, top = margin.top, bottom = margin.top + height;
    const rows = [];
    if (info.x) {
        const up = info.x.rho > 0;   // the feature grows to the right
        rows.push({ key: "x", a: info.x, text: up ? `${axisText(info.x.feature)} →` : `← ${axisText(info.x.feature)}`,
            transform: `translate(${up ? right : left},${bottom + 20})`, anchor: up ? "end" : "start" });
    }
    if (info.y) {
        const up = info.y.rho > 0;   // the feature grows upward
        rows.push({ key: "y", a: info.y, text: up ? `${axisText(info.y.feature)} →` : `← ${axisText(info.y.feature)}`,
            transform: `translate(${left - 12},${up ? top : bottom}) rotate(-90)`, anchor: up ? "end" : "start" });
    }
    axisLabels.selectAll("text").data(rows, r => r.key).join("text")
        .attr("class", "axis-label")
        .attr("transform", r => r.transform)
        .attr("text-anchor", r => r.anchor)
        .text(r => r.text)
        .selectAll("title").data(r => [r]).join("title")
        .text(r => `rank correlation with this axis: ${Math.abs(r.a.rho).toFixed(2)}`);
}

// style colors: their own pair, apart from the site's more/less blue and red (--c1, --c2)
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const baseClusterColors = [cssVar("--c1") || "#199e70", cssVar("--c2") || "#8a63d2"];
const colorScale = d3.scaleOrdinal()
    .domain([0, 1])
    .range(baseClusterColors);

const modalityDocs = {
    all: {
        title: "Everything",
        desc: "Combines all features across modalities to capture overall trailer archetypes, and compares baseline vs. genre/decade-residualized structure.",
        labels: {
            base: {
                0: "Low-Action / Restrained",
                1: "High-Action / Thrill"
            },
            res: {
                0: "High-Action / Thrill",
                1: "Low-Action / Restrained"
            }
        }
    },
    pacing: {
        title: "Pacing",
        desc: "Captures editing tempo from shot cut rate, shot counts, and pacing across segments.",
        labels: {
            base: {
                0: "Slow Cuts / Sustained",
                1: "Fast Cuts / High Velocity"
            },
            res: {
                0: "Fast Cuts / High Velocity",
                1: "Slow Cuts / Sustained"
            }
        }
    },
    audio: {
        title: "Sound",
        desc: "Captures overall loudness (RMS), quiet fraction, and dynamic range swings across the trailer audio track.",
        labels: {
            base: {
                0: "Dynamic / Quiet-to-Loud",
                1: "Loud / Steady Mix"
            },
            res: {
                0: "Loud / Steady Mix",
                1: "Dynamic / Quiet-to-Loud"
            }
        }
    },
    story: {
        title: "Story",
        desc: "Reflects where matched story moments fall across the film's narrative arc (early setup vs. later/final act).",
        labels: {
            base: {
                0: "Early-Act / Setup-Leaning",
                1: "Late-Act / Climax-Leaning"
            },
            res: {
                0: "Late-Act / Climax-Leaning",
                1: "Early-Act / Setup-Leaning"
            }
        }
    },
    visual: {
        title: "Visuals",
        desc: "Captures visual concept activations across action, tension, and setting cues.",
        labels: {
            base: {
                0: "Low-Action / Grounded Cues",
                1: "Action & Intensity Cues"
            },
            res: {
                0: "Low-Action / Grounded Cues",
                1: "Action & Intensity Cues"
            }
        }
    },
    dialogue: {
        title: "Dialogue",
        desc: "Captures dialogue word density per shot alongside sentiment mean, negativity share, and emotional volatility.",
        labels: {
            base: {
                0: "Volatile / Wordy Dialogue",
                1: "Positive / Upbeat Dialogue",
                2: "Mildly Positive / Balanced",
                3: "Negative / Conflict-Heavy",
                4: "Sparse Lines per Shot",
                5: "High-Volume / High-Variance",
                6: "Flat / Low-Variance & Low-Neg",
                7: "Dark / Negative Tilt"
            },
            res: {
                0: "Steady / Low-Negativity",
                1: "Brief Lines per Shot",
                2: "Wordy & Emotionally Swinging",
                3: "High-Negativity Dialogue",
                4: "Positive & Wordy Dialogue"
            }
        }
    },
    speech: {
        title: "Speech",
        desc: "Captures spoken word rate, speech share, script line matching, and shot-to-speech synchronization.",
        labels: {
            base: {
                0: "High Speech Share, Few Matched Lines",
                1: "High Shot-Sync Speech",
                2: "Low Sync, Moderate Lines",
                3: "Low Speech / Sparse Talk",
                4: "Dense / High-Word Speech"
            },
            res: {
                0: "Minimal Speech",
                1: "Few Matched Script Lines",
                2: "High Line Count & Word Rate",
                3: "Fast Word Rate / High Share",
                4: "Out-of-Sync / Voiceover-Like",
                5: "Low Sync & Low Speech Share",
                6: "Many Matched Lines, Lower Share",
                7: "Strong Sync Alignment"
            }
        }
    }
};

const detailModalities = [
    { key: "all", label: "Everything" },
    { key: "pacing", label: "Pacing" },
    { key: "audio", label: "Sound" },
    { key: "story", label: "Story" },
    { key: "visual", label: "Visuals" },
    { key: "speech", label: "Speech" },
    { key: "dialogue", label: "Dialogue" }
];

const csvPaths = ["/data/features/clusters.csv", "data/features/clusters.csv"];

loadClustersCsv(csvPaths).then(data => {
    globalData = parseData(data);
    orientMaps(globalData);
    populateGenreDropdown();
    renderPlot(false);
    setupEventListeners();
}).catch(err => {
    console.error("Failed to load clusters CSV:", err);
});

function loadClustersCsv(paths) {
    return paths.reduce(
        (chain, path) => chain.catch(() => d3.csv(path)),
        Promise.reject(new Error("No CSV paths tried"))
    );
}

function parseData(data) {
    trailerMap.clear();

    data.forEach(d => {
        d.year = +d.year;
        d.cluster_all = +d.cluster_all;
        d.cluster_pacing = +d.cluster_pacing;
        d.cluster_audio = +d.cluster_audio;
        d.cluster_story = +d.cluster_story;
        d.cluster_visual = +d.cluster_visual;
        d.cluster_speech = +d.cluster_speech;
        d.cluster_dialogue = +d.cluster_dialogue;
        d.decade = (d.decade === "" || d.decade == null || d.decade === "nan") ? NaN : +d.decade;

        d.cluster_res_all = +d.cluster_res_all;
        d.cluster_res_pacing = +d.cluster_res_pacing;
        d.cluster_res_audio = +d.cluster_res_audio;
        d.cluster_res_story = +d.cluster_res_story;
        d.cluster_res_visual = +d.cluster_res_visual;
        d.cluster_res_speech = +d.cluster_res_speech;
        d.cluster_res_dialogue = +d.cluster_res_dialogue;

        const mods = ["all", "pacing", "audio", "story", "visual", "speech", "dialogue"];
        for (const m of mods) {
            for (const sp of ["base", "res"]) {
                const kx = `umap_${sp}_${m}_x`;
                const ky = `umap_${sp}_${m}_y`;
                if (d[kx] !== undefined && d[kx] !== "") d[kx] = +d[kx];
                if (d[ky] !== undefined && d[ky] !== "") d[ky] = +d[ky];
            }
        }
        // Keep legacy fallbacks just in case
        d.umap_x_base = +(d.umap_x_base ?? d.umap_base_all_x);
        d.umap_y_base = +(d.umap_y_base ?? d.umap_base_all_y);
        d.umap_x_res = +(d.umap_x_res ?? d.umap_res_all_x);
        d.umap_y_res = +(d.umap_y_res ?? d.umap_res_all_y);

        trailerMap.set(d.imdb_id, d);
    });

    return data;
}

function populateGenreDropdown() {
    const genres = Array.from(
        new Set(globalData.map(d => d.primary_genre).filter(Boolean))
    ).sort();

    d3.select("#genre-select")
        .selectAll("option.genre-option")
        .data(genres, d => d)
        .join("option")
        .attr("class", "genre-option")
        .attr("value", d => d)
        .text(d => d);
}

function getClusterValue(d, modality = currentModality, space = currentSpace) {
    const col = space === "res" ? `cluster_res_${modality}` : `cluster_${modality}`;
    const val = d[col];
    return val !== undefined && val !== null && val !== "" ? +val : null;
}

function updateColorDomain() {
    const doc = modalityDocs[currentModality] || modalityDocs.all;
    const spaceLabels = doc.labels?.[currentSpace] || doc.labels || {};
    const keys = Object.keys(spaceLabels).map(Number).sort((a, b) => a - b);
    const range = keys.length <= 2
        ? baseClusterColors.slice(0, keys.length)
        : d3.schemeTableau10.slice(0, keys.length);
    colorScale.domain(keys).range(range);
}

function getX(d, modality = currentModality, space = currentSpace) {
    const col = `umap_${space}_${modality}_x`;
    if (Number.isFinite(d[col])) return d[col];
    return space === "res" ? d.umap_x_res : d.umap_x_base;
}

function getY(d, modality = currentModality, space = currentSpace) {
    const col = `umap_${space}_${modality}_y`;
    if (Number.isFinite(d[col])) return d[col];
    return space === "res" ? d.umap_y_res : d.umap_y_base;
}

function getZoomedScales() {
    return {
        zx: currentTransform.rescaleX(baseXScale),
        zy: currentTransform.rescaleY(baseYScale)
    };
}

// Semantic zoom: rescale coordinates without scaling circle radius like a geometric transform
function applySemanticZoom() {
    const { zx, zy } = getZoomedScales();

    dotsGroup.selectAll("circle.dot")
        .attr("cx", d => zx(getX(d)))
        .attr("cy", d => zy(getY(d)));
}

function positionTooltip(event, svgWrapperNode) {
    const [px, py] = d3.pointer(event, svgWrapperNode);
    const wrapperRect = svgWrapperNode.getBoundingClientRect();
    const tooltipNode = tooltip.node();

    // Measure current tooltip size (fallback if first frame)
    const tipWidth = tooltipNode ? tooltipNode.offsetWidth : 180;
    const tipHeight = tooltipNode ? tooltipNode.offsetHeight : 44;

    const pad = 10;
    const offsetX = 12;
    const offsetY = 12;

    // Prefer right of cursor; flip left if it would overflow right edge
    let left = px + offsetX;
    if (left + tipWidth + pad > wrapperRect.width) {
        left = px - tipWidth - offsetX;
    }
    left = Math.max(pad, Math.min(left, wrapperRect.width - tipWidth - pad));

    // Prefer above cursor; flip below if it would overflow top edge
    let top = py - tipHeight - offsetY;
    if (top < pad) {
        top = py + offsetY;
    }
    top = Math.max(pad, Math.min(top, wrapperRect.height - tipHeight - pad));

    tooltip
        .style("left", `${left}px`)
        .style("top", `${top}px`);
}

function renderPlot(animate = true) {
    updateColorDomain();

    const xExtent = d3.extent(globalData, d => getX(d));
    const yExtent = d3.extent(globalData, d => getY(d));

    baseXScale = d3.scaleLinear().domain(xExtent).range([0, width]).nice();
    baseYScale = d3.scaleLinear().domain(yExtent).range([height, 0]).nice();

    const { zx, zy } = getZoomedScales();
    const svgWrapperNode = d3.select(".svg-wrapper").node();
    const t = d3.transition().duration(animate ? 500 : 0).ease(d3.easeCubicInOut);

    dotsGroup.selectAll("circle.dot")
        .data(globalData, d => d.imdb_id)
        .join(
            enter => enter.append("circle")
                .attr("class", "dot")
                .attr("cx", d => zx(getX(d)))
                .attr("cy", d => zy(getY(d)))
                .attr("r", 5)
                .attr("fill", d => getDotColor(d))
                .attr("opacity", 0.85)
                .on("mouseover", function (event, d) {
                    if (!filterPasses(d)) return;
                    tooltip
                        .html(`<strong>${d.title}</strong><br/><span style="color: var(--muted);">${d.primary_genre}</span>`)
                        .style("opacity", 1);
                    positionTooltip(event, svgWrapperNode);
                })
                .on("mousemove", function (event) {
                    if (!filterPasses(d3.select(this).datum())) return;
                    positionTooltip(event, svgWrapperNode);
                })
                .on("mouseout", function () {
                    tooltip.style("opacity", 0);
                })
                .on("click", function (event, d) {
                    event.stopPropagation();
                    selectTrailer(d);
                }),
            update => update
                .call(sel => sel.interrupt().transition(t)
                    .attr("cx", d => zx(getX(d)))
                    .attr("cy", d => zy(getY(d)))
                    .attr("fill", d => getDotColor(d))
                ),
            exit => exit.remove()
        );

    updateSelectionHighlight();
    applyFilters();
    updateLegendAndDocs();
    drawAxisLabels();
}

function getLegendKey(d) {
    if (currentColorBy === "genre") {
        return d.primary_genre && d.primary_genre !== "nan" ? d.primary_genre : "Unknown";
    }
    if (currentColorBy === "decade") {
        return getDecadeLabel(d);
    }
    return getClusterValue(d);
}

function filterPasses(d) {
    const searchTerm = d3.select("#search-input").property("value").trim().toLowerCase();
    const matchesGenre = currentGenre === "all" || d.primary_genre === currentGenre;
    const matchesSearch = !searchTerm || (d.title && d.title.toLowerCase().includes(searchTerm));
    const matchesLegend = activeLegendKey === null || getLegendKey(d) === activeLegendKey;
    return matchesGenre && matchesSearch && matchesLegend;
}

function applyFilters() {
    const searchTerm = d3.select("#search-input").property("value").trim().toLowerCase();

    dotsGroup.selectAll("circle.dot")
        .attr("opacity", d => {
            const isMatch = filterPasses(d);
            if (!isMatch) return 0.08;
            return selectedImdbId ? (d.imdb_id === selectedImdbId ? 1 : 0.25) : 0.85;
        })
        .attr("r", d => {
            if (!filterPasses(d)) return 3;
            if (d.imdb_id === selectedImdbId) return 9;
            return (searchTerm || activeLegendKey !== null) ? 6.5 : 5;
        })
        .style("pointer-events", d => filterPasses(d) ? "all" : "none");
}

function syncClusterProfile() {
    if (selectedImdbId) {
        renderClusterProfile(null);
        return;
    }

    if (activeLegendKey !== null) {
        renderClusterProfile({
            mode: currentColorBy,
            key: activeLegendKey
        });
        return;
    }

    renderClusterProfile(null);
}

function updateSelectionHighlight() {
    if (!selectedImdbId) {
        svg.classed("has-selection", false);
        dotsGroup.selectAll("circle.dot").classed("selected", false);
        return;
    }

    svg.classed("has-selection", true);
    dotsGroup.selectAll("circle.dot")
        .classed("selected", d => d.imdb_id === selectedImdbId)
        .filter(d => d.imdb_id === selectedImdbId)
        .raise();
}

function resetView({ resetProjection = false } = {}) {
    let needsPlotRecompute = false;

    // 1. Reset filter state & form inputs
    currentGenre = "all";
    activeLegendKey = null;
    d3.select("#genre-select").property("value", "all");
    d3.select("#search-input").property("value", "");

    // Optional: reset modality & projection space back to defaults
    if (resetProjection) {
        if (currentModality !== "all" || currentSpace !== "res" || currentColorBy !== "cluster") {
            currentModality = "all";
            currentSpace = "res";
            currentColorBy = "cluster";
            d3.select("#modality-select").property("value", "all");
            d3.select("#space-select").property("value", "res");
            d3.select("#color-by-select").property("value", "cluster");
            needsPlotRecompute = true;
        }
    }

    // 2. Clear selected point, drawer, and active video
    resetSelection();
    syncClusterProfile();

    // 3. Hide any lingering tooltip
    tooltip.style("opacity", 0);

    // 4. Recompute plot if modality/space changed, otherwise refresh legend & filters
    if (needsPlotRecompute) {
        renderPlot(true);
    } else {
        updateLegendAndDocs();
        applyFilters();
    }

    // 5. Reset zoom & pan back to identity
    zoomSurface
        .transition()
        .duration(400)
        .call(zoomBehavior.transform, d3.zoomIdentity);
}

function resetSelection() {
    selectedImdbId = null;
    workspace.classed("drawer-open", false);
    updateSelectionHighlight();
    applyFilters();
    syncClusterProfile();

    d3.select("#detail-title").text("Pick a trailer");
    d3.select("#detail-meta").text("");
    d3.select("#detail-clusters").selectAll("*").remove();
    d3.select("#imdb-link").style("display", "none");
    d3.select("#trailer-player").attr("src", "");
    d3.select("#trailer-player-container").style("display", "none");
    d3.select("#trailer-view-link").style("display", "none");
}

function updateLegendAndDocs() {
    const doc = modalityDocs[currentModality] || modalityDocs.all;
    d3.select("#info-heading").text(doc.title);
    d3.select("#info-description").text(doc.desc);

    let entries = [];
    let getItemColor = d => colorScale(d.key);

    if (currentColorBy === "cluster") {
        const spaceLabels = doc.labels?.[currentSpace] || {};
        const validIds = new Set(Object.keys(spaceLabels).map(Number));
        if (activeLegendKey !== null && !validIds.has(activeLegendKey)) {
            activeLegendKey = null;
        }
        d3.select("#legend-title").text(`${doc.title} Archetypes`);
        entries = Object.entries(spaceLabels).map(([clusterId, label]) => ({
            key: +clusterId,
            label
        }));
        getItemColor = d => colorScale(d.key);
    } else if (currentColorBy === "genre") {
        d3.select("#legend-title").text("Colored by Primary Genre");
        const uniqueGenres = Array.from(new Set(globalData.map(d => getColorCategory(d).key))).sort();
        entries = uniqueGenres.map(g => ({ key: g, label: g }));
        getItemColor = d => genreColorScale(d.key);
    } else if (currentColorBy === "decade") {
        d3.select("#legend-title").text("Colored by Decade");
        const uniqueDecades = Array.from(new Set(globalData.map(d => getColorCategory(d).key))).sort();
        entries = uniqueDecades.map(dec => ({ key: dec, label: dec }));
        getItemColor = d => decadeColorScale(d.key);
    }

    const items = d3.select("#legend-items")
        .selectAll("div.legend-item")
        .data(entries, d => d.key)
        .join(
            enter => {
                const row = enter.append("div")
                    .attr("class", "legend-item")
                    .style("cursor", "pointer");
                row.append("div").attr("class", "legend-color");
                row.append("span");
                return row;
            },
            update => update,
            exit => exit.remove()
        );

    items
        .classed("active", d => activeLegendKey === d.key)
        .classed("dimmed", d => activeLegendKey !== null && activeLegendKey !== d.key)
        .style("opacity", d => (activeLegendKey !== null && activeLegendKey !== d.key) ? 0.45 : 1)
        .style("font-weight", d => activeLegendKey === d.key ? "600" : "400")
        .on("click", function (event, d) {
            event.stopPropagation();
            activeLegendKey = activeLegendKey === d.key ? null : d.key;
            if (selectedImdbId !== null) {
                resetSelection();
            }
            updateLegendAndDocs();
            applyFilters();
            syncClusterProfile();
        });

    items.select(".legend-color")
        .style("background", d => getItemColor(d));

    items.select("span")
        .text(d => d.label);
}

function setupEventListeners() {
    d3.select("#modality-select").on("change", function () {
        currentModality = this.value;
        activeLegendKey = null;
        currentTransform = d3.zoomIdentity;
        zoomSurface.property("__zoom", d3.zoomIdentity);

        renderPlot(true);
        syncClusterProfile();

        if (selectedImdbId !== null && trailerMap.has(selectedImdbId)) {
            selectTrailer(trailerMap.get(selectedImdbId));
        }
    });

    d3.select("#space-select").on("change", function () {
        currentSpace = this.value;
        currentTransform = d3.zoomIdentity;
        zoomSurface.property("__zoom", d3.zoomIdentity);

        renderPlot(true);
        syncClusterProfile();
    });

    d3.select("#color-by-select").on("change", function () {
        currentColorBy = this.value;
        activeLegendKey = null;
        renderPlot(true);
        syncClusterProfile();
    });

    d3.select("#genre-select").on("change", function () {
        currentGenre = this.value;
        applyFilters();
    });

    d3.select("#search-input").on("input", function () {
        applyFilters();
    });

    d3.select("#reset-btn").on("click", function () {
        resetView({ resetProjection: true });
    });
}

function selectTrailer(item) {
    // const cId = getClusterValue(item, currentModality, currentSpace);
    // renderClusterProfile(cId);

    selectedImdbId = item.imdb_id;
    syncClusterProfile();
    workspace.classed("drawer-open", true);

    updateSelectionHighlight();
    applyFilters();

    d3.select("#detail-title").text(item.title.replace(/\s*\(\d{4}\)$/, ""));
    d3.select("#detail-meta").text(`${item.year}, ${item.primary_genre}`);

    const rowsData = detailModalities.map(m => {
        const cluster = getClusterValue(item, m.key, currentSpace);
        const description = getClusterLabel(m.key, cluster, currentSpace);
        return { ...m, description };
    });

    const rows = d3.select("#detail-clusters")
        .selectAll("div.cluster-row")
        .data(rowsData, d => d.key)
        .join(
            enter => {
                const row = enter.append("div")
                    .attr("class", "cluster-row")
                    .style("margin-top", "6px");
                row.append("strong");
                row.append("span").attr("class", "cluster-tag").style("margin-left", "6px");
                return row;
            },
            update => update,
            exit => exit.remove()
        );

    rows.select("strong").text(d => `${d.label}:`);
    rows.select("span.cluster-tag").text(d => d.description);

    d3.select("#imdb-link")
        .style("display", "inline-flex")
        .attr("href", `https://www.imdb.com/title/${item.imdb_id}/`);

    d3.select("#trailer-view-link")
        .style("display", "block")
        .attr("href", `trailerView.html?id=${encodeURIComponent(item.imdb_id)}`);

    if (item.youtube_id && item.youtube_id !== "" && item.youtube_id !== "undefined") {
        d3.select("#trailer-player")
            .attr("src", `https://www.youtube.com/embed/${item.youtube_id}`);
        d3.select("#trailer-player-container")
            .style("display", "block");
    } else {
        d3.select("#trailer-player").attr("src", "");
        d3.select("#trailer-player-container")
            .style("display", "none");
    }
}

function getTopCounts(items, accessor, limit = 3) {
    const counts = new Map();
    for (const item of items) {
        const raw = accessor(item);
        if (raw == null) continue;
        const val = String(raw).trim();
        if (!val || ["nan", "none", "null", "unknown"].includes(val.toLowerCase())) continue;
        counts.set(val, (counts.get(val) || 0) + 1);
    }
    return Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit);
}

function renderClusterProfile(target) {
    const container = d3.select("#cluster-profile");
    if (!target || target.key == null) {
        container.html("");
        return;
    }

    let members = [];
    let heading = "";
    let breakdownHtml = "";

    if (target.mode === "cluster") {
        const clusterId = +target.key;
        if (!Number.isFinite(clusterId)) {
            container.html("");
            return;
        }
        members = globalData.filter(d => getClusterValue(d, currentModality, currentSpace) === clusterId);
        heading = getClusterLabel(currentModality, clusterId, currentSpace);

        const topGenres = getTopCounts(members, d => d.primary_genre, 3)
            .map(([g, c]) => `${g} (${c})`)
            .join(", ");
        const topDecades = getTopCounts(members, d => getDecadeLabel(d), 2)
            .map(([dec, c]) => `${dec} (${c})`)
            .join(", ");

        breakdownHtml = `
            <div><strong>Top Genres:</strong> ${topGenres || "N/A"}</div>
            <div><strong>Top Decades:</strong> ${topDecades || "N/A"}</div>
        `;
    } else if (target.mode === "genre") {
        members = globalData.filter(d => {
            const g = d.primary_genre && d.primary_genre !== "nan" ? d.primary_genre : "Unknown";
            return g === target.key;
        });
        heading = `Genre: ${target.key}`;

        const topDecades = getTopCounts(members, d => getDecadeLabel(d), 2)
            .map(([dec, c]) => `${dec} (${c})`)
            .join(", ");
        const clusterSplit = getTopCounts(
            members,
            d => getClusterLabel(currentModality, getClusterValue(d, currentModality, currentSpace), currentSpace),
            3
        )
            .map(([lbl, c]) => `${lbl} (${c})`)
            .join(", ");

        breakdownHtml = `
            <div><strong>Top Decades:</strong> ${topDecades || "N/A"}</div>
            <div><strong>Archetype Split:</strong> ${clusterSplit || "N/A"}</div>
        `;
    } else if (target.mode === "decade") {
        members = globalData.filter(d => getDecadeLabel(d) === target.key);
        heading = `Decade: ${target.key}`;

        const topGenres = getTopCounts(members, d => d.primary_genre, 3)
            .map(([g, c]) => `${g} (${c})`)
            .join(", ");
        const clusterSplit = getTopCounts(
            members,
            d => getClusterLabel(currentModality, getClusterValue(d, currentModality, currentSpace), currentSpace),
            3
        )
            .map(([lbl, c]) => `${lbl} (${c})`)
            .join(", ");

        breakdownHtml = `
            <div><strong>Top Genres:</strong> ${topGenres || "N/A"}</div>
            <div><strong>Archetype Split:</strong> ${clusterSplit || "N/A"}</div>
        `;
    }

    const pct = globalData.length ? ((members.length / globalData.length) * 100).toFixed(1) : "0.0";

    container.html(`
        <div style="padding-top: 10px; border-top: 1px solid var(--border);">
            <div style="font-weight: 600; color: var(--ink); margin-bottom: 4px;">
                ${heading}
            </div>
            <div><strong>Size:</strong> ${members.length} trailers (${pct}% of dataset)</div>
            ${breakdownHtml}
        </div>
    `);
}