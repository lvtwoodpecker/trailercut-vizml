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

// UMAP axes have no units: say what the layout means instead of ticking them
svg.append("text").attr("class", "axis-note")
    .attr("x", margin.left).attr("y", height + margin.top + margin.bottom - 8)
    .text("Similarity map (UMAP): nearby trailers have similar features. The axes have no units.");

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

const baseClusterColors = ["#f43f5e", "#3b82f6"];
const colorScale = d3.scaleOrdinal()
    .domain([0, 1])
    .range(baseClusterColors);

const modalityDocs = {
    all: {
        title: "Overall Composite (All)",
        desc: "Combines all features across modalities to capture overall trailer archetypes—and compares baseline vs. genre/decade-residualized structure.",
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
        title: "Pacing & Edit Velocity",
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
        title: "Audio & Dynamic Range",
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
        title: "Story & Narrative Alignment",
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
        title: "Visual Concept Scaling",
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
        title: "Dialogue & Sentiment",
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
        title: "Speech Synchronization",
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
    { key: "all", label: "Overall Archetype" },
    { key: "pacing", label: "Pacing" },
    { key: "audio", label: "Audio" },
    { key: "story", label: "Story" },
    { key: "visual", label: "Visual" },
    { key: "speech", label: "Speech" },
    { key: "dialogue", label: "Dialogue" }
];

const csvPaths = ["/data/features/clusters.csv", "data/features/clusters.csv"];

loadClustersCsv(csvPaths).then(data => {
    globalData = parseData(data);
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
                        .html(`<strong>${d.title}</strong><br/><span style="color: #94a3b8;">${d.primary_genre}</span>`)
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

    d3.select("#detail-title").text("Select a trailer to begin...");
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
    // what an archetype is, and what the projection choice does
    d3.select("#archetype-note").text(
        "Archetypes are groups k-means finds in these features; each name describes its group's average trailer. " +
        (currentSpace === "res"
            ? "Residualized: genre and decade are regressed out first, so the groups show style that genre and era don't explain."
            : "Baseline: raw features, so genre and decade still shape the groups."));

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
        .attr("href", `/viz/trailerView.html?id=${encodeURIComponent(item.imdb_id)}`);

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