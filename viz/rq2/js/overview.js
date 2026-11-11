const margin = { top: 30, right: 30, bottom: 30, left: 30 };
const width = 800 - margin.left - margin.right;
const height = 540 - margin.top - margin.bottom;

const svg = d3.select("#scatterplot")
    .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`);

// Background click rect to reset selection when clicking empty space
svg.append("rect")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("fill", "transparent")
    .on("click", () => {
        resetSelection();
    });

const plotArea = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

const tooltip = d3.select("#tooltip");
const trailerMap = new Map();
let globalData = [];

let currentModality = "all";
let currentSpace = "res";
let currentGenre = "all";
let selectedImdbId = null;

const colorScale = d3.scaleOrdinal()
    .domain([0, 1])
    .range(["#f43f5e", "#3b82f6"]);

const modalityDocs = {
    all: {
        title: "Overall Composite (All)",
        desc: "Combines all features across modalities into a single UMAP projection to reveal overall trailer archetypes (Accelerated Montage vs. Sustained Narrative).",
        labels: { 0: "Cluster 0 (Accelerated Montage)", 1: "Cluster 1 (Sustained Narrative)" }
    },
    pacing: {
        title: "Pacing & Edit Velocity",
        desc: "Clustered using shot cut rate (cuts/min), shot duration median/CV, quarterly acceleration, and edit momentum.",
        labels: { 0: "Cluster 0 (High Cut Velocity)", 1: "Cluster 1 (Sustained Takes)" }
    },
    audio: {
        title: "Audio & Dynamic Range",
        desc: "Clustered using RMS energy (dB), dynamic range percentile differences, silence fraction, onset rate, and spectral centroid.",
        labels: { 0: "Cluster 0 (Music / Sound Dense)", 1: "Cluster 1 (Dialogue Dense)" }
    },
    story: {
        title: "Story & Narrative Alignment",
        desc: "Clustered using match rates with movie scenes, act distribution shares (First Act vs. Third Act), and Spearman order correlation.",
        labels: { 0: "Cluster 0 (Non-linear Sampling)", 1: "Cluster 1 (Linear Chronological)" }
    },
    dialogue: {
        title: "Dialogue & Subtitles",
        desc: "Clustered using VADER sentiment mean/std and negative sentiment shares extracted from matched movie subtitles.",
        labels: { 0: "Cluster 0 (Low Sentiment / Conflict)", 1: "Cluster 1 (Neutral / Warm)" }
    },
    visual: {
        title: "Visual Concept Scaling",
        desc: "Clustered using zero-shot CLIP visual concept activations across scale, motion, close-ups, and action tropes.",
        labels: { 0: "Cluster 0 (Action Tropes & Motion)", 1: "Cluster 1 (Grounded Staging)" }
    },
    speech: {
        title: "Speech Synchronization",
        desc: "Clustered using Whisper transcript speech share (words/min) and sync rate (lines playing over their native film shot).",
        labels: { 0: "Cluster 0 (Low Voiceover Density)", 1: "Cluster 1 (High Voiceover Density)" }
    }
};

const csvPath = "../../data/features/clusters.csv";

d3.csv(csvPath).then(data => {
    globalData = parseData(data);
    populateGenreDropdown();
    renderPlot();
    setupEventListeners();
}).catch(() => {
    d3.csv("data/features/clusters.csv").then(data => {
        globalData = parseData(data);
        populateGenreDropdown();
        renderPlot();
        setupEventListeners();
    });
});

function parseData(data) {
    data.forEach(d => {
        d.year = +d.year;
        d.cluster_all = +d.cluster_all;
        d.cluster_pacing = +d.cluster_pacing;
        d.cluster_audio = +d.cluster_audio;
        d.cluster_visual = +d.cluster_visual;
        d.cluster_res_all = +d.cluster_res_all;
        d.genre_list = (d.genres || "").split("|").filter(Boolean);  // every genre, IMDb lists them alphabetically

        d.umap_x_base = +d.umap_x_base;
        d.umap_y_base = +d.umap_y_base;
        d.umap_x_res = +d.umap_x_res;
        d.umap_y_res = +d.umap_y_res;

        trailerMap.set(d.imdb_id, d);
    });
    return data;
}

function populateGenreDropdown() {
    const genres = Array.from(new Set(globalData.flatMap(d => d.genre_list))).sort();
    const select = d3.select("#genre-select");

    genres.forEach(genre => {
        select.append("option").attr("value", genre).text(genre);
    });
}

function renderPlot() {
    plotArea.selectAll("*").remove();

    const xKey = currentSpace === "res" ? "umap_x_res" : "umap_x_base";
    const yKey = currentSpace === "res" ? "umap_y_res" : "umap_y_base";
    const clusterKey = currentSpace === "res" ? `cluster_res_${currentModality}` : `cluster_${currentModality}`;

    const xExtent = d3.extent(globalData, d => d[xKey]);
    const yExtent = d3.extent(globalData, d => d[yKey]);

    const xScale = d3.scaleLinear().domain(xExtent).range([0, width]).nice();
    const yScale = d3.scaleLinear().domain(yExtent).range([height, 0]).nice();

    const circles = plotArea.selectAll(".dot")
        .data(globalData)
        .enter()
        .append("circle")
        .attr("class", "dot")
        .attr("cx", d => xScale(d[xKey]))
        .attr("cy", d => yScale(d[yKey]))
        .attr("r", 5)
        .attr("fill", d => colorScale(d[clusterKey] !== undefined ? d[clusterKey] : d.cluster_all))
        .attr("opacity", 0.85);

    const vizContainerNode = d3.select("#viz-container").node();
    circles
        .on("mouseover", (event, d) => {
            if (!filterPasses(d)) return;

            // Get mouse coordinates relative to #viz-container
            const [x, y] = d3.pointer(event, vizContainerNode);

            tooltip.style("opacity", 1)
                .html(`<strong>${d.title}</strong><br/><span style="color: #94a3b8;">${d.genre_list.join(", ")}</span>`)
                .style("left", (x + 8) + "px")
                .style("top", (y - 80) + "px");
        })
        .on("mousemove", (event) => {
            const [x, y] = d3.pointer(event, vizContainerNode);

            tooltip
                .style("left", (x + 8) + "px")
                .style("top", (y - 80) + "px");
        })
        .on("mouseout", () => {
            tooltip.style("opacity", 0);
        })
        .on("click", (event, d) => {
            event.stopPropagation();
            window.dispatchEvent(new CustomEvent("trailerSelect", { detail: { imdb_id: d.imdb_id } }));
        });

    updateSelectionHighlight();
    applyFilters();
    updateLegendAndDocs();
}

function filterPasses(d) {
    const searchTerm = d3.select("#search-input").property("value").toLowerCase();
    const matchesGenre = currentGenre === "all" || d.genre_list.includes(currentGenre);
    const matchesSearch = !searchTerm || d.title.toLowerCase().includes(searchTerm);
    return matchesGenre && matchesSearch;
}

function applyFilters() {
    const searchTerm = d3.select("#search-input").property("value").toLowerCase();

    plotArea.selectAll(".dot")
        .attr("opacity", d => {
            const isMatch = filterPasses(d);
            if (!isMatch) return 0.05;
            return selectedImdbId ? (d.imdb_id === selectedImdbId ? 1 : 0.25) : 0.85;
        })
        .attr("r", d => filterPasses(d) ? (d.imdb_id === selectedImdbId ? 9 : (searchTerm ? 7 : 5)) : 3)
        .style("pointer-events", d => filterPasses(d) ? "all" : "none");
}

function updateSelectionHighlight() {
    if (!selectedImdbId) {
        svg.classed("has-selection", false);
        plotArea.selectAll(".dot").classed("selected", false);
        return;
    }

    svg.classed("has-selection", true);
    plotArea.selectAll(".dot")
        .classed("selected", d => d.imdb_id === selectedImdbId)
        .filter(d => d.imdb_id === selectedImdbId)
        .raise();
}

function resetSelection() {
    selectedImdbId = null;
    updateSelectionHighlight();
    applyFilters();

    d3.select("#detail-title").text("Select a trailer to begin...");
    d3.select("#detail-meta").text("");
    d3.select("#detail-clusters").html("");
    d3.select("#imdb-link").style("display", "none");
    d3.select("#tv-link").style("display", "none");
    d3.select("#trailer-player").attr("src", "");
    d3.select("#trailer-player-container").style("display", "none");
    d3.select("#trailer-link").style("display", "none");
}

function updateLegendAndDocs() {
    const doc = modalityDocs[currentModality] || modalityDocs.all;

    d3.select("#info-heading").text(doc.title);
    d3.select("#info-description").text(doc.desc);

    d3.select("#legend-title").text(`${doc.title} Archetypes`);

    const legendHtml = `
    <div class="legend-item">
      <div class="legend-color" style="background: ${colorScale(0)};"></div>
      <span>${doc.labels[0]}</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background: ${colorScale(1)};"></div>
      <span>${doc.labels[1]}</span>
    </div>
  `;
    d3.select("#legend-items").html(legendHtml);
}

function setupEventListeners() {
    d3.select("#modality-select").on("change", function () {
        currentModality = this.value;
        renderPlot();
    });

    d3.select("#space-select").on("change", function () {
        currentSpace = this.value;
        renderPlot();
    });

    d3.select("#genre-select").on("change", function () {
        currentGenre = this.value;
        applyFilters();
    });

    d3.select("#search-input").on("input", function () {
        const term = this.value.toLowerCase();
        plotArea.selectAll(".dot")
            .attr("opacity", d => d.title.toLowerCase().includes(term) ? 0.9 : 0.1)
            .attr("r", d => d.title.toLowerCase().includes(term) ? (term ? 7 : 5) : 3);
    });
}

window.addEventListener("trailerSelect", (event) => {
    const imdbId = event.detail.imdb_id;
    const item = trailerMap.get(imdbId);
    if (!item) return;

    selectedImdbId = imdbId;
    updateSelectionHighlight();
    applyFilters();

    d3.select("#detail-title").text(item.title);
    d3.select("#detail-meta").text(`${item.year} • ${item.genre_list.join(", ")}`);

    const activeCluster = currentSpace === "res" ? item.cluster_res_all : item.cluster_all;

    const clusterHTML = `
    <div><strong>Overall Archetype:</strong> <span class="cluster-tag">${activeCluster === 0 ? "Accelerated Montage" : "Sustained Narrative"}</span></div>
    <div style="margin-top: 6px;"><strong>Pacing:</strong> <span class="cluster-tag">${item.cluster_pacing === 0 ? "Fast Velocity" : "Sustained Takes"}</span></div>
    <div style="margin-top: 6px;"><strong>Audio:</strong> <span class="cluster-tag">${item.cluster_audio === 0 ? "Music / Sound Dense" : "Dialogue Dense"}</span></div>
    <div style="margin-top: 6px;"><strong>Visual:</strong> <span class="cluster-tag">${item.cluster_visual === 0 ? "Action Tropes" : "Grounded Staging"}</span></div>
  `;
    d3.select("#detail-clusters").html(clusterHTML);

    d3.select("#imdb-link")
        .style("display", "inline-flex")
        .attr("href", `https://www.imdb.com/title/${item.imdb_id}/`);

    d3.select("#tv-link")
        .style("display", "inline-flex")
        .attr("href", `../index.html?id=${item.imdb_id}`);

    const playerContainer = d3.select("#trailer-player-container");
    const trailerBtn = d3.select("#trailer-link").style("display", "block");

    if (item.youtube_id && item.youtube_id !== "" && item.youtube_id !== "undefined") {
        d3.select("#trailer-player")
            .attr("src", `https://www.youtube.com/embed/${item.youtube_id}?enablejsapi=1`);
        playerContainer.style("display", "block");
    } else {
        d3.select("#trailer-player").attr("src", "");
        playerContainer.style("display", "none");

        const searchQuery = encodeURIComponent(`${item.title} ${item.year} official trailer`);
        trailerBtn
            .attr("href", `https://www.youtube.com/results?search_query=${searchQuery}`)
            .text("Search YouTube ↗");
    }
});