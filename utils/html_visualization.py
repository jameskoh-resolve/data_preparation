"""HTML Visualization generator for Auto-Annotate curation pipeline.

Generates a standalone, interactive HTML visualization gallery for detection and LLM validation outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__TITLE__</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .header h1 {
            font-size: 16px;
            white-space: nowrap;
            color: #f8fafc;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        .header .stats {
            font-size: 12px;
            color: #94a3b8;
            background: #0f172a;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid #334155;
        }
        .controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: auto;
            flex-wrap: wrap;
        }
        .controls input[type="text"] {
            background: #334155;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 6px 12px;
            color: #e2e8f0;
            font-size: 12px;
            width: 240px;
            outline: none;
        }
        .controls input[type="text"]:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }
        .controls select {
            background: #334155;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 6px 10px;
            color: #e2e8f0;
            font-size: 12px;
            outline: none;
            cursor: pointer;
        }
        .controls select:focus {
            border-color: #3b82f6;
        }
        .toggle-group {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #0f172a;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid #334155;
        }
        .controls label {
            font-size: 12px;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            user-select: none;
            padding: 2px 6px;
            border-radius: 4px;
            transition: background 0.15s;
        }
        .controls label:hover {
            background: #334155;
        }
        .controls label input[type="checkbox"] {
            accent-color: #3b82f6;
            cursor: pointer;
            width: 14px;
            height: 14px;
        }
        .controls label.llm-toggle {
            font-weight: 600;
            color: #60a5fa;
            border-left: 1px solid #334155;
            padding-left: 8px;
            margin-left: 2px;
        }
        .nav-controls {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .nav-controls button {
            padding: 5px 12px;
            border-radius: 6px;
            border: 1px solid #475569;
            background: #334155;
            color: #e2e8f0;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: background 0.15s;
        }
        .nav-controls button:hover { background: #475569; }
        .nav-controls input {
            width: 45px;
            background: #334155;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 4px;
            color: #e2e8f0;
            font-size: 12px;
            text-align: center;
        }
        .grid-view {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 16px;
            padding: 20px;
        }
        .grid-item {
            background: #1e293b;
            border-radius: 10px;
            border: 1px solid #334155;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
            position: relative;
            display: flex;
            flex-direction: column;
        }
        .grid-item:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        }
        .grid-item .image-container {
            position: relative;
            width: 100%;
            background: #090d16;
            min-height: 260px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .grid-item img {
            width: 100%;
            height: auto;
            max-height: 460px;
            object-fit: contain;
            display: block;
        }
        .grid-item svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .grid-item .info {
            padding: 10px 12px;
            font-size: 11px;
            border-top: 1px solid #334155;
            background: #1e293b;
            margin-top: auto;
        }
        .grid-item .info .im-id {
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 4px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .grid-item .info .concepts {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 6px;
        }
        .concept-tag {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            color: #ffffff;
        }
        .concept-tag.rejected {
            opacity: 0.55;
            text-decoration: line-through;
            border: 1px dashed rgba(239, 68, 68, 0.8);
            background: #450a0a !important;
            color: #fca5a5 !important;
        }
        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.96);
            z-index: 200;
            flex-direction: column;
        }
        .modal-overlay.active { display: flex; }
        .modal-header {
            padding: 12px 20px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .modal-header h3 {
            font-size: 14px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: #f8fafc;
        }
        .modal-header .close-btn {
            margin-left: auto;
            background: #dc2626;
            border: none;
            color: white;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: background 0.15s;
        }
        .modal-header .close-btn:hover { background: #b91c1c; }
        .modal-body {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .modal-image-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 20px;
            background: #090d16;
        }
        .modal-img-wrapper {
            position: relative;
            max-width: 88vw;
            max-height: 82vh;
        }
        .modal-img-wrapper img {
            max-width: 88vw;
            max-height: 82vh;
            object-fit: contain;
            display: block;
            border-radius: 6px;
        }
        .modal-img-wrapper svg {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
        }
        .modal-sidebar {
            width: 360px;
            background: #1e293b;
            border-left: 1px solid #334155;
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .modal-sidebar h4 {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .box-detail-card {
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid #334155;
            padding: 12px;
            font-size: 11px;
            transition: border-color 0.15s;
        }
        .box-detail-card.rejected {
            border-color: #ef4444;
            background: rgba(239, 68, 68, 0.08);
        }
        .box-detail-card.validated {
            border-color: #22c55e;
            background: rgba(34, 197, 94, 0.08);
        }
        .box-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 6px;
        }
        .status-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        }
        .status-badge.valid { background: #15803d; color: #ffffff; }
        .status-badge.invalid { background: #b91c1c; color: #ffffff; }
        .status-badge.unvalidated { background: #475569; color: #cbd5e1; }
        .box-coords {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: #94a3b8;
            font-size: 10px;
            margin-bottom: 4px;
        }
        .box-reason {
            color: #cbd5e1;
            margin-top: 6px;
            line-height: 1.4;
            background: rgba(0, 0, 0, 0.2);
            padding: 6px 8px;
            border-radius: 4px;
            border-left: 2px solid #3b82f6;
        }
        .box-reason.invalid-reason {
            border-left-color: #ef4444;
        }
        .legend {
            position: fixed;
            bottom: 12px; right: 12px;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 11px;
            z-index: 50;
            max-height: 50vh;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .legend-title { font-weight: 700; color: #94a3b8; margin-bottom: 6px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
        .legend-item { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
        .legend-swatch { width: 14px; height: 4px; border-radius: 2px; }
        .empty-state {
            grid-column: 1 / -1;
            padding: 60px 20px;
            text-align: center;
            color: #64748b;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>__TITLE__</h1>
        <div class="stats" id="stats">Loading data...</div>
        <div class="controls">
            <input type="text" id="searchInput" placeholder="Filter by image ID, URL, concept, or reason..." oninput="applyFilters()">
            <select id="conceptFilter" onchange="applyFilters()">
                <option value="">All concepts</option>
            </select>
            <select id="llmStatusFilter" onchange="applyFilters()">
                <option value="all">All detections</option>
                <option value="valid_only">LLM Validated Only (✓)</option>
                <option value="invalid_only">LLM Rejected Only (✕)</option>
            </select>
            <div class="toggle-group">
                <label><input type="checkbox" id="showBoxes" checked onchange="renderCurrentPage()"> Boxes</label>
                <label><input type="checkbox" id="showLabels" checked onchange="renderCurrentPage()"> Labels</label>
                <label class="llm-toggle" title="Toggle ON to filter out LLM-rejected boxes. Toggle OFF to view all raw candidate boxes.">
                    <input type="checkbox" id="useLlmValidation" checked onchange="applyFilters()"> LLM Filtered
                </label>
            </div>
        </div>
        <div class="nav-controls">
            <button onclick="changePage(-1)">← Prev</button>
            <input type="number" id="pageInput" value="1" min="1" onchange="jumpToPage()">
            <span id="pageInfo" style="font-size:11px;color:#94a3b8;">/1</span>
            <button onclick="changePage(1)">Next →</button>
        </div>
    </div>

    <div class="grid-view" id="gridView"></div>

    <!-- Modal detail viewer -->
    <div class="modal-overlay" id="modal">
        <div class="modal-header">
            <h3 id="modalTitle">Image Details</h3>
            <div style="font-size: 11px; color: #94a3b8;" id="modalMeta"></div>
            <button class="close-btn" onclick="closeModal()">✕ Close</button>
        </div>
        <div class="modal-body">
            <div class="modal-image-area">
                <div class="modal-img-wrapper" id="modalImgWrapper">
                    <img id="modalImage" alt="Modal Detail">
                    <svg id="modalSvg"></svg>
                </div>
            </div>
            <div class="modal-sidebar">
                <h4>Bounding Box Details</h4>
                <div id="modalBoxList"></div>
            </div>
        </div>
    </div>

    <div class="legend" id="legend">
        <div class="legend-title">Classes</div>
        <div id="legendList"></div>
    </div>

    <script>
    const DATA = __DATA_JSON__;

    const CONCEPT_COLORS = {
        top: '#22c55e',
        bottom: '#3b82f6',
        dress: '#f43f5e',
        skirt: '#d946ef',
        outerwear: '#8b5cf6',
        innertop: '#06b6d4',
        ethnic_wear: '#f97316',
        shoe: '#64748b',
        bag: '#ec4899',
        belt: '#f59e0b',
        tie: '#818cf8',
        gloves: '#2dd4bf',
        scarf: '#14b8a6',
        eyewear: '#38bdf8',
        headwear: '#a3e635',
        hair_accessories: '#c084fc',
        earring: '#f472b6',
        necklace: '#a78bfa',
        bracelet: '#fb923c',
        watch: '#fbbf24',
        ring: '#eab308',
        head_jewelry: '#fda4af',
        anklet: '#0ea5e9'
    };

    function getColor(concept) {
        if (!concept) return '#94a3b8';
        return CONCEPT_COLORS[concept.toLowerCase()] || '#94a3b8';
    }

    let filteredData = DATA;
    let currentPage = 1;
    const PAGE_SIZE = 50;
    let activeModalItemIndex = -1;

    function init() {
        populateFilters();
        updateStats();
        applyFilters();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
            if (document.getElementById('modal').classList.contains('active')) {
                if (e.key === 'ArrowLeft') stepModal(-1);
                if (e.key === 'ArrowRight') stepModal(1);
            }
        });
    }

    function populateFilters() {
        const conceptsSet = new Set();
        DATA.forEach(item => {
            (item.detections || []).forEach(d => {
                if (d.name) conceptsSet.add(d.name.toLowerCase());
            });
        });

        const conceptSelect = document.getElementById('conceptFilter');
        [...conceptsSet].sort().forEach(c => {
            const opt = document.createElement('option');
            opt.value = c;
            opt.textContent = c;
            conceptSelect.appendChild(opt);
        });

        const legendList = document.getElementById('legendList');
        [...conceptsSet].sort().forEach(c => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            const color = getColor(c);
            item.innerHTML = `<div class="legend-swatch" style="background:${color};"></div><span>${c}</span>`;
            legendList.appendChild(item);
        });
    }

    function updateStats() {
        let totalDets = 0;
        let validDets = 0;
        let rejectedDets = 0;

        DATA.forEach(item => {
            (item.detections || []).forEach(d => {
                totalDets++;
                if (d.is_valid === true) validDets++;
                if (d.is_valid === false) rejectedDets++;
            });
        });

        const passRate = totalDets > 0 ? ((validDets / totalDets) * 100).toFixed(1) : '100.0';
        document.getElementById('stats').textContent =
            `${DATA.length.toLocaleString()} images | ${totalDets.toLocaleString()} candidate boxes | ${validDets.toLocaleString()} LLM validated (${passRate}% pass rate)`;
    }

    function getDetectionsForItem(item, useLlmFilter) {
        const dets = item.detections || [];
        if (!useLlmFilter) {
            return dets; // Show all candidate detections
        }
        // Show only boxes where is_valid is not false
        return dets.filter(d => d.is_valid !== false);
    }

    function applyFilters() {
        const searchVal = document.getElementById('searchInput').value.toLowerCase().trim();
        const conceptVal = document.getElementById('conceptFilter').value.toLowerCase();
        const llmStatusVal = document.getElementById('llmStatusFilter').value;
        const useLlmFilter = document.getElementById('useLlmValidation').checked;

        filteredData = DATA.filter(item => {
            const activeDets = getDetectionsForItem(item, useLlmFilter);

            if (conceptVal) {
                const hasConcept = activeDets.some(d => (d.name || '').toLowerCase() === conceptVal);
                if (!hasConcept) return false;
            }

            if (llmStatusVal === 'valid_only') {
                const hasValid = activeDets.some(d => d.is_valid === true);
                if (!hasValid) return false;
            } else if (llmStatusVal === 'invalid_only') {
                const hasInvalid = (item.detections || []).some(d => d.is_valid === false);
                if (!hasInvalid) return false;
            }

            if (searchVal) {
                const matchId = (item.im_id || '').toLowerCase().includes(searchVal);
                const matchUrl = (item.im_url || '').toLowerCase().includes(searchVal);
                const matchConcept = (item.detections || []).some(d => (d.name || '').toLowerCase().includes(searchVal));
                const matchReason = (item.detections || []).some(d => (d.reason || '').toLowerCase().includes(searchVal));
                if (!matchId && !matchUrl && !matchConcept && !matchReason) return false;
            }

            return true;
        });

        currentPage = 1;
        renderCurrentPage();
    }

    function renderCurrentPage() {
        const totalPages = Math.max(1, Math.ceil(filteredData.length / PAGE_SIZE));
        if (currentPage > totalPages) currentPage = totalPages;

        document.getElementById('pageInput').value = currentPage;
        document.getElementById('pageInput').max = totalPages;
        document.getElementById('pageInfo').textContent = `/ ${totalPages}`;

        const startIdx = (currentPage - 1) * PAGE_SIZE;
        const endIdx = Math.min(filteredData.length, startIdx + PAGE_SIZE);
        const pageItems = filteredData.slice(startIdx, endIdx);

        const gridView = document.getElementById('gridView');
        gridView.innerHTML = '';

        if (pageItems.length === 0) {
            gridView.innerHTML = '<div class="empty-state">No matching images found.</div>';
            return;
        }

        const useLlmFilter = document.getElementById('useLlmValidation').checked;
        const showBoxes = document.getElementById('showBoxes').checked;
        const showLabels = document.getElementById('showLabels').checked;

        pageItems.forEach((item, pageOffset) => {
            const globalIdx = startIdx + pageOffset;
            const itemEl = document.createElement('div');
            itemEl.className = 'grid-item';
            itemEl.onclick = () => openModal(globalIdx);

            const activeDets = getDetectionsForItem(item, useLlmFilter);

            const conceptsHtml = activeDets.map(d => {
                const color = getColor(d.name);
                const isRejected = d.is_valid === false;
                const cls = isRejected ? 'concept-tag rejected' : 'concept-tag';
                const labelText = isRejected ? `✕ ${d.name}` : d.name;
                return `<span class="${cls}" style="background:${color}">${labelText}</span>`;
            }).join('');

            itemEl.innerHTML = `
                <div class="image-container">
                    <img src="${item.im_url}" alt="${item.im_id}" loading="lazy" onload="drawCardSvg(this)" onerror="this.onerror=null;this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect width=%22100%22 height=%22100%22 fill=%22%231e293b%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%2364748b%22 text-anchor=%22middle%22 font-size=%2210%22>Image Load Error</text></svg>';">
                    <svg class="card-svg" data-idx="${globalIdx}"></svg>
                </div>
                <div class="info">
                    <div class="im-id">
                        <span>#${globalIdx + 1} · ${item.im_id.slice(0, 12)}</span>
                        <span style="color:#64748b;font-weight:400;">${activeDets.length} boxes</span>
                    </div>
                    <div class="concepts">${conceptsHtml || '<span style="color:#64748b;font-style:italic;">No boxes</span>'}</div>
                </div>
            `;

            gridView.appendChild(itemEl);
        });

        // Trigger SVG drawing for visible cards
        document.querySelectorAll('.card-svg').forEach(svg => {
            const idx = parseInt(svg.dataset.idx, 10);
            const item = filteredData[idx];
            if (item) {
                const img = svg.previousElementSibling;
                if (img && img.complete && img.naturalWidth > 0) {
                    drawSvgBoxes(svg, img, item, useLlmFilter, showBoxes, showLabels);
                }
            }
        });
    }

    function drawCardSvg(img) {
        const svg = img.nextElementSibling;
        if (!svg) return;
        const idx = parseInt(svg.dataset.idx, 10);
        const item = filteredData[idx];
        if (item) {
            const useLlmFilter = document.getElementById('useLlmValidation').checked;
            const showBoxes = document.getElementById('showBoxes').checked;
            const showLabels = document.getElementById('showLabels').checked;
            drawSvgBoxes(svg, img, item, useLlmFilter, showBoxes, showLabels);
        }
    }

    function drawSvgBoxes(svgEl, imgEl, item, useLlmFilter, showBoxes, showLabels) {
        svgEl.innerHTML = '';
        if (!showBoxes) return;

        const nw = imgEl.naturalWidth || 800;
        const nh = imgEl.naturalHeight || 800;
        svgEl.setAttribute('viewBox', `0 0 ${nw} ${nh}`);

        const dets = getDetectionsForItem(item, useLlmFilter);

        dets.forEach(det => {
            const box = det.box;
            if (!box || box.length < 4) return;

            const ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];
            const w = xmax - xmin;
            const h = ymax - ymin;
            const color = getColor(det.name);
            const isRejected = det.is_valid === false;

            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', xmin);
            rect.setAttribute('y', ymin);
            rect.setAttribute('width', w);
            rect.setAttribute('height', h);
            rect.setAttribute('fill', isRejected ? 'rgba(239, 68, 68, 0.08)' : `${color}22`);
            rect.setAttribute('stroke', isRejected ? '#ef4444' : color);
            rect.setAttribute('stroke-width', Math.max(2, Math.round(nw / 400)));
            if (isRejected) {
                rect.setAttribute('stroke-dasharray', '6,4');
            }
            svgEl.appendChild(rect);

            if (showLabels) {
                const textGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                const fontSize = Math.max(12, Math.round(nw / 42));

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', xmin + 4);
                text.setAttribute('y', ymin + fontSize);
                text.setAttribute('fill', '#ffffff');
                text.setAttribute('font-size', fontSize);
                text.setAttribute('font-weight', '600');
                text.setAttribute('font-family', 'sans-serif');
                text.textContent = isRejected ? `✕ ${det.name}` : det.name;

                const textBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                textBg.setAttribute('x', xmin);
                textBg.setAttribute('y', ymin);
                textBg.setAttribute('height', fontSize + 6);
                textBg.setAttribute('fill', isRejected ? '#dc2626' : color);
                textBg.setAttribute('rx', 3);

                textGroup.appendChild(textBg);
                textGroup.appendChild(text);
                svgEl.appendChild(textGroup);

                setTimeout(() => {
                    try {
                        const bb = text.getBBox();
                        textBg.setAttribute('width', bb.width + 8);
                    } catch(e){}
                }, 0);
            }
        });
    }

    function changePage(delta) {
        const totalPages = Math.max(1, Math.ceil(filteredData.length / PAGE_SIZE));
        currentPage += delta;
        if (currentPage < 1) currentPage = 1;
        if (currentPage > totalPages) currentPage = totalPages;
        renderCurrentPage();
    }

    function jumpToPage() {
        const val = parseInt(document.getElementById('pageInput').value, 10);
        if (!isNaN(val)) {
            const totalPages = Math.max(1, Math.ceil(filteredData.length / PAGE_SIZE));
            currentPage = Math.max(1, Math.min(totalPages, val));
            renderCurrentPage();
        }
    }

    function openModal(globalIdx) {
        activeModalItemIndex = globalIdx;
        const item = filteredData[globalIdx];
        if (!item) return;

        const modal = document.getElementById('modal');
        const modalImg = document.getElementById('modalImage');
        const modalSvg = document.getElementById('modalSvg');

        document.getElementById('modalTitle').textContent = `[${globalIdx + 1}/${filteredData.length}] ID: ${item.im_id}`;
        document.getElementById('modalMeta').innerHTML = `<a href="${item.im_url}" target="_blank" style="color:#60a5fa;text-decoration:none;">Open Image Link ↗</a>`;

        modalImg.src = item.im_url;
        modal.classList.add('active');

        modalImg.onload = () => {
            const useLlmFilter = document.getElementById('useLlmValidation').checked;
            const showBoxes = document.getElementById('showBoxes').checked;
            const showLabels = document.getElementById('showLabels').checked;
            drawSvgBoxes(modalSvg, modalImg, item, useLlmFilter, showBoxes, showLabels);
        };

        renderModalSidebar(item);
    }

    function renderModalSidebar(item) {
        const sidebar = document.getElementById('modalBoxList');
        sidebar.innerHTML = '';

        const useLlmFilter = document.getElementById('useLlmValidation').checked;
        const dets = getDetectionsForItem(item, useLlmFilter);

        if (dets.length === 0) {
            sidebar.innerHTML = '<div style="color:#64748b;font-style:italic;">No bounding boxes for current filters.</div>';
            return;
        }

        dets.forEach((d, i) => {
            const card = document.createElement('div');
            const isRejected = d.is_valid === false;
            const isValidated = d.is_valid === true;
            card.className = `box-detail-card ${isRejected ? 'rejected' : (isValidated ? 'validated' : '')}`;

            const color = getColor(d.name);
            let statusBadge = '<span class="status-badge unvalidated">Detection Only</span>';
            if (isValidated) statusBadge = '<span class="status-badge valid">✓ Validated</span>';
            if (isRejected) statusBadge = '<span class="status-badge invalid">✕ Rejected</span>';

            card.innerHTML = `
                <div class="box-title">
                    <span style="display:flex;align-items:center;gap:6px;">
                        <span style="width:10px;height:10px;border-radius:2px;background:${color};display:inline-block;"></span>
                        ${d.name}
                    </span>
                    ${statusBadge}
                </div>
                <div class="box-coords">Box: [${(d.box || []).join(', ')}]</div>
                <div style="color:#94a3b8;font-size:10px;">Source: ${d.source || 'detector'} · Score: ${d.score !== undefined && d.score !== null ? d.score : 1.0}</div>
                ${d.reason ? `<div class="box-reason ${isRejected ? 'invalid-reason' : ''}">${d.reason}</div>` : ''}
            `;
            sidebar.appendChild(card);
        });
    }

    function stepModal(delta) {
        if (activeModalItemIndex < 0) return;
        let newIdx = activeModalItemIndex + delta;
        if (newIdx < 0) newIdx = filteredData.length - 1;
        if (newIdx >= filteredData.length) newIdx = 0;
        openModal(newIdx);
    }

    function closeModal() {
        document.getElementById('modal').classList.remove('active');
        activeModalItemIndex = -1;
    }

    window.onload = init;
    </script>
</body>
</html>
"""


def generate_html_visualization(
    viz_items: List[Dict[str, Any]],
    output_html_path: Union[Path, str],
    title: str = "Auto-Annotate Pipeline Visualization",
) -> Path:
    """Generate a standalone interactive HTML visualization file."""
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_json = json.dumps(viz_items, indent=2)
    html_content = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA_JSON__", data_json)

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
