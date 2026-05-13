const queryInput = document.querySelector("#queryInput");
const searchButton = document.querySelector("#searchButton");
const modeSelect = document.querySelector("#modeSelect");
const topicSelect = document.querySelector("#topicSelect");
const yearSelect = document.querySelector("#yearSelect");
const resultsEl = document.querySelector("#results");
const expandedQueryEl = document.querySelector("#expandedQuery");
const resultCountEl = document.querySelector("#resultCount");
const evaluateButton = document.querySelector("#evaluateButton");
const evaluationBox = document.querySelector("#evaluationBox");
const detailDialog = document.querySelector("#detailDialog");
const closeDialog = document.querySelector("#closeDialog");
const paperDetail = document.querySelector("#paperDetail");
const paperCount = document.querySelector("#paperCount");
const cnQueryButton = document.querySelector("#cnQueryButton");
const DEFAULT_QUERY = "\u0052\u0041\u0047 \u5982\u4f55\u7f13\u89e3\u5e7b\u89c9";

function encodeParams(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, value);
    }
  });
  return search.toString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function names(authors) {
  if (authors.length <= 2) return authors.join(", ");
  return `${authors[0]}, ${authors[1]} et al.`;
}

function tags(topics) {
  return topics.map((topic) => `<span class="tag">${escapeHtml(topic)}</span>`).join("");
}

function reasons(items) {
  if (!items || !items.length) return "";
  return `<div class="reasons">${items.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function signalRows(paper) {
  return `
    <div><span>BM25</span><strong>${paper.bm25_score.toFixed(2)}</strong></div>
    <div><span>Semantic</span><strong>${paper.semantic_score.toFixed(2)}</strong></div>
    <div><span>Field</span><strong>${paper.field_score.toFixed(2)}</strong></div>
  `;
}

function renderResults(payload) {
  expandedQueryEl.textContent = payload.expanded_query || "-";
  resultCountEl.textContent = `${payload.count} 篇相关论文`;
  if (payload.facets?.paper_count) {
    paperCount.textContent = payload.facets.paper_count;
  }
  if (!payload.results.length) {
    resultsEl.innerHTML = `<article class="result-card"><div><h3>没有匹配论文</h3><p class="meta">可以尝试更宽泛的 query，或清空主题/年份筛选。</p></div></article>`;
    return;
  }
  resultsEl.innerHTML = payload.results
    .map((paper, index) => {
      const score = Math.max(0, Math.min(100, Math.round(paper.score * 100)));
      return `
        <article class="result-card">
          <div>
            <h3>${index + 1}. ${escapeHtml(paper.title)}</h3>
            <p class="meta">${escapeHtml(names(paper.authors))} / ${escapeHtml(paper.venue)} / ${paper.year}</p>
            <p class="abstract">${escapeHtml(paper.abstract)}</p>
            <div class="evidence">${paper.evidence.highlighted}</div>
            ${reasons(paper.ranking_reasons)}
            <div class="tags">${tags(paper.topics)}</div>
          </div>
          <aside class="result-sidebar">
            <div class="score-meter">
              <strong>${score}%</strong>
              <div class="bar"><span style="--w:${score}%"></span></div>
            </div>
            <div class="signals">${signalRows(paper)}</div>
            <button class="link-button" type="button" data-paper="${paper.id}">查看详情</button>
          </aside>
        </article>
      `;
    })
    .join("");
}

async function loadFacets() {
  const response = await fetch("/api/topics");
  const payload = await response.json();
  paperCount.textContent = payload.paper_count ?? "--";
  topicSelect.innerHTML =
    `<option value="">全部主题</option>` +
    payload.topics.map((topic) => `<option value="${escapeHtml(topic)}">${escapeHtml(topic)}</option>`).join("");
  yearSelect.innerHTML =
    `<option value="">全部年份</option>` +
    payload.years.map((year) => `<option value="${year}">${year}</option>`).join("");
}

async function search() {
  const params = encodeParams({
    q: queryInput.value.trim(),
    mode: modeSelect.value,
    topic: topicSelect.value,
    year: yearSelect.value,
  });
  resultsEl.innerHTML = `<article class="result-card"><div><h3>正在检索...</h3><p class="meta">正在计算 BM25、Semantic、Field 和 MMR 重排序信号。</p></div></article>`;
  const response = await fetch(`/api/search?${params}`);
  renderResults(await response.json());
}

async function showDetail(paperId) {
  const response = await fetch(`/api/papers/${paperId}`);
  const paper = await response.json();
  paperDetail.innerHTML = `
    <h2>${escapeHtml(paper.title)}</h2>
    <p class="meta">${escapeHtml(names(paper.authors))} / ${escapeHtml(paper.venue)} / ${paper.year}</p>
    <div class="tags">${tags(paper.topics)}</div>
    <p class="abstract">${escapeHtml(paper.abstract)}</p>
    <h3>推荐引用证据</h3>
    ${paper.citation_snippets.map((snippet) => `<div class="evidence">${escapeHtml(snippet)}</div>`).join("")}
    <h3>相似论文</h3>
    <ul class="similar-list">
      ${paper.similar
        .map(
          (item) => `
          <li>
            <strong>${escapeHtml(item.title)}</strong>
            <div class="meta">${escapeHtml(item.venue)} / ${item.year} / 相似度 ${item.score.toFixed(2)}</div>
          </li>
        `
        )
        .join("")}
    </ul>
    <p><a href="${escapeHtml(paper.url)}" target="_blank" rel="noreferrer">打开论文来源页面</a></p>
  `;
  detailDialog.showModal();
}

async function evaluate() {
  evaluationBox.innerHTML = `<div class="metric-row"><strong>评测</strong><span>正在运行...</span></div>`;
  const response = await fetch("/api/evaluate");
  const payload = await response.json();
  evaluationBox.innerHTML = Object.entries(payload)
    .map(
      ([mode, metrics]) => `
      <div class="metric-row">
        <strong>${escapeHtml(mode)}</strong>
        <span>P@5 ${metrics.precision_at_5.toFixed(2)} / R@10 ${metrics.recall_at_10.toFixed(2)} / MRR ${metrics.mrr.toFixed(2)}</span>
      </div>
    `
    )
    .join("");
}

searchButton.addEventListener("click", search);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") search();
});
[modeSelect, topicSelect, yearSelect].forEach((control) => control.addEventListener("change", search));
evaluateButton.addEventListener("click", evaluate);
document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    search();
  });
});
resultsEl.addEventListener("click", (event) => {
  const button = event.target.closest("[data-paper]");
  if (button) showDetail(button.dataset.paper);
});
closeDialog.addEventListener("click", () => detailDialog.close());

loadFacets().then(search);
queryInput.value = DEFAULT_QUERY;
if (cnQueryButton) {
  cnQueryButton.dataset.query = DEFAULT_QUERY;
}
