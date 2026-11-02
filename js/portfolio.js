// Game data comes from portfolio-data.js (built from assets/portfolio/*.json by tools/build-portfolio.cjs).
const portfolioGameIds = portfolioGames.map(g => g.id);

// YouTube refuses embeds without a Referer (error 153), which happens when the page is opened from file://.
// In that case show the poster as a link to YouTube instead of the iframe.
const portfolioCanEmbed = location.protocol !== 'file:';
const portfolioPoster = g => g.image || `https://img.youtube.com/vi/${g.youtube}/hqdefault.jpg`;
const portfolioWatchUrl = g => g.youtube ? `https://www.youtube.com/watch?v=${g.youtube}` : g.store;

function portfolioMedia(g) {
  if (g.youtube && portfolioCanEmbed) {
    return `<div class="portfolio-video"><iframe src="https://www.youtube-nocookie.com/embed/${g.youtube}?rel=0" title="${g.title} 遊戲影片" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>`;
  }
  return `<a class="portfolio-video portfolio-video-fallback" href="${portfolioWatchUrl(g)}" target="_blank" rel="noopener noreferrer"><img src="${portfolioPoster(g)}" alt="${g.title} 遊戲畫面"></a>`;
}

function portfolioLinks(g) {
  const links = [];
  if (g.youtube) links.push(`<a href="${portfolioWatchUrl(g)}" target="_blank" rel="noopener noreferrer">在 YouTube 觀看 ↗</a>`);
  if (g.store) links.push(`<a href="${g.store}" target="_blank" rel="noopener noreferrer">Steam 商店頁 ↗</a>`);
  return links.join('');
}

const portfolioPages = {
  works: {
    name: 'PORTFOLIO.md',
    html: `<h2>作品集</h2><p>參與開發的遊戲作品，與持續累積的創作實驗。</p>
      <div class="portfolio-grid">${portfolioGames.map(g => `<button class="portfolio-card" data-detail="${g.id}">
        <span class="portfolio-poster"><img src="${portfolioPoster(g)}" alt="${g.title} 遊戲畫面" loading="lazy"></span>
        <span class="portfolio-card-content"><span class="reading-tag">${g.tag}</span><strong>${g.title}</strong><span class="reading-cta">${g.youtube ? '播放作品影片 ↗' : '查看作品 ↗'}</span></span>
      </button>`).join('')}</div>`
  }
};

portfolioGames.forEach(g => {
  portfolioPages[g.id] = {
    name: `PORTFOLIO.md / ${(g.fullTitle || g.title).toUpperCase()}`,
    html: `<button class="back" data-detail="works">← 回到作品集</button>
      <div class="eyebrow portfolio-kicker">${g.tag}</div><h2>${g.fullTitle || g.title}</h2>
      ${portfolioMedia(g)}
      <div class="portfolio-video-footer"><span>${g.caption || g.fullTitle || g.title}</span><span class="portfolio-links">${portfolioLinks(g)}</span></div>`
  };
});
