// Builds js/development-library.js from assets/development/<id>/index.html.
// Add a note: create a folder assets/development/<id>/ with index.html and its images, then run `node tools/build-development.cjs`.
// Notes are listed newest first by date and numbered DEVELOPMENT NOTE / 001, 002, ... in that order.
//
// index.html format: a JSON comment at the top, then the article body as HTML.
// Image paths (image and <img src>) are relative to the article folder; full URLs also work.
//   <!--
//   {
//     "title": "文章標題",
//     "date": "2026-10-30",
//     "category": "Unity · ECS",
//     "excerpt": "卡片上的一句簡介",
//     "image": "demo.gif",
//     "imageAlt": "卡片圖片的說明文字"
//   }
//   -->
//   <p>內文……</p>
//
// Required: title, date (YYYY-MM-DD), category, excerpt, image (image file in the article folder), imageAlt.
// Optional: shortTitle  short name for the window title bar (defaults to title), e.g. "Unity DOTS"
//           heading     HTML for the article's <h1> (defaults to title), e.g. "Unity DOTS<br><span>副標</span>"
//           source      URL of the original post; shown as a 原文 link next to the date
// Videos: <video src="clip.mp4" poster="clip-poster.jpg" autoplay loop muted playsinline> — paths are relative like images.
// Link to another note with: <a href="#development-<id>" data-detail="development-<id>">…</a>
const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname, '..');
const { highlightCodeBlocks } = require('./csharp-highlight.cjs');

const dir = path.join(root, 'assets', 'development');
const out = path.join(root, 'js', 'development-library.js');
const required = ['title', 'date', 'category', 'excerpt', 'image', 'imageAlt'];
const allowed = [...required, 'shortTitle', 'heading', 'source'];
const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const notes = fs.readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => d.name).sort().map(slug => {
  const file = `${slug}/index.html`;
  if (!fs.existsSync(path.join(dir, file))) throw new Error(`${slug}/: missing index.html`);
  // Relative image paths point into the article folder.
  const local = p => /^(https?:|\/|#|data:|assets\/)/.test(p) ? p : `assets/development/${slug}/${p}`;
  const text = fs.readFileSync(path.join(dir, file), 'utf8').replace(/^﻿/, '');
  const m = text.match(/^\s*<!--([\s\S]*?)-->\s*([\s\S]*)$/);
  if (!m) throw new Error(`${file}: must start with a <!-- { ...JSON... } --> block`);
  let meta;
  try { meta = JSON.parse(m[1]); } catch (e) { throw new Error(`${file}: invalid JSON in the top comment (${e.message})`); }
  const media = /<(?:img|video)\b[^>]*>/g;
  const body = m[2].trim().replace(media, tag => tag.replace(/(\s(?:src|poster)=")([^"]+)"/g, (_, attr, src) => attr + local(src) + '"'));
  if (meta.image) meta.image = local(meta.image);
  const problems = [];
  if (!/^[a-z0-9_-]+$/.test(slug)) problems.push('folder name must use a-z, 0-9, _ and -');
  required.filter(k => !meta[k]).forEach(k => problems.push(`"${k}" is required`));
  Object.keys(meta).filter(k => !allowed.includes(k)).forEach(k => problems.push(`unknown field "${k}"`));
  if (meta.date && !/^\d{4}-\d{2}-\d{2}$/.test(meta.date)) problems.push('"date" must be YYYY-MM-DD');
  if (meta.image && !/^https?:/.test(meta.image) && !fs.existsSync(path.join(root, meta.image))) problems.push(`image not found: ${meta.image}`);
  if (!body) problems.push('article body is empty');
  [...body.matchAll(media)].flatMap(([tag]) => [...tag.matchAll(/\s(?:src|poster)="([^"]+)"/g)].map(([, src]) => src)).filter(src => !/^https?:/.test(src) && !fs.existsSync(path.join(root, src))).forEach(src => problems.push(`image not found: ${src}`));
  if (problems.length) throw new Error(`${file}: ${problems.join('; ')}`);
  return { id: `development-${slug}`, ...meta, body };
}).sort((a, b) => b.date.localeCompare(a.date) || a.id.localeCompare(b.id));

const ids = new Set(notes.map(a => a.id));
notes.forEach(a => [...a.body.matchAll(/data-detail="(development-[^"]+)"/g)].forEach(([, id]) => {
  if (!ids.has(id)) console.warn(`warning: ${a.id} links to missing note ${id}`);
}));

const result = notes.map((a, i) => ({
  id: a.id, title: a.title, date: a.date, category: a.category, image: a.image, imageAlt: a.imageAlt, excerpt: a.excerpt,
  windowName: (a.shortTitle || a.title).toUpperCase(),
  html: `<div class="reader-toolbar"><button class="back" data-detail="notes">← 回到開發筆記</button><span>DEVELOPMENT NOTE / ${String(i + 1).padStart(3, '0')}</span></div>`
    + `<div class="article-scroll"><article class="native-article development-article"><div class="eyebrow">${esc(a.category.toUpperCase())}</div><h1>${a.heading || esc(a.title)}</h1>`
    + `<div class="native-meta"><time datetime="${a.date}">${a.date.replaceAll('-', '.')}</time>${a.source ? `<a href="${esc(a.source)}" target="_blank" rel="noopener noreferrer">原文</a>` : ''}</div>`
    + `<div class="article-body">${highlightCodeBlocks(a.body)}</div><div class="article-end"><span>END OF NOTE / 感謝閱讀</span></div></article></div>`
}));

fs.writeFileSync(out, `// Generated by tools/build-development.cjs from assets/development/<id>/index.html. Do not edit by hand.\n// Full articles migrated with the author's authorization.\nconst developmentArticles = ${JSON.stringify(result)};\n`);
console.log(`Built ${path.relative(root, out)} with ${notes.length} notes:`);
notes.forEach((a, i) => console.log(`  ${String(i + 1).padStart(3, '0')}  ${a.date}  ${a.title}`));
