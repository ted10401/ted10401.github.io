// Local preview: `node tools/server.cjs`, then open http://127.0.0.1:4173
// Serves only the site files (index.html, assets/, css/, js/) from the project root.
const http = require('node:http'), fs = require('node:fs'), path = require('node:path');
const root = path.join(__dirname, '..');
const publicDirs = ['assets', 'css', 'js'];
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'application/javascript; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml' };
http.createServer((req, res) => {
  let rel;
  try { rel = path.posix.normalize(decodeURIComponent(new URL(req.url, 'http://localhost').pathname)).replace(/^\/+/, ''); }
  catch { res.writeHead(400).end(); return; }
  if (rel === '' || rel === '.') rel = 'index.html';
  const top = rel.split('/')[0];
  if (rel.includes('..') || (rel !== 'index.html' && !publicDirs.includes(top))) { res.writeHead(404).end('Not found'); return; }
  fs.readFile(path.join(root, rel), (err, data) => {
    if (err) { res.writeHead(404).end('Not found'); return; }
    res.setHeader('Content-Type', types[path.extname(rel).toLowerCase()] || 'application/octet-stream');
    res.end(data);
  });
}).listen(4173, '127.0.0.1', () => console.log('Local: http://127.0.0.1:4173'));
