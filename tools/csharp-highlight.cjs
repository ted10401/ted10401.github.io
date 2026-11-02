// Minimal C# syntax highlighter used at build time (no runtime dependency).
// Wraps tokens in <span class="tok-…"> so style.css can colour them.
const KEYWORDS = new Set(('abstract as base bool break byte case catch char checked class const continue decimal default delegate do double else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is lock long namespace new null object operator out override params private protected public readonly ref return sbyte sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked unsafe ushort using virtual void volatile while '
  + 'var async await get set value where yield nameof partial dynamic when').split(' '));
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const unesc = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
const TOKEN = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|(\$?@?"(?:[^"\\\n]|\\.|"")*"|'(?:[^'\\\n]|\\.)')|(\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[fFdDmMuUlL]*\b)|([A-Za-z_]\w*)|(\s+)|([^\sA-Za-z_0-9"'\/]+|\/)/g;

function highlightCSharp(source) {
  const code = unesc(source);
  let out = '', m, prev = '', prevNonSpace = '\n', attrNext = false;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(code))) {
    const [text, com, str, num, word, space, punct] = m;
    let cls = null;
    if (com) cls = 'com';
    else if (str) cls = 'str';
    else if (num) cls = 'num';
    else if (word) {
      const rest = code.slice(TOKEN.lastIndex);
      if (attrNext) { cls = 'attr'; attrNext = false; }
      else if (KEYWORDS.has(word)) cls = 'kw';
      else if (prev === 'new') cls = 'type';
      else if (/^\s*(<[\w\s,<>\[\]]*>)?\s*\(/.test(rest)) cls = 'fn';
      else if (/^m_/.test(word) || prevNonSpace === '.' && /^[a-z]/.test(word)) cls = 'field';
      else if (/^[A-Z]/.test(word)) cls = 'type';
      prev = word;
    } else if (punct) {
      // "[" at the start of a line or right after another attribute starts an attribute.
      if (punct.startsWith('[') && (prevNonSpace === '\n' || prevNonSpace === ']' || prevNonSpace === '{' || prevNonSpace === ';')) attrNext = true;
    }
    out += cls ? `<span class="tok-${cls}">${esc(text)}</span>` : esc(text);
    if (space) { if (space.includes('\n')) prevNonSpace = '\n'; }
    else prevNonSpace = punct ? punct[punct.length - 1] : text[text.length - 1];
  }
  return out;
}

const highlightCodeBlocks = html => html.replace(/(<code class="language-csharp">)([\s\S]*?)(<\/code>)/g, (_, open, body, close) => open + highlightCSharp(body) + close);
module.exports = { highlightCSharp, highlightCodeBlocks };
