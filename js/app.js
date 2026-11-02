const panel=document.querySelector('#panel'),content=document.querySelector('#panel-content');
function readingEscape(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const readingListHtml='<h2>閱讀日常</h2><p>讀過的文字，留下來的想法。</p>'+readingArticles.map((a,i)=>'<button class="reading-card '+(i%2?'thinking-card':'')+'" data-detail="'+a.id+'"><img class="book-cover" src="'+a.cover+'" alt="《'+readingEscape(a.cardTitle)+'》書封" loading="lazy"><span class="reading-info"><span class="reading-tag">'+readingEscape(a.category)+'<time datetime="'+a.date+'">'+a.date.replaceAll('-','.')+'</time></span><strong>'+readingEscape(a.cardTitle)+'</strong><span class="reading-source">'+readingEscape(a.excerpt)+'</span><span class="reading-cta">閱讀全文 ↗</span></span></button>').join('');
const readingPages=Object.fromEntries(readingArticles.map(a=>[a.id,{name:'READING.md / '+a.cardTitle,html:a.html}]));
const developmentListHtml='<h2>開發筆記</h2><p>記錄實作過程中的問題、解法，以及每次嘗試的收穫。</p>'+developmentArticles.map(a=>'<button class="development-card" data-detail="'+a.id+'"><img class="development-thumbnail" src="'+a.image+'" alt="'+readingEscape(a.imageAlt)+'" loading="lazy"><span class="development-card-content"><span class="reading-tag">'+readingEscape(a.category)+'<time datetime="'+a.date+'">'+a.date.replaceAll('-','.')+'</time></span><strong>'+readingEscape(a.title)+'</strong><span class="reading-source">'+readingEscape(a.excerpt)+'</span><span class="reading-cta">閱讀全文 ↗</span></span></button>').join('');
const developmentPages=Object.fromEntries(developmentArticles.map(a=>[a.id,{name:'DEVELOPMENT.md / '+a.windowName,html:a.html}]));
const pages={
...portfolioPages,
reading:{name:'READING.md',html:readingListHtml},
...readingPages,
about:{name:'ABOUT.md',html:'<div class="eyebrow">HELLO, I\'M TED · 12+ YEARS IN GAMES</div><h2>關於我</h2><p class="about-lead">嗨，我是 <strong>Ted Sie</strong>，在遊戲業寫了十二年多 C# 的 Unity 工程師。</p><p>我喜歡把複雜的系統慢慢理出頭緒：規劃架構、整理解法、追出問題，再一點一點把效能磨順，讓專案一直保持在好狀態。我希望經手的每個專案，從技術到製程，都能讓人安心。</p><ul class="about-skills" aria-label="專長"><li>C#</li><li>Unity</li><li>架構設計</li><li>問題排查</li><li>效能優化</li></ul><p class="about-en" lang="en">Hi, I\'m Ted, a Unity engineer who has been writing C# in the game industry for 12+ years. I enjoy untangling complex systems: shaping the architecture, sorting out solutions, tracking down issues and patiently polishing performance until everything runs smoothly. I want every project I work on to feel solid, in its tech and in how it\'s made.</p><p class="about-links-intro">想聊聊遊戲或程式，都歡迎來這裡找我：</p><div class="about-links"><a href="https://www.linkedin.com/in/ted-sie/" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a><a href="https://github.com/ted10401" target="_blank" rel="noopener noreferrer">GitHub ↗</a><a href="https://www.facebook.com/tedsieblog" target="_blank" rel="noopener noreferrer">Facebook ↗</a></div><p class="about-room">房間不大，電腦裡存著參與過的遊戲，書架上是讀過的書，桌上攤著開發筆記。隨意坐，慢慢逛。</p>'},
notes:{name:'DEVELOPMENT.md',html:developmentListHtml},
...developmentPages
};
// Embeds in development notes need a Referer, which file:// pages don't send (YouTube shows error 153).
// Replace them with a link: YouTube gets its thumbnail, other embeds a plain link.
function noteEmbedFallback(f){const id=(f.src.match(/\/embed\/([\w-]+)/)||[])[1];const yt=/youtube/.test(f.src)&&id;const a=document.createElement('a');a.href=yt?'https://www.youtube.com/watch?v='+id:f.src.replace(/\/embed(\?.*)?$/,'');a.target='_blank';a.rel='noopener noreferrer';a.className='note-embed-fallback';if(yt){const img=document.createElement('img');img.src='https://img.youtube.com/vi/'+id+'/hqdefault.jpg';img.alt=f.title;a.append(img)}const label=document.createElement('span');label.textContent='在新分頁開啟 '+f.title+' ↗';a.append(label);f.replaceWith(a)}
function openPage(key){
panel.classList.toggle('portfolio-player',portfolioGameIds.includes(key));
panel.classList.toggle('article-reader',Object.hasOwn(readingPages,key)||Object.hasOwn(developmentPages,key));panel.scrollTop=0;const p=pages[key];document.querySelector('#window-name').textContent=p.name;content.innerHTML=p.html;if(!panel.open)panel.showModal();
if(!portfolioCanEmbed)content.querySelectorAll('.note-embed iframe').forEach(noteEmbedFallback);
content.querySelectorAll('[data-detail]').forEach(b=>b.onclick=e=>{e.preventDefault();openPage(b.dataset.detail)});
content.querySelectorAll('[data-section]').forEach(b=>b.onclick=()=>document.getElementById(b.dataset.section).scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'}));
}
document.querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>openPage(b.dataset.open));document.querySelector('#close').onclick=()=>panel.close();panel.addEventListener('close',()=>{if(panel.classList.contains('portfolio-player'))content.innerHTML=''});panel.addEventListener('click',e=>{if(e.target===panel){const r=panel.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)panel.close()}});
const theme=document.querySelector('#theme');theme.onclick=()=>{const night=document.body.classList.toggle('night');theme.setAttribute('aria-pressed',String(night));theme.setAttribute('aria-label',night?'切換日光模式':'切換夜晚模式');theme.innerHTML=night?'☾ <span>夜晚</span>':'☼ <span>日光</span>'};
function clock(){document.querySelector('#clock').textContent=new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Taipei',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date())}clock();setInterval(clock,10000);
document.querySelector('#help').onclick=()=>{const active=document.querySelector('.room').classList.toggle('hints');document.querySelector('#help').textContent=active?'收起提示 ×':'探索房間 ↗'};

