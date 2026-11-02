const canvas=document.querySelector('#room-canvas'),c=canvas.getContext('2d');c.imageSmoothingEnabled=false;
const ink='#2b3a49';
function poly(points,fill,stroke=ink,line=2){c.beginPath();points.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.closePath();c.fillStyle=fill;c.fill();if(stroke){c.strokeStyle=stroke;c.lineWidth=line;c.lineJoin='miter';c.stroke()}}
function line(points,color=ink,w=2){c.beginPath();points.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle=color;c.lineWidth=w;c.stroke()}
function rect(x,y,w,h,color){c.fillStyle=color;c.fillRect(x,y,w,h)}
function box(x,y,w,d,h,top='#e7edf0',front='#b1bfcc',side='#7f91a3'){poly([[x,y],[x+w,y-w*.4],[x+w+d,y-w*.4+d*.55],[x+d,y+d*.55]],top);poly([[x+d,y+d*.55],[x+w+d,y-w*.4+d*.55],[x+w+d,y-w*.4+d*.55+h],[x+d,y+d*.55+h]],front);poly([[x,y],[x+d,y+d*.55],[x+d,y+d*.55+h],[x,y+h]],side)}
let rand=143;function rng(){rand=(rand*16807)%2147483647;return(rand-1)/2147483646}
function foliage(x,y,s,color){for(let i=0;i<30;i++){const a=rng()*Math.PI*2,r=rng()*s;const px=Math.round((x+Math.cos(a)*r)/4)*4,py=Math.round((y+Math.sin(a)*r*.8)/4)*4;rect(px,py,8+rng()*9,7+rng()*11,i%3===0?'#879b58':i%3===1?color:'#607642')}}
function pot(x,y,s=1){poly([[x-18*s,y],[x+18*s,y-7*s],[x+14*s,y+30*s],[x-10*s,y+34*s]],'#70828b');poly([[x-18*s,y],[x-10*s,y+5*s],[x-10*s,y+34*s],[x-18*s,y+24*s]],'#4b5e6b');poly([[x-18*s,y],[x+9*s,y-11*s],[x+18*s,y-7*s],[x-10*s,y+5*s]],'#a5b6bb');line([[x,y],[x-2*s,y-55*s]],'#5b6e49',4*s);foliage(x-4*s,y-52*s,26*s,'#758d50')}
// One shared monitor transform keeps the drawing, hit area and night light aligned.
const monitorOffset={x:70,y:-83.1};
const monitorCorners=[[243,300],[557,174],[557,356],[243,483]].map(([x,y])=>[x+monitorOffset.x,y+monitorOffset.y]);
const roomElement=document.querySelector('.room');
roomElement.style.setProperty('--monitor-left',`${monitorCorners[0][0]/960*100}%`);
roomElement.style.setProperty('--monitor-top',`${monitorCorners[1][1]/720*100}%`);
// Architectural shell.
rect(0,0,960,720,'#bac7d6');poly([[0,0],[720,0],[720,395],[0,680]],'#bdc9d8',null);poly([[720,0],[960,0],[960,545],[720,395]],'#93a6b9',null);poly([[0,680],[720,395],[960,533],[960,720],[0,720]],'#344553',null);
// Floorboard seams share the desk axes and stay inside the floor plane.
c.save();c.beginPath();c.moveTo(0,680);c.lineTo(720,395);c.lineTo(960,527);c.lineTo(960,720);c.lineTo(0,720);c.closePath();c.clip();
for(let b=450;b<1300;b+=38)line([[-300,b],[1300,b-640]],'#253744',2);
for(let b=-200;b<1000;b+=65)line([[-300,b],[1300,b+880]],'#52606b',1);
c.restore();rect(713,0,13,399,'#61768d');rect(728,0,8,408,'#e0e8ed');
// Window, sky, clouds and sunlight.
poly([[751,-20],[960,0],[960,359],[751,238]],'#566c80',null);poly([[764,-5],[960,0],[960,335],[764,224]],'#d2eced',null);poly([[764,124],[960,217],[960,335],[764,224]],'#b8d5d9',null);poly([[764,30],[960,145],[960,161],[764,47]],'#eff6f0',null);poly([[844,0],[859,0],[859,280],[844,271]],'#eef3ef');poly([[764,113],[960,225],[960,236],[764,124]],'#eef3ef');poly([[743,232],[960,354],[960,376],[743,251]],'#e6ecef');poly([[743,251],[960,376],[960,389],[743,264]],'#7e919f');
// Wall poster.
poly([[32,181],[133,140],[133,294],[32,335]],'#edf1ef');poly([[39,188],[125,153],[125,286],[39,321]],'#f6f4e5');poly([[43,265],[121,234],[121,284],[43,315]],'#4f8aa4',null);poly([[50,266],[61,242],[72,240],[76,250],[88,247],[92,262],[107,256],[112,280],[50,305]],'#405a68',null);poly([[60,239],[61,227],[74,217],[90,222],[98,232],[84,237]],'#bd6f61',null);line([[51,302],[111,277]],'#c3e1df',4);
// Floating shelf above the rear-set monitor.
c.save();c.transform(.9,0,0,.9,85,-32);
box(171,209,329,31,16,'#eaf0f2','#9eafbf','#8195a9');
const covers=['#d7c087','#3d636b','#708b86','#dbe3df','#ac6868','#3c596e','#a9bdb9'];
// Thin spines, deep covers and paper edges distinguish books from blocks.
// The stack at the far end is painted behind the upright books.
for(let i=0;i<3;i++){
  const x=418+(i===1?4:0),y=137-i*12,w=65,d=27;
  poly([[x-d,y-d*.55],[x+w-d,y-w*.4-d*.55],[x+w,y-w*.4],[x,y]],covers[(i+2)%7]);
  poly([[x,y+2],[x+w,y-w*.4+2],[x+w,y-w*.4+8],[x,y+8]],'#e3e3cc');
  line([[x+4,y+5],[x+w-3,y-w*.4+5]],'#b6baa6',1);
  line([[x,y+9],[x+w,y-w*.4+9]],covers[(i+2)%7],3);
  poly([[x-d,y-d*.55],[x,y],[x,y+9],[x-d,y-d*.55+9]],'#566f70');
}
for(let i=11;i>=0;i--){
  const x=202+i*17,y=224-i*6.8,w=[7,10,6,9][i%4],h=[63,78,69,84,73][i%5],d=29;
  c.save();c.translate(x,y);
  if(i===8||i===9)c.transform(1,0,.11,1,0,0);
  const dy=-w*.4,color=covers[i%covers.length];
  poly([[-d,-h-d*.55],[0,-h],[0,0],[-d,-d*.55]],color);
  line([[-d+4,-h-d*.55+5],[-4,-h+5]],'#ffffff35',1);
  line([[-d+4,-d*.55-4],[-4,-4]],'#233f4b40',1);
  poly([[-d+2,-h-d*.55+1],[w-d,-h+dy-d*.55],[w-1,-h+dy-1],[1,-h-1]],'#e8e6ce');
  for(let page=0;page<3;page++)line([[-d+5+page*7,-h-(d-5-page*7)*.55],[w-d+4+page*7,-h+dy-(d-4-page*7)*.55]],'#b9bda8',.7);
  poly([[0,-h],[w,-h+dy],[w,dy],[0,0]],color);
  line([[2,-h+10],[w-1,-h+dy+10]],'#eee7c5',1);
  line([[2,-9],[w-1,dy-9]],'#cdd6bf',1);
  if(w>7){rect(3,-h+21,2,15,'#e4e4c3');rect(3,-h+41,2,6,'#e4e4c3')}
  c.restore();
}

pot(519,74,.42);c.restore();
// Wall pegboard.
poly([[36,365],[150,319],[150,436],[36,482]],'#e4e6df');for(let x=45;x<145;x+=10)for(let y=364;y<443;y+=11)if(y>365-(x-45)*.4)rect(x,y,2,3,'#8b9caa');box(23,467,123,24,10,'#647785','#394b57','#2a3b49');
// Cabinet, set back under the desktop.
c.save();c.translate(-18.4,-10.1);box(723,443,124,81,180,'#a3b4c3','#657b8d','#415868');for(let i=0;i<4;i++){line([[807,474+i*35],[917,430+i*35]],'#344b5c',2);line([[855,473+i*35],[876,465+i*35]],'#2d4352',4)}c.restore();
// Rear brace runs along the back legs; it is drawn first so the leg covers its end and it reads as joined into the leg.
poly([[194,632.7],[721,421.9],[727,425.2],[200,636]],'#6d8496');poly([[200,636],[727,425.2],[727,437.2],[200,648]],'#425768');
// Left desk leg.
poly([[168,532],[189,540],[189,685],[168,674]],'#b4c4cc');poly([[189,540],[216,529],[216,674],[189,685]],'#6d8496');
// Front corner leg, drawn behind the desktop fascia.
poly([[349,624],[367,634],[367,787],[349,777]],'#b4c4cc');
poly([[367,634],[386,626.4],[386,779.4],[367,787]],'#6d8496');
line([[353,644],[353,777]],'#d8e0df',2);
// Main desktop.
poly([[104,485],[654,265],[920,411.3],[370,631.3]],'#f0f3f1');poly([[104,485],[370,631.3],[370,651.3],[104,505]],'#b9c9d6');poly([[370,631.3],[920,411.3],[920,431.3],[370,651.3]],'#93a9bb');line([[117,487],[372,626],[908,412]],'#fff',3);
// Rear-set monitor platform, pushed back along the table depth axis so it sits
// entirely behind the soundbar (the soundbar stands on the desk in front of it).
c.save();c.translate(-66,-36.3);
box(320,481,340,44,24,'#f7f8f3','#b6c7d4','#8ca3b5');
// The stand's foot sits on the platform behind the soundbar, centred under the display
// (shifted along the long axis); its upright continues up behind the display.
c.translate(24,-9.6);
box(476,434,65,20,7,'#b7c9d0','#748e9f','#587180');
poly([[511,348],[532,339.6],[532,425],[511,433.4]],'#7795a4');
poly([[511,348],[501,342.5],[501,428],[511,433.4]],'#425f72');
line([[514,353],[514,427]],'#c5d6dc',2);
c.restore();
// Main monitor silhouette, screen and mountain wallpaper.
c.save();c.translate(monitorOffset.x,monitorOffset.y);
poly([[222,288.4],[555,155.4],[567,162],[234,295]],'#3d5664');poly([[222,288.4],[234,295],[234,499],[222,492.4]],'#263d49');poly([[234,295],[567,162],[567,365],[234,499]],'#718799');poly([[243,300],[557,174],[557,356],[243,483]],'#c9e5eb',null);
poly([[243,396],[269,366],[292,371],[333,299],[366,322],[395,290],[429,300],[461,243],[495,256],[526,220],[557,232],[557,356],[243,483]],'#83b1c4',null);poly([[243,432],[290,383],[319,390],[363,352],[386,359],[420,311],[456,327],[484,276],[517,284],[557,256],[557,356],[243,483]],'#4d7b99',null);poly([[243,461],[288,422],[321,430],[361,398],[399,413],[450,358],[482,369],[524,332],[557,338],[557,356],[243,483]],'#34536f',null);line([[243,481],[557,355]],'#9bafbd',3);c.restore();
// One compact soundbar with a central clock, placed below the display.
box(388,428,180,18,31,'#718993','#243e49','#496672');
line([[392,429],[566,359.4]],'#c1ced0',1);
// Desk mat, keyboard and mouse.
poly([[350,544],[660,420],[750,470],[440,594]],'#a3b5c0',null);poly([[409,535],[587,464],[660,501],[482,573]],'#293e47');poly([[410,531],[587,460],[660,497],[482,569]],'#749a9e');for(let row=0;row<5;row++)for(let col=0;col<14;col++){const x=420+col*11.7+row*11.8,y=530-col*4.68+row*6.3;poly([[x,y],[x+9,y-3.6],[x+18,y+1],[x+9,y+4.8]],row===4&&col>3&&col<9?'#6a8e8c':covers[(col+row)%7]==='#ac6868'?'#6b999b':'#88b4ad','#466c72',1)}
poly([[650,461],[666,455],[682,464],[682,476],[667,482],[651,474]],'#f1f4ed');line([[664,457],[667,467]],'#8199a0',2);
// Mug.
box(288,522,22,19,29,'#b2d0c6','#6b9d94','#4d7473');line([[326,528],[336,524],[336,540],[327,544]],'#71928d',4);
// Notebook and pen.
poly([[388,571],[441,550],[490,575],[436,598]],'#87999b');poly([[389,566],[440,546],[489,571],[436,592]],'#eeeee0');line([[436,550],[436,585]],'#a7b2a8',1);for(let i=0;i<4;i++)line([[397,567+i*4],[423,557+i*4]],'#9baaa3',1);line([[464,586],[491,575]],'#345d65',4);
// Window plants and little shelving.
// A continuous window sill supports the pots above the desktop.
poly([[742,264],[1006,409.2],[964,426],[700,280.8]],'#e8edef');
poly([[700,280.8],[964,426],[964,444],[700,298.8]],'#97adbc');
line([[702,281],[964,425]],'#f9faf3',3);
line([[702,298],[964,442]],'#526c80',2);
pot(882,330,1.18);pot(944,360,.72);line([[887,293],[887,154],[880,89],[894,31]],'#7b7848',4);line([[885,230],[855,182],[848,116]],'#7f8050',3);line([[885,182],[917,133],[921,66]],'#8d8a54',3);for(let i=0;i<45;i++){const x=850+rng()*81,y=26+rng()*204;poly([[x,y],[x+16,y-7],[x+10,y+3],[x-3,y+6]],i%3===0?'#9d9a54':'#7b8a49',null)}
// Subtle light texture.
for(let i=0;i<220;i++){const x=Math.floor(rng()*960),y=Math.floor(rng()*720);rect(x,y,1,1,'#ffffff0c')}
// Reuse the exact screen pixels for the lit display at night.
const screenLight=document.querySelector('#monitor-light').getContext('2d');
screenLight.imageSmoothingEnabled=false;
screenLight.save();
screenLight.beginPath();
monitorCorners.forEach(([x,y],i)=>i?screenLight.lineTo(x,y):screenLight.moveTo(x,y));screenLight.closePath();
screenLight.clip();screenLight.drawImage(canvas,0,0);screenLight.restore();


// Reflected screen light follows the tabletop and fades across the keyboard.
const deskGlow=document.querySelector('#desk-glow').getContext('2d');
deskGlow.save();deskGlow.beginPath();
deskGlow.moveTo(104,485);deskGlow.lineTo(654,265);deskGlow.lineTo(920,411.3);deskGlow.lineTo(370,631.3);deskGlow.closePath();deskGlow.clip();
deskGlow.drawImage(canvas,0,0);
deskGlow.globalCompositeOperation='source-atop';deskGlow.fillStyle='#a1e2ea25';deskGlow.fillRect(0,0,960,720);
deskGlow.globalCompositeOperation='destination-in';
deskGlow.translate(507,482);deskGlow.transform(1,-.25,.4,.65,0,0);
const glowFade=deskGlow.createRadialGradient(0,0,15,0,0,245);
glowFade.addColorStop(0,'rgba(255,255,255,.72)');glowFade.addColorStop(.5,'rgba(255,255,255,.38)');glowFade.addColorStop(1,'rgba(255,255,255,0)');
deskGlow.fillStyle=glowFade;deskGlow.fillRect(-600,-600,1200,1200);deskGlow.restore();




// A warm daylight wash from the window.
const sunlight=document.querySelector('#daylight').getContext('2d');
const sunWash=sunlight.createRadialGradient(930,90,30,770,235,950);
sunWash.addColorStop(0,'rgba(255,222,155,.27)');sunWash.addColorStop(.6,'rgba(255,225,170,.13)');sunWash.addColorStop(1,'rgba(255,229,187,.02)');
sunlight.fillStyle=sunWash;sunlight.fillRect(0,0,960,720);
// The display emits its own light; sunlight remains on the surrounding surfaces.
sunlight.globalCompositeOperation='destination-out';sunlight.beginPath();monitorCorners.forEach(([x,y],i)=>i?sunlight.lineTo(x,y):sunlight.moveTo(x,y));sunlight.closePath();sunlight.fillStyle='#000';sunlight.fill();sunlight.globalCompositeOperation='source-over';

