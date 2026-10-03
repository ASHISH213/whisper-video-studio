const $=s=>document.querySelector(s), $$=s=>document.querySelectorAll(s);
const observer=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)e.target.classList.add('visible')}),{threshold:.08});
$$('.reveal').forEach(e=>observer.observe(e));
$$('.tab').forEach(btn=>btn.addEventListener('click',()=>{ $$('.tab').forEach(b=>b.classList.remove('active')); $$('.tab-panel').forEach(p=>p.classList.remove('active')); btn.classList.add('active'); $('#'+btn.dataset.tab).classList.add('active') }));
$$('.shot').forEach(s=>s.addEventListener('click',()=>{ $('#lightboxImg').src=s.dataset.img; $('#lightbox').classList.add('open') }));
const close=()=>$('#lightbox').classList.remove('open'); $('.close').addEventListener('click',close); $('#lightbox').addEventListener('click',e=>{if(e.target.id==='lightbox')close()}); document.addEventListener('keydown',e=>{if(e.key==='Escape')close()});
$('#copyLocal').addEventListener('click',async()=>{const t=$('#local pre').innerText;await navigator.clipboard.writeText(t);$('#copyLocal').textContent='COPIED';setTimeout(()=>$('#copyLocal').textContent='COPY',1400)});
const menu=$('.menu'); menu.addEventListener('click',()=>{const nav=$('.nav-links');nav.style.display=nav.style.display==='flex'?'none':'flex';nav.style.position='absolute';nav.style.top='70px';nav.style.right='18px';nav.style.flexDirection='column';nav.style.padding='16px';nav.style.background='#0d0f15';nav.style.border='1px solid var(--line)';nav.style.borderRadius='10px'});
