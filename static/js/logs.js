'use strict';
const PER=25; let pg=1,fil=[];
document.addEventListener('DOMContentLoaded',()=>{
  fil=Array.from(document.querySelectorAll('#logsBody .log-row'));
  document.getElementById('logSearch')?.addEventListener('input',doFilter);
  document.getElementById('typeFilter')?.addEventListener('change',doFilter);
  document.getElementById('severityFilter')?.addEventListener('change',doFilter);
  document.getElementById('btnRefresh')?.addEventListener('click',()=>location.reload());
  document.getElementById('btnExport')?.addEventListener('click',exportCSV);
  document.getElementById('btnPrevPage')?.addEventListener('click',()=>goPage(pg-1));
  document.getElementById('btnNextPage')?.addEventListener('click',()=>goPage(pg+1));
  document.querySelectorAll('.sortable').forEach(th=>th.addEventListener('click',()=>sortBy(th.dataset.col)));
  render();
});
function doFilter(){
  const q=(document.getElementById('logSearch')?.value||'').toLowerCase();
  const t=document.getElementById('typeFilter')?.value||'';
  const s=document.getElementById('severityFilter')?.value||'';
  fil=Array.from(document.querySelectorAll('#logsBody .log-row')).filter(r=>
    (!q||r.textContent.toLowerCase().includes(q))&&(!t||r.dataset.type===t)&&(!s||r.dataset.severity===s));
  pg=1; render();
}
function render(){
  document.querySelectorAll('#logsBody .log-row').forEach(r=>r.style.display='none');
  const s=(pg-1)*PER; fil.slice(s,s+PER).forEach(r=>r.style.display='');
  const tot=Math.max(1,Math.ceil(fil.length/PER));
  const info=document.getElementById('pageInfo');
  if(info)info.textContent=`Page ${pg} of ${tot} (${fil.length} rows)`;
  const pv=document.getElementById('btnPrevPage'),nx=document.getElementById('btnNextPage');
  if(pv)pv.disabled=pg<=1; if(nx)nx.disabled=pg>=tot;
}
function goPage(n){pg=Math.max(1,Math.min(n,Math.ceil(fil.length/PER)||1));render();}
let sc='id',sa=false;
function sortBy(col){
  sa=sc===col?!sa:true; sc=col;
  const idx={id:0,timestamp:1,detection_type:2,severity:7}[col]??0;
  fil.sort((a,b)=>{const av=a.cells[idx]?.textContent.trim()||'',bv=b.cells[idx]?.textContent.trim()||'';const c=isNaN(av)?av.localeCompare(bv):parseFloat(av)-parseFloat(bv);return sa?c:-c;});
  const tb=document.getElementById('logsBody'); fil.forEach(r=>tb.appendChild(r)); render();
}
function exportCSV(){
  const h=['ID','Timestamp','Type','EAR','MAR','Pitch','Yaw','Severity'];
  const rows=fil.map(r=>Array.from(r.cells).slice(0,8).map(c=>`"${c.textContent.trim().replace(/"/g,'""')}"`).join(','));
  const a=Object.assign(document.createElement('a'),{href:URL.createObjectURL(new Blob([[h.join(','),...rows].join('\n')],{type:'text/csv'})),download:`logs_${Date.now()}.csv`});
  a.click(); URL.revokeObjectURL(a.href);
}
async function clearLogs(){
  if(!confirm('Delete ALL detection logs? This cannot be undone.'))return;
  const r=await apiPost('/api/clear_logs');
  showToast(r?'Logs cleared':'Failed to clear',r?'success':'error');
  if(r)setTimeout(()=>location.reload(),800);
}
function viewScreenshot(path){
  const m=document.getElementById('screenshotModal'),img=document.getElementById('screenshotImg');
  if(!m||!img)return; img.src='/screenshots/'+path.split(/[/\\]/).pop(); m.classList.remove('hidden');
}
function closeScreenshot(){document.getElementById('screenshotModal')?.classList.add('hidden');}
window.clearLogs=clearLogs; window.viewScreenshot=viewScreenshot; window.closeScreenshot=closeScreenshot;
