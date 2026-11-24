'use strict';
let running=false,sessStart=null,sessTimer=null,earChart=null,alertLock=false;
const EAR_H=Array(60).fill(0.3),ET=0.25;
document.addEventListener('DOMContentLoaded',()=>{
  initChart();
  document.getElementById('btnStart').onclick=startDet;
  document.getElementById('btnStop').onclick=stopDet;
  document.getElementById('btnSnapshot').onclick=()=>apiGet('/api/screenshots').then(r=>showToast(r&&r.screenshots.length?'Snap: '+r.screenshots[0]:'No snapshots yet','info'));
  const s=window.drowsySocket; if(s)s.on('driver_state',render);
  setInterval(()=>{ if(!running)return; apiGet('/api/driver_state').then(d=>d&&render(d)); },1200);
});
async function startDet(){
  if(!(await apiPost('/start_detection')))return showToast('Could not start','error');
  running=true; sessStart=Date.now();
  document.getElementById('videoFeed').src='/video_feed';
  document.getElementById('videoFeed').classList.remove('hidden');
  document.getElementById('videoPlaceholder').style.display='none';
  document.getElementById('btnStart').disabled=true;
  document.getElementById('btnStop').disabled=false;
  clearInterval(sessTimer);
  sessTimer=setInterval(()=>{const e=document.getElementById('ssDuration');if(e)e.textContent=formatDuration(Math.floor((Date.now()-sessStart)/1000));},1000);
  showToast('Detection started','success');
}
async function stopDet(){
  await apiPost('/stop_detection'); running=false;
  document.getElementById('videoFeed').src='';
  document.getElementById('videoFeed').classList.add('hidden');
  document.getElementById('videoPlaceholder').style.display='';
  document.getElementById('btnStart').disabled=false;
  document.getElementById('btnStop').disabled=true;
  clearInterval(sessTimer); showToast('Detection stopped','info');
  render({status:'NORMAL',eye_status:'Open',yawn_status:'No Yawn',head_pose:'Forward',phone_detected:false,sunglasses_detected:false,ear:0,mar:0,head_pitch:0,head_yaw:0,head_roll:0,fps:0,alert_count:0,face_detected:false});
}
const COL={DROWSY:'#ef4444',DISTRACTED:'#f97316',YAWNING:'#f59e0b',NORMAL:'#10b981'};
const DESC={DROWSY:'Driver appears drowsy — pull over!',DISTRACTED:'Driver is distracted!',YAWNING:'Yawning — fatigue warning',NORMAL:'Driver is alert'};
const ICO={DROWSY:'fa-eye-slash',DISTRACTED:'fa-exclamation-triangle',YAWNING:'fa-face-tired',NORMAL:'fa-circle-check'};
function set(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}
function setStyle(id,p,v){const e=document.getElementById(id);if(e)e.style[p]=v;}
function tog(id,cls,on){const e=document.getElementById(id);if(e)e.classList.toggle(cls,!!on);}
function render(st){
  const s=st.status||'NORMAL',sl=s.toLowerCase();
  const c=COL[s]||'#10b981';
  const card=document.getElementById('statusCard');
  if(card)card.className='status-card '+sl;
  set('statusMain',s); setStyle('statusMain','color',c);
  set('statusSub',DESC[s]||'Monitoring...');
  const ic=document.getElementById('statusIcon');
  if(ic){ic.className='fa-solid '+(ICO[s]||'fa-circle-check')+' status-icon';ic.style.color=c;}
  const ov=document.getElementById('overlayStatus');
  if(ov){ov.textContent=s;ov.className='overlay-status '+sl;}
  set('indEyeVal',st.eye_status||'—'); set('indYawnVal',st.yawn_status||'—');
  set('indHeadVal',st.head_pose||'—'); set('indPhoneVal',st.phone_detected?'DETECTED':'None');
  set('indGlassesVal',st.sunglasses_detected?'DETECTED':'None');
  tog('indEye','alert',st.eye_status==='Closed'||st.eye_status==='Hidden (Sunglasses)');
  tog('indYawn','alert',st.yawn_status==='Yawning');
  tog('indHead','alert',st.head_pose!=='Forward');
  tog('indPhone','alert',!!st.phone_detected);
  tog('indGlasses','alert',!!st.sunglasses_detected);
  const ear=parseFloat(st.ear)||0,mar=parseFloat(st.mar)||0;
  setStyle('earBar','width',Math.min(100,(ear/0.45)*100)+'%'); set('earVal',ear.toFixed(3));
  setStyle('marBar','width',Math.min(100,(mar)*100)+'%');      set('marVal',mar.toFixed(3));
  [['posePitch','posePitchBar',st.head_pitch],['poseYaw','poseYawBar',st.head_yaw],['poseRoll','poseRollBar',st.head_roll]].forEach(([t,b,v])=>{
    const d=parseFloat(v)||0; set(t,d.toFixed(1)+'°'); setStyle(b,'width',((d+90)/180*100)+'%');
  });
  set('ssFps',st.fps||0); set('ssFace',st.face_detected?'Yes':'No'); set('ssAlerts',st.alert_count||0);
  updChart(ear);
  if((s==='DROWSY'||s==='DISTRACTED'||s==='YAWNING')&&!alertLock){
    alertLock=true; setTimeout(()=>alertLock=false,8000);
    const titles={DROWSY:'DROWSINESS DETECTED!',DISTRACTED:'DISTRACTION ALERT!',YAWNING:'YAWNING DETECTED!'};
    const msgs={DROWSY:'Driver appears drowsy. Please pull over safely.',DISTRACTED:'Driver is distracted. Focus on the road.',YAWNING:'Persistent yawning. Consider a break.'};
    set('alertModalTitle',titles[s]); set('alertModalMsg',msgs[s]);
    const m=document.getElementById('alertModal'); if(m)m.classList.add('active');
  }
}
function initChart(){
  const ctx=document.getElementById('earChart');
  if(!ctx||typeof Chart==='undefined')return;
  earChart=new Chart(ctx,{type:'line',data:{labels:Array(60).fill(''),datasets:[
    {data:[...EAR_H],borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,0.1)',borderWidth:1.5,pointRadius:0,fill:true,tension:0.3},
    {data:Array(60).fill(ET),borderColor:'#ef4444',borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false}
  ]},options:{responsive:true,animation:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{min:0,max:0.5,ticks:{color:'#9ca3c8',font:{size:10}},grid:{color:'rgba(255,255,255,0.05)'}}}}});
}
function updChart(ear){
  if(!earChart)return;
  EAR_H.shift(); EAR_H.push(parseFloat(ear)||0);
  earChart.data.datasets[0].data=[...EAR_H]; earChart.update('none');
}
function dismissAlert(){const m=document.getElementById('alertModal');if(m)m.classList.remove('active');}
window.dismissAlert=dismissAlert;
