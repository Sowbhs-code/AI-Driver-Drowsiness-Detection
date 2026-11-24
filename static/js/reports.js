'use strict';
document.addEventListener('DOMContentLoaded',()=>{
  const sd=window.statsData||{};
  buildTypeChart(sd.detection_by_type||{});
  buildDailyChart(sd.daily_counts||{});
  buildHourlyChart(sd.hourly_distribution||{});
  buildSeverityChart(sd.severity_breakdown||{});
});

function chartDefaults(){
  return {responsive:true,plugins:{legend:{labels:{color:'#9ca3c8',font:{size:11}}}},scales:{x:{ticks:{color:'#9ca3c8'},grid:{color:'rgba(255,255,255,0.05)'}},y:{ticks:{color:'#9ca3c8'},grid:{color:'rgba(255,255,255,0.05)'}}}};
}

function buildTypeChart(data){
  const ctx=document.getElementById('typeChart'); if(!ctx)return;
  const labels=Object.keys(data).length?Object.keys(data):['DROWSY','YAWN','DISTRACTED','PHONE'];
  const values=Object.values(data).length?Object.values(data):[0,0,0,0];
  new Chart(ctx,{type:'doughnut',data:{labels,datasets:[{data:values,backgroundColor:['#ef4444','#f59e0b','#f97316','#8b5cf6'],borderColor:'#1e2130',borderWidth:2}]},
    options:{responsive:true,plugins:{legend:{position:'bottom',labels:{color:'#9ca3c8',font:{size:11},padding:12}}}}});
}

function buildDailyChart(data){
  const ctx=document.getElementById('dailyChart'); if(!ctx)return;
  const last7=[...Array(7)].map((_,i)=>{const d=new Date();d.setDate(d.getDate()-6+i);return d.toISOString().slice(0,10);});
  const vals=last7.map(d=>data[d]||0);
  new Chart(ctx,{type:'bar',data:{labels:last7,datasets:[{label:'Detections',data:vals,backgroundColor:'rgba(59,130,246,0.7)',borderColor:'#3b82f6',borderWidth:1,borderRadius:4}]},
    options:{...chartDefaults(),plugins:{legend:{display:false}}}});
}

function buildHourlyChart(data){
  const ctx=document.getElementById('hourlyChart'); if(!ctx)return;
  const hours=Array.from({length:24},(_,i)=>String(i).padStart(2,'0'));
  const vals=hours.map(h=>data[h]||0);
  new Chart(ctx,{type:'line',data:{labels:hours.map(h=>h+':00'),datasets:[{label:'Events',data:vals,borderColor:'#10b981',backgroundColor:'rgba(16,185,129,0.1)',borderWidth:2,pointRadius:3,fill:true,tension:0.4}]},
    options:{...chartDefaults(),plugins:{legend:{display:false}}}});
}

function buildSeverityChart(data){
  const ctx=document.getElementById('severityChart'); if(!ctx)return;
  const labels=['CRITICAL','HIGH','MEDIUM','LOW'];
  const vals=labels.map(l=>data[l]||0);
  new Chart(ctx,{type:'bar',data:{labels,datasets:[{label:'Count',data:vals,backgroundColor:['#ef4444','#f97316','#f59e0b','#10b981'],borderRadius:4}]},
    options:{...chartDefaults(),indexAxis:'y',plugins:{legend:{display:false}}}});
}
