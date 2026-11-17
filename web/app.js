const N = 24, ALPHA = 0.05, DT = 0.003, STEPS = 100, AMBIENT = 20, DX = 1 / (N - 1);
const $ = (id) => document.getElementById(id);
const controls = { x: $('heaterX'), y: $('heaterY'), strength: $('strength'), time: $('time') };
const state = { model: null, timer: null };

async function loadModel() {
  const response = await fetch('model.json');
  if (!response.ok) throw new Error('Could not load model.json');
  state.model = await response.json();
}

function features(row) {
  const [x, y, strength, time] = row;
  const result = [2*x-1, 2*y-1, 2*strength-1, 2*time-1];
  for (const v of [x, y]) for (let k=1;k<=4;k++) result.push(Math.sin(v*k*Math.PI));
  for (const v of [x, y]) for (let k=1;k<=4;k++) result.push(Math.cos(v*k*Math.PI));
  return result;
}

function predict(x, y, strength, seconds) {
  const m=state.model, t=seconds/(DT*STEPS), f=features([x,y,strength,t]);
  const hidden = m.w1[0].map((_,j)=>Math.tanh(f.reduce((sum,v,i)=>sum+v*m.w1[i][j],m.b1[j])));
  const rise = new Float64Array(N*N);
  for (let i=0;i<rise.length;i++) {
    let value=m.b2[i]; for(let j=0;j<hidden.length;j++) value+=hidden[j]*m.w2[j][i];
    const r=Math.floor(i/N), c=i%N;
    const mask=(r===0||r===N-1||c===0||c===N-1)?0:1;
    rise[i]=AMBIENT+value*t*mask;
  }
  return rise;
}

function solve(x,y,strength,steps) {
  const u=new Float64Array(N*N), next=new Float64Array(N*N);
  const power=new Float64Array(N*N);
  for(let r=0;r<N;r++) for(let c=0;c<N;c++) {
    const px=c*DX, py=r*DX;
    power[r*N+c]=strength*20*Math.exp(-((px-x)**2+(py-y)**2)/(2*.08**2));
  }
  for(let s=0;s<steps;s++) {
    for(let r=1;r<N-1;r++) for(let c=1;c<N-1;c++) {
      const i=r*N+c, lap=(u[i+N]+u[i-N]+u[i+1]+u[i-1]-4*u[i])/(DX*DX);
      next[i]=u[i]+DT*(ALPHA*lap+power[i]);
    }
    u.set(next);
  }
  for(let i=0;i<u.length;i++)u[i]+=AMBIENT;
  return u;
}

const palette=[[23,19,44],[70,28,104],[131,35,103],[187,47,78],[229,87,47],[247,174,56],[255,243,130]];
function color(value,min,max) {
  const z=Math.max(0,Math.min(0.999999,(value-min)/(max-min||1)))*(palette.length-1),i=Math.floor(z),q=z-i;
  return palette[i].map((v,k)=>Math.round(v*(1-q)+palette[i+1][k]*q));
}
function draw(canvas,values,min,max,x,y) {
  const ctx=canvas.getContext('2d'), tile=canvas.width/N;
  for(let r=0;r<N;r++)for(let c=0;c<N;c++){
    const [red,green,blue]=color(values[r*N+c],min,max);
    ctx.fillStyle=`rgb(${red},${green},${blue})`;
    ctx.fillRect(c*tile,(N-1-r)*tile,tile+.5,tile+.5);
  }
  ctx.strokeStyle='rgba(255,255,255,.85)';ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(x*(canvas.width-1),(1-y)*(canvas.height-1),7,0,Math.PI*2);ctx.stroke();
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x*(canvas.width-1),(1-y)*(canvas.height-1),2,0,Math.PI*2);ctx.fill();
}
function rmse(a,b,where=null) {let sum=0,n=0;for(let i=0;i<a.length;i++)if(!where||where(i)){sum+=(a[i]-b[i])**2;n++;}return Math.sqrt(sum/Math.max(1,n));}
function median(v){v.sort((a,b)=>a-b);return v[Math.floor(v.length/2)];}
function elapsed(fn){const start=performance.now();fn();return performance.now()-start;}
function benchmark(x,y,strength,steps){
  const modelTimes=[],solverTimes=[];
  for(let i=0;i<7;i++) modelTimes.push(elapsed(()=>predict(x,y,strength,steps*DT)));
  for(let i=0;i<7;i++) solverTimes.push(elapsed(()=>solve(x,y,strength,steps)));
  return {model:median(modelTimes),solver:median(solverTimes)};
}

function update(){
  const x=+controls.x.value,y=+controls.y.value,strength=+controls.strength.value,seconds=+controls.time.value;
  $('xOut').value=x.toFixed(2);$('yOut').value=y.toFixed(2);$('strengthOut').value=strength.toFixed(2);$('timeOut').value=seconds.toFixed(3);
  const steps=Math.round(seconds/DT),reference=solve(x,y,strength,steps),prediction=predict(x,y,strength,seconds);
  const min=AMBIENT,max=Math.max(...reference,...prediction);
  draw($('solverCanvas'),reference,min,max,x,y);draw($('modelCanvas'),prediction,min,max,x,y);
  $('tempRange').textContent=`${min.toFixed(2)}° — ${max.toFixed(2)}°`;
  $('rmse').textContent=rmse(reference,prediction).toFixed(4);
  const hottest=Math.max(...reference)-AMBIENT;
  $('hotError').textContent=rmse(reference,prediction,i=>reference[i]-AMBIENT>=hottest*.5).toFixed(4);
  $('solverTime').textContent='calculating…';$('modelTime').textContent='calculating…';$('speedup').textContent='—';
  clearTimeout(state.timer);
  state.timer=setTimeout(()=>{
    const timing=benchmark(x,y,strength,steps);
    $('solverTime').textContent=`${timing.solver.toFixed(3)} ms`;
    $('modelTime').textContent=`${timing.model.toFixed(3)} ms`;
    $('speedup').textContent=`${(timing.solver/Math.max(timing.model,.00001)).toFixed(1)}×`;
  },50);
}

for(const input of Object.values(controls))input.addEventListener('input',update);
loadModel().then(update).catch(error=>{
  document.querySelector('.lab-card').insertAdjacentHTML('afterbegin',`<p class="load-error">The trained model could not be loaded. Serve this folder through a local web server so the browser can load model.json.</p>`);
  console.error(error);
});
