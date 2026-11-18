// Approximate inverse TM screening only; Bessel datum shift is not applied.
function inverse(x,y,c){let [a,f,lon0,lat0,fe,fn]=c,e=f*(2-f),ep=e/(1-e),rad=Math.PI/180;
const mer=p=>a*((1-e/4-3*e**2/64-5*e**3/256)*p-(3*e/8+3*e**2/32+45*e**3/1024)*Math.sin(2*p)+(15*e**2/256+45*e**3/1024)*Math.sin(4*p)-35*e**3/3072*Math.sin(6*p));
let M=mer(lat0*rad)+y-fn,mu=M/(a*(1-e/4-3*e**2/64-5*e**3/256)),e1=(1-Math.sqrt(1-e))/(1+Math.sqrt(1-e));
let p=mu+(3*e1/2-27*e1**3/32)*Math.sin(2*mu)+(21*e1**2/16-55*e1**4/32)*Math.sin(4*mu)+151*e1**3/96*Math.sin(6*mu)+1097*e1**4/512*Math.sin(8*mu);
let n=a/Math.sqrt(1-e*Math.sin(p)**2),t=Math.tan(p)**2,C=ep*Math.cos(p)**2,R=a*(1-e)/(1-e*Math.sin(p)**2)**1.5,D=(x-fe)/n;
return [(lon0*rad+(D-(1+2*t+C)*D**3/6+(5-2*C+28*t-3*C*C+8*ep+24*t*t)*D**5/120)/Math.cos(p))/rad,(p-n*Math.tan(p)/R*(D*D/2-(5+3*t+10*C-4*C*C-9*ep)*D**4/24+(61+90*t+298*C+45*t*t-252*ep-3*C*C)*D**6/720))/rad];}
export function spatial(rows,cafe){let cs={'5174':[6377397.155,1/299.1528128,127.002890277778,38,200000,500000],'2097':[6377397.155,1/299.1528128,127,38,200000,500000],'5181':[6378137,1/298.257222101,127,38,200000,500000],'5186':[6378137,1/298.257222101,127,38,200000,600000]};
const valid=r=>r['좌표정보(X)']&&r['좌표정보(Y)']&&Number.isFinite(+r['좌표정보(X)'])&&Number.isFinite(+r['좌표정보(Y)']);
const inside=(r,c)=>{let [x,y]=inverse(+r['좌표정보(X)'],+r['좌표정보(Y)'],c);return x>=126.76&&x<=127.19&&y>=37.42&&y<=37.71;};
return Object.entries(cs).map(([k,c])=>{let a=rows.filter(valid),n=a.filter(r=>inside(r,c)).length;return[k,a.length,n,a.length-n,cafe.filter(valid).filter(r=>!inside(r,c)).length];});}
