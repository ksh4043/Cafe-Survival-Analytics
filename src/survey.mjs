import fs from 'node:fs';
import crypto from 'node:crypto';
export const SOURCE='E:/projects/DataAnalysis/data/식품_휴게음식점_서울특별시.csv';
const ROOT='E:/projects/DataAnalysis',TODAY='2026-11-12',PAGE='2025-11-27';
const bytes=fs.readFileSync(SOURCE),hash=crypto.createHash('sha256').update(bytes).digest('hex');
let utf8=true;try{new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{utf8=false;}
const {execFileSync}=await import('node:child_process');
const tmp=ROOT+'/data/processed/survey_decoded.tmp';
execFileSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT+'/src/decode_cp949.ps1',SOURCE,tmp]);
const text=fs.readFileSync(tmp,'utf8');fs.unlinkSync(tmp);
function csv(s){let rows=[],row=[],cell='',q=false;for(let i=0;i<s.length;i++){let c=s[i];if(c==='"'){if(q&&s[i+1]==='"'){cell+='"';i++;}else q=!q;}else if(c===','&&!q){row.push(cell);cell='';}else if((c==='\n'||c==='\r')&&!q){if(c==='\r'&&s[i+1]==='\n')i++;row.push(cell);if(row.some(Boolean))rows.push(row);row=[];cell='';}else cell+=c;}if(cell||row.length){row.push(cell);rows.push(row);}if(q)throw Error('Unclosed quote');return rows;}
const raw=csv(text),headers=raw.shift();if(raw.some(r=>r.length!==headers.length))throw Error('Column width mismatch');
const rows=raw.map(r=>Object.fromEntries(headers.map((h,i)=>[h,r[i].trim()]))),N=rows.length;
const num=v=>v?Number(v.replaceAll(',','')):NaN;
const count=(rs,key)=>{let m=new Map();for(let r of rs){let k=typeof key==='function'?key(r):(r[key]||'(결측)');m.set(k,(m.get(k)||0)+1);}return [...m].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0],'ko'));};
const pct=(n,d=N)=>d?(100*n/d).toFixed(2)+'%':'해당 없음';
const cand=r=>/커피|다방|찻집|카페|기타 휴게/.test(r.업태구분명),cafe=rows.filter(cand),named=r=>/카페|커피|coffee|cafe/i.test(r.사업장명);
const closed=r=>/폐업/.test(r.영업상태명+' '+r.상세영업상태명),active=r=>/영업|정상/.test(r.영업상태명+' '+r.상세영업상태명)&&!closed(r);
function dateUncached(v){let m=(v||'').match(/^(\d{4})[-.]?(\d{2})[-.]?(\d{2})/);if(!m)return '';let s=m.slice(1).join('-'),d=new Date(s+'T00:00:00Z');return Number.isFinite(+d)&&d.toISOString().slice(0,10)===s?s:'';}
const dateCache=new Map();
function date(v){if(!dateCache.has(v))dateCache.set(v,dateUncached(v));return dateCache.get(v);}
const xy=r=>Number.isFinite(num(r['좌표정보(X)']))&&Number.isFinite(num(r['좌표정보(Y)']));
let out=['# 01 개발 환경 및 인허가 파일 조사','', '조사일 '+TODAY+'. 정의 확정·표본 제외 없음. 전체 '+N+'행 유지. 원본 위치·내용 보존.','','## 입력 및 재현','', '- 입력: '+SOURCE,'- 크기: '+bytes.length+'바이트; SHA256: '+hash,'- UTF-8 엄격 해독 성공: '+utf8+'; CP949 .NET 엄격 해독 및 원바이트 roundtrip SHA256 일치, 한글 헤더 확인.','- '+headers.length+'컬럼; CSV 인용부호·열 수 검사 통과.','- 단계: 읽기 → CSV 파싱 → 앞뒤 공백 trim → 기술통계. 모든 단계 '+N+'행, 제외 0건. 원본 변경 없음.','- 실행: node src/survey.mjs (Node 20 이상). 난수 사용 없음.',''];
const table=(s,hs,rs)=>out.push('## '+s,'','| '+hs.join(' | ')+' |','| '+hs.map(()=> '---').join(' | ')+' |',...rs.map(r=>'| '+r.map(v=>String(v??'').replaceAll('|','\\|').replaceAll('\n',' ')).join(' | ')+' |'),'');
table('컬럼 및 결측률',['컬럼','결측','결측률'],headers.map(h=>{let n=rows.filter(r=>!r[h]).length;return[h,n,pct(n)];}));
let ids=count(rows,'관리번호');table('관리번호 중복',['항목','건수'],[['결측',rows.filter(r=>!r.관리번호).length],['고유값',ids.length],['중복 그룹',ids.filter(x=>x[1]>1).length],['중복 초과 행',ids.reduce((s,x)=>s+x[1]-1,0)]]);
for(let h of ['업태구분명','위생업태명'])table(h+' 전체 값',['값','건수'],count(rows,h));
table('카페 관련 업태 후보 (확정 아님)',['업태','N','이름 키워드 포함'],count(cafe,'업태구분명').map(([k,n])=>[k,n,rows.filter(r=>r.업태구분명===k&&named(r)).length]));
table('사업장명 키워드 × 전체 업태',['업태','N','카페/커피/coffee/cafe 포함'],count(rows,'업태구분명').map(([k,n])=>[k,n,rows.filter(r=>(r.업태구분명||'(결측)')===k&&named(r)).length]));
table('상태 코드·명칭 조합',['영업코드 / 명 / 상세코드 / 명','건수'],count(rows,r=>[r.영업상태코드,r.영업상태명,r.상세영업상태코드,r.상세영업상태명].join(' / ')));
const fields=['인허가일자','폐업일자','데이터갱신시점','최종수정시점'];
table('날짜 범위·이상',['컬럼','결측','비결측 파싱 실패','최소 원문','최대 원문','조사일 이후','페이지 수정일 이후'],fields.map(h=>{let a=rows.map(r=>r[h]).filter(Boolean).sort();return[h,N-a.length,a.filter(v=>!date(v)).length,a[0],a.at(-1),a.filter(v=>date(v)>TODAY).length,a.filter(v=>date(v)>PAGE).length];}));
table('상태·날짜 모순',['항목','전체','카페 후보'],[['폐업 상태인데 폐업일 없음',r=>closed(r)&&!r.폐업일자],['영업/정상인데 폐업일 있음',r=>active(r)&&!!r.폐업일자],['폐업일 < 인허가일',r=>date(r.폐업일자)&&date(r.인허가일자)&&date(r.폐업일자)<date(r.인허가일자)]].map(([s,f])=>[s,rows.filter(f).length,cafe.filter(f).length]));
out.push('페이지 수정일 2025-11-27은 사용자 제공값이며 온라인 재조회하지 않았다. 갱신시점은 사건일 아닌 배포·갱신시각일 수 있다. 페이지 수정일 이후 갱신과 사건일을 구분해야 하며 다운로드일까지 관측이 완전하다는 근거는 없다.','');
const years=[...new Set(cafe.flatMap(r=>[date(r.인허가일자).slice(0,4),date(r.폐업일자).slice(0,4)]).filter(Boolean))].sort();
table('카페 후보 연도별 개업·폐업·좌표 누락',['연도','개업','폐업일 기준 폐업','개업 중 좌표 없음','누락률'],years.map(y=>{let a=cafe.filter(r=>date(r.인허가일자).startsWith(y)),n=a.filter(r=>!xy(r)).length;return[y,a.length,cafe.filter(r=>date(r.폐업일자).startsWith(y)).length,n,pct(n,a.length)];}));
const stats=vs=>{let a=vs.filter(Number.isFinite).sort((a,b)=>a-b),q=p=>a.length?a[Math.floor((a.length-1)*p)]:'';return[a.length,a.filter(v=>v===0).length,a.filter(v=>v<0).length,q(0),q(.25),q(.5),q(.75),q(.95),q(1)];};
table('면적·임대료 분포 (단위 미확인, 분위수는 하위 순위)',['항목','숫자 N','0','음수','최소','Q1','중앙값','Q3','P95','최대'],['소재지면적','시설총규모','월세액','보증액'].map(h=>[h,...stats(rows.map(r=>num(r[h])))]));
let both=rows.filter(r=>Number.isFinite(num(r.소재지면적))&&Number.isFinite(num(r.시설총규모))),eq=both.filter(r=>num(r.소재지면적)===num(r.시설총규모)).length;
table('면적 결측·일치',['항목','N','비율'],['소재지면적','시설총규모'].map(h=>[h+' 결측',rows.filter(r=>!r[h]).length,pct(rows.filter(r=>!r[h]).length)]).concat([['양쪽 숫자',both.length,pct(both.length)],['양쪽 숫자 중 정확히 일치',eq,pct(eq,both.length)]]));
table('임대료 채움률',['범위','N','월세 비결측','비율','월세 양수','보증 비결측','비율','보증 양수'],[['전체',rows],['카페 후보',cafe],...years.map(y=>[y+' 후보 개업',cafe.filter(r=>date(r.인허가일자).startsWith(y))])].map(([s,a])=>[s,a.length,a.filter(r=>r.월세액).length,pct(a.filter(r=>r.월세액).length,a.length),a.filter(r=>num(r.월세액)>0).length,a.filter(r=>r.보증액).length,pct(a.filter(r=>r.보증액).length,a.length),a.filter(r=>num(r.보증액)>0).length]));
out.push('임대료 0은 무상인지 미기재 대체값인지 불명이다. 양수 관측이 0건이고 변이가 없어 이 파일의 월세액·보증액으로 임대료 효과를 추정할 수 없다. 금액 단위·기록시점·개업 당시 값 여부도 불명이다.','');
table('좌표 품질',['항목','전체','카페 후보'],[['X 결측',r=>!r['좌표정보(X)']],['Y 결측',r=>!r['좌표정보(Y)']],['누락/비숫자',r=>!xy(r)],['0 좌표',r=>xy(r)&&(num(r['좌표정보(X)'])===0||num(r['좌표정보(Y)'])===0)]].map(([s,f])=>[s,rows.filter(f).length,cafe.filter(f).length]));
table('좌표 원문 공백',['컬럼','앞뒤 공백 있는 행'],['좌표정보(X)','좌표정보(Y)'].map(h=>[h,raw.filter(r=>r[headers.indexOf(h)]!==r[headers.indexOf(h)].trim()).length]));
for(let [label,key] of [['동일 좌표',r=>xy(r)?num(r['좌표정보(X)'])+'|'+num(r['좌표정보(Y)']):''],['동일 지번주소',r=>r.지번주소],['동일 도로명주소',r=>r.도로명주소]]){let g=new Map();for(let r of cafe){let k=key(r);if(k){if(!g.has(k))g.set(k,[]);g.get(k).push(r);}}let multi=[...g.values()].filter(a=>a.length>1),after=multi.filter(a=>a.some(r=>date(r.폐업일자)&&a.some(s=>s!==r&&date(s.인허가일자)>date(r.폐업일자)))),same=multi.filter(a=>a.some(r=>date(r.폐업일자)&&a.some(s=>s!==r&&date(s.인허가일자)===date(r.폐업일자))));table(label+' 반복 (카페 후보)',['단서','그룹','그룹 내 행'],[['복수 인허가',multi.length,multi.flat().length],['폐업 후 다른 인허가',after.length,after.flat().length],['폐업일=다른 인허가일',same.length,same.flat().length]]);}
out.push('주소는 건물 단위, 좌표는 대표점 중복일 수 있다. 주소는 trim 후 문자열 일치, 좌표는 숫자로 변환한 정확한 일치를 기준으로 묶었다. 위 건수는 승계 확정이 아니다. 소유자·승계일·이전 관리번호·지위승계 사유 전용 컬럼 없음. 실제 양수도 건수는 식별 불가(0건 의미 아님). 동일일 단서는 오탐·누락 가능성이 있다.','');
table('카페 후보 사업장명 상위 50',['사업장명','건수'],count(cafe,'사업장명').slice(0,50));
out.push('프랜차이즈 전용 컬럼 없음. 브랜드명 사전 매칭은 가능하나 지점명·띄어쓰기·과거 상호·동명 독립점 검증 필요. 현재 상호로 개업 당시 가맹 여부 소급 확정 불가.','');
table('결정 필요: 카페 정의',['선택지','N','폐업 상태','폐업일 있음'],[['커피숍만',r=>r.업태구분명==='커피숍'],['커피숍+다방',r=>['커피숍','다방'].includes(r.업태구분명)],['카페 관련 업태 후보',cand],['이름 키워드 (전체 업태)',named],['후보 업태 ∪ 이름 키워드',r=>cand(r)||named(r)]].map(([s,f])=>{let a=rows.filter(f);return[s,a.length,a.filter(closed).length,a.filter(r=>r.폐업일자).length];}));
out.push('커피숍은 비커피숍 카페 누락 가능, 다방 포함은 전통 음료업 확장, 기타 휴게음식점 포함은 비카페 혼입 가능. 이름 키워드는 누락·오탐 가능. 후보는 임시 기술통계 묶음이며 정의를 채택한 것이 아니다.','');
table('결정 필요: 폐업·행정처분',['관측 단서/선택지','전체','카페 후보'],[['상태에 폐업 포함',closed],['폐업 상태 + 유효 폐업일',r=>closed(r)&&!!date(r.폐업일자)],['유효 폐업일 (상태 불문)',r=>!!date(r.폐업일자)],['휴업 명시',r=>/휴업/.test(r.영업상태명+' '+r.상세영업상태명)],['직권폐업 명시',r=>/직권/.test(r.영업상태명+' '+r.상세영업상태명)],['취소/말소/만료 명시',r=>/취소|말소|만료/.test(r.영업상태명+' '+r.상세영업상태명)]].map(([s,f])=>[s,rows.filter(f).length,cafe.filter(f).length]));
out.push('폐업으로만 표시되면 자진·직권폐업 분리 불가. 명시 0건은 실제 행정처분 0건을 뜻하지 않는다. 휴업은 사건/검열/제외 선택 필요. 취소·말소는 날짜 의미 확인 필요. 양수도는 점포 지속/인허가 종료 중 분석 단위 결정이 필요하며 확정 식별 불가.','');
const ends=[PAGE,...fields.map(h=>rows.map(r=>date(r[h])).filter(Boolean).sort().at(-1)),TODAY].filter((x,i,a)=>x&&a.indexOf(x)===i).sort();
table('결정 필요: 관측 종료일',['종료일 후보','후보 개업 ≤ 종료','후보 폐업일 ≤ 종료','후보 개업 > 종료','후보 폐업일 > 종료'],ends.map(d=>[d,cafe.filter(r=>date(r.인허가일자)&&date(r.인허가일자)<=d).length,cafe.filter(r=>date(r.폐업일자)&&date(r.폐업일자)<=d).length,cafe.filter(r=>date(r.인허가일자)>d).length,cafe.filter(r=>date(r.폐업일자)>d).length]));
table('결정 필요: 개업 기간 (종료일 미확정)',['범위','후보 N','폐업 상태','좌표 유효'],[['전체 유효일','0000'],['2010년 이후','2010'],['2015년 이후','2015'],['2020년 이후','2020']].map(([s,y])=>{let a=cafe.filter(r=>date(r.인허가일자)&&date(r.인허가일자).slice(0,4)>=y);return[s,a.length,a.filter(closed).length,a.filter(xy).length];}));
out.push('연도 절단은 선택지 예시일 뿐 채택하지 않았다. 개업 이전 공변량 제공 시작연도를 확인하고 결정해야 한다. 최대 사건일은 모든 점포의 공통 관측 종료일을 보장하지 않는다.','');
try{const {spatial}=await import('./survey_spatial.mjs');table('좌표계 후보 근사 검증',['EPSG','숫자 좌표 N','서울 근사 범위 안','범위 밖','후보 중 범위 밖'],spatial(rows,cafe));}catch(e){out.push('좌표 변환 미완료: '+e.message,'');}
out.push('서울 근사 사각형은 경도 126.76–127.19, 위도 37.42–37.71이다. 로컬 경계 파일이 없어 실제 서울 밖 건수 확정 불가. 5174/2097은 Bessel 역투영이며 WGS84 datum 변환이 생략된 근사값이다. 5181/5186도 pyproj 검증 전이다. 범위 검사는 원점·축 오류 탐지용이며 근접 CRS를 판별하지 못한다. 공식 명세/기준점과 pyproj 검증 필요.','','## 환경 실패 및 미완료','','PATH에는 WindowsApps Python 스텁만 존재하며 py/uv/conda/git을 찾지 못했다. 일반 설치 위치 검색도 실제 Python/Git을 찾지 못했다. uv 설치용 GitHub/PyPI HTTPS가 PowerShell·curl·Node에서 실패했다(Node PyPI 연결 오류: EACCES TCP 443). winget도 실행 접근에 실패했다. git init, .venv 생성, 요청 패키지 설치·버전 고정은 미완료. requirements.in은 설치 대상 목록이며 버전 고정 파일이 아니다. 네트워크 데이터 다운로드 없음. .gitignore와 요청 폴더는 생성했다.');

table('업태구분명·위생업태명 불일치',['업태구분명','위생업태명','N'],count(rows.filter(r=>r.업태구분명!==r.위생업태명),r=>r.업태구분명+' / '+r.위생업태명).map(([k,n])=>[...k.split(' / '),n]));
table('추가 카페 정의 선택지',['범위','N'],[['커피숍+다방+전통찻집',r=>['커피숍','다방','전통찻집'].includes(r.업태구분명)],['위 범위+떡카페+키즈카페',r=>['커피숍','다방','전통찻집','떡카페','키즈카페'].includes(r.업태구분명)],['기타 휴게음식점 중 이름 키워드',r=>r.업태구분명==='기타 휴게음식점'&&named(r)],['커피숍+다방+전통찻집+떡카페+키즈카페 및 기타 중 이름 키워드',r=>['커피숍','다방','전통찻집','떡카페','키즈카페'].includes(r.업태구분명)||(r.업태구분명==='기타 휴게음식점'&&named(r))]].map(([s,f])=>[s,rows.filter(f).length]));
table('검토 필요: 오래된 인허가일 (자동 제외 아님)',['범위','전체','후보'],[['1950년 이전','1950'],['1970년 이전','1970']].map(([s,y])=>[s,rows.filter(r=>date(r.인허가일자)&&date(r.인허가일자).slice(0,4)<y).length,cafe.filter(r=>date(r.인허가일자)&&date(r.인허가일자).slice(0,4)<y).length]));
table('후보 임대료 분포',['항목','숫자 N','0','음수','최소','Q1','중앙값','Q3','P95','최대'],['월세액','보증액'].map(h=>[h,...stats(cafe.map(r=>num(r[h])))]));
table('카페 후보 업태별·연도별 개업/폐업',['연도','업태','개업','폐업일 기준 폐업'],years.flatMap(y=>count(cafe,'업태구분명').map(([k])=>[y,k,cafe.filter(r=>r.업태구분명===k&&date(r.인허가일자).startsWith(y)).length,cafe.filter(r=>r.업태구분명===k&&date(r.폐업일자).startsWith(y)).length])));

fs.writeFileSync(ROOT+'/outputs/01_data_survey.md',out.join('\n'),'utf8');
fs.writeFileSync(ROOT+'/data/SOURCES.md','# 데이터 출처\n\n- 데이터명: 식품_휴게음식점 (서울특별시, 지역별 CSV)\n- 다운로드 페이지: https://file.localdata.go.kr/file/rest_cafes/info\n- 페이지 표시 수정일: 2025-11-27 (사용자 제공)\n- 다운로드일: 2026-11-12 (사용자가 수동 다운로드)\n- 파일: 식품_휴게음식점_서울특별시.csv (원위치 보존)\n- 크기: '+bytes.length+'바이트\n- 행 수: '+N+' (헤더 제외)\n- 인코딩: Windows-949/CP949 (.NET 엄격 해독 및 roundtrip SHA256 일치, UTF-8 실패)\n- SHA256: '+hash+'\n','utf8');
if(crypto.createHash('sha256').update(fs.readFileSync(SOURCE)).digest('hex')!==hash)throw Error('Source changed');
console.log(JSON.stringify({rows:N,columns:headers.length,cafeCandidates:cafe.length,hash}));
