import{q as x,s as k,v as B,x as L,y as T,f as y,z as V,h as u,e as _,g as O,A as W,o,c as l,B as P,j as p,F as m,k as j,a as N,C as D,n as F,i as M}from"./index.b27f00f4.js";function R(e){return L()?(T(e),!0):!1}const c=typeof window!="undefined",U=e=>typeof e=="string",w=()=>{};function q(e,t=!0){x()?k(e):t?e():B(e)}const E=c?window:void 0;c&&window.document;c&&window.navigator;c&&window.location;function A(...e){let t,n,s,r;if(U(e[0])?([n,s,r]=e,t=E):[t,n,s,r]=e,!t)return w;let i=w;const d=V(()=>u(t),f=>{i(),f&&(f.addEventListener(n,s,r),i=()=>{f.removeEventListener(n,s,r),i=w})},{immediate:!0,flush:"post"}),a=()=>{d(),i()};return R(a),a}const v=typeof globalThis!="undefined"?globalThis:typeof window!="undefined"?window:typeof global!="undefined"?global:typeof self!="undefined"?self:{},b="__vueuse_ssr_handlers__";v[b]=v[b]||{};v[b];var S,C;c&&(window==null?void 0:window.navigator)&&((S=window==null?void 0:window.navigator)==null?void 0:S.platform)&&/iP(ad|hone|od)/.test((C=window==null?void 0:window.navigator)==null?void 0:C.platform);var H=Object.defineProperty,I=Object.getOwnPropertySymbols,K=Object.prototype.hasOwnProperty,Y=Object.prototype.propertyIsEnumerable,z=(e,t,n)=>t in e?H(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,G=(e,t)=>{for(var n in t||(t={}))K.call(t,n)&&z(e,n,t[n]);if(I)for(var n of I(t))Y.call(t,n)&&z(e,n,t[n]);return e};const J={top:0,left:0,bottom:0,right:0,height:0,width:0};G({text:""},J);function Q({window:e=E,initialWidth:t=1/0,initialHeight:n=1/0}={}){const s=y(t),r=y(n),i=()=>{e&&(s.value=e.innerWidth,r.value=e.innerHeight)};return i(),q(i),A("resize",i,{passive:!0}),{width:s,height:r}}const X=["src"],Z=["src"],ie=_({__name:"ProjectVideo",props:{src:null},setup(e){const t=e,{width:n,height:s}=Q(),r=y(),i=O(()=>{if(typeof t.src=="string")try{return new URL(t.src)}catch{}}),d=O(()=>i.value&&i.value.host.match(/youtube/g)?"youtube":"local");return k(()=>{r.value}),W(()=>{}),(a,f)=>{var $;return o(),l(m,null,[u(d)=="youtube"?(o(),l("iframe",P({key:0},a.$attrs,{src:`https://youtube.com/embed/${($=u(i))==null?void 0:$.searchParams.get("v")}`,title:"YouTube video player",frameborder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowfullscreen:""}),null,16,X)):p("",!0),u(d)=="local"?(o(),l("video",P({key:1,ref_key:"video",ref:r},a.$attrs,{autoplay:"",loop:"",muted:""}),[(o(!0),l(m,null,j(Object.entries(e.src).sort((h,g)=>parseInt(g[0])-parseInt(h[0])),([h,g])=>(o(),l(m,null,[u(n)>parseInt(h)?(o(),l("source",{key:0,src:g,type:"video/mp4"},null,8,Z)):p("",!0)],64))),256))],16)):p("",!0)],64)}}}),ee=["srcset","sizes","type"],te=["src","alt","width","height"],ne=_({__name:"Image",props:{sources:{default:{}},width:null,height:null,sizes:{default:""},src:null,alt:null,imgClass:null,imgStyle:null},setup(e){return(t,n)=>(o(),l("picture",null,[(o(!0),l(m,null,j(Object.entries(e.sources),([s,r])=>(o(),l("source",{srcset:r,sizes:e.sizes,type:`image/${s}`},null,8,ee))),256)),N("img",{src:e.src,alt:e.alt,width:e.width,height:e.height,class:D(`rounded-inherit-parent w-full h-full ${e.imgClass}`),style:F(e.imgStyle)},null,14,te)]))}}),se=_({__name:"ProjectImage",props:{image:null,alt:null,imgClass:null,imgStyle:null},setup(e){return(t,n)=>(o(),M(ne,{sources:e.image.sources,src:e.image.fallback,width:e.image.width,height:e.image.height,alt:e.alt,"img-class":e.imgClass,"img-style":e.imgStyle},null,8,["sources","src","width","height","alt","img-class","img-style"]))}});export{ie as _,se as a};
