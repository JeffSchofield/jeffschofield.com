import{e as h,g as f,o as t,c as s,b as a,h as n,H as x,a as o,w as r,d,t as l,W as _,j as b,n as g,X as k,F as p,l as C}from"./index.6f3d243b.js";const v={class:"fixed inset-0 flex items-center justify-center min-h-screen z-40"},w={class:"max-w-sm min-w-xs rounded-1/2 shadow-md z-50"},y={key:0,class:"p-1/2 rounded-t-inherit bg-neutral-775"},N={class:"flex gap-1/2 p-1/2 rounded-b-inherit bg-neutral-850"},V=o("div",{class:"flex-1"},null,-1),z=["onClick"],F=h({__name:"GenericDialog",props:{title:null,content:null,buttons:null},emits:["close"],setup(e,{emit:u}){const c=e,m=f(()=>c.buttons?c.buttons:[{text:"Okay",click(){u("close")}}]);return(B,j)=>(t(),s("div",v,[a(n(x),{class:"fixed inset-0 bg-black opacity-30"}),o("div",w,[e.title?(t(),s("div",y,[a(n(_),{as:"h2"},{default:r(()=>[d(l(e.title),1)]),_:1})])):b("",!0),a(n(k),{class:g(["p-1 text-lg bg-neutral-825",{"rounded-t-inherit":!e.title}])},{default:r(()=>[d(l(e.content),1)]),_:1},8,["class"]),o("div",N,[V,(t(!0),s(p,null,C(n(m),i=>(t(),s("button",{onClick:i.click},l(i.text),9,z))),256))])])]))}});export{F as default};
