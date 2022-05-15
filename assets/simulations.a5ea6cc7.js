import{a as v,r as c,q as d,P as g,C as x,c as r,l as u,v as e,U as _,V as p,F as h,x as b,e as y,f as S}from"./vendor.fd246b83.js";import{u as j}from"./index.de39d3cb.js";import{_ as B}from"./ProjectListItem.4411ff36.js";import"./ProjectVideo.108bb0aa.js";import"./index.a64afb2b.js";import"./index.fb1087b0.js";const V={class:"w-full max-w-lg"},k={class:"flex 0-sm:flex-col sm:items-center gap-1/2 p-1/2 bg-gradient-to-b from-neutral-850 to-neutral-875"},w=e("h2",{class:"flex-1 font-bold text-3xl"},"Simulations",-1),C={class:"flex flex-col gap-1/4"},D=e("div",{class:"text-sm leading-1/2 font-bold"},"Sorting",-1),T={class:"flex items-center gap-1/2"},U=e("option",{value:"start_date"},"Start Date",-1),F=e("option",{value:"name"},"Name",-1),N=[U,F],P=e("option",{value:"asc"},"Ascending",-1),q=e("option",{value:"desc"},"Descending",-1),A=[P,q],E={class:"flex flex-col gap-1/2 px-1 py-1-1/2 bg-gradient-feather-y"},I=v({setup(H){const{simulation_projects:f}=j(),a=c("start_date"),n=c("desc"),l=d(()=>n.value=="asc"?1:-1);g(a,o=>{o=="start_date"&&(n.value="desc"),o=="name"&&(n.value="asc")});const m=d(()=>[...f].sort((o,s)=>{const t=o[a.value],i=s[a.value];return typeof t=="object"&&typeof t.getTime=="function"?o.start_date?s.start_date?(o.start_date.getTime()-s.start_date.getTime())*l.value:1*l.value:-1*l.value:typeof t=="string"?t.localeCompare(i,"en")*l.value:t-i}));return x({meta:[{name:"description",content:"Simulations by Jeff Schofield."}]}),(o,s)=>(r(),u("section",V,[e("div",k,[w,e("div",C,[D,e("div",T,[_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[0]||(s[0]=t=>a.value=t)},N,512),[[p,a.value]]),_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[1]||(s[1]=t=>n.value=t)},A,512),[[p,n.value]])])])]),e("div",E,[(r(!0),u(h,null,b(S(m),t=>(r(),y(B,{project:t},null,8,["project"]))),256))])]))}});export{I as default};
