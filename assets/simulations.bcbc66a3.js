import{e as v,l as g,f as c,g as d,z as x,u as h,o as r,c as u,a as e,L as _,M as m,F as b,k as y,i as S,h as j}from"./index.b27f00f4.js";import{_ as k}from"./ProjectListItem.de009b65.js";import"./ProjectImage.a8861991.js";import"./index.61f96e6d.js";import"./index.3746861a.js";const B={class:"mx-auto w-full max-w-lg pointer-events-auto bg-neutral-900 rounded-t-1/2 border border-neutral-900"},w={class:"flex 0-sm:flex-col sm:items-center gap-1/2 p-1/2 bg-gradient-to-b from-neutral-850 to-neutral-875 sm:rounded-t-1/2"},D=e("h2",{class:"flex-1 sm:self-end font-bold text-3xl"},"Simulations",-1),T={class:"flex flex-col gap-1/4"},V=e("div",{class:"text-sm leading-1/2 font-bold"},"Sorting",-1),C={class:"flex items-center gap-1/2"},F=e("option",{value:"start_date"},"Start Date",-1),L=e("option",{value:"name"},"Name",-1),M=[F,L],N=e("option",{value:"asc"},"Ascending",-1),U=e("option",{value:"desc"},"Descending",-1),z=[N,U],A={class:"flex flex-col gap-1/2 px-1 py-1-1/2 bg-gradient-feather-y"},G=v({__name:"simulations",setup(E){const{simulation_projects:p}=g(),a=c("start_date"),n=c("desc"),l=d(()=>n.value=="asc"?1:-1);x(a,o=>{o=="start_date"&&(n.value="desc"),o=="name"&&(n.value="asc")});const f=d(()=>[...p].sort((o,s)=>{const t=o[a.value],i=s[a.value];return typeof t=="object"&&typeof t.getTime=="function"?o.start_date?s.start_date?(o.start_date.getTime()-s.start_date.getTime())*l.value:1*l.value:-1*l.value:typeof t=="string"?t.localeCompare(i,"en")*l.value:t-i}));return h({meta:[{name:"description",content:"Simulations by Jeff Schofield."}]}),(o,s)=>(r(),u("section",B,[e("div",w,[D,e("div",T,[V,e("div",C,[_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[0]||(s[0]=t=>a.value=t)},M,512),[[m,a.value]]),_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[1]||(s[1]=t=>n.value=t)},z,512),[[m,n.value]])])])]),e("div",A,[(r(!0),u(b,null,y(j(f),t=>(r(),S(k,{project:t},null,8,["project"]))),256))])]))}});export{G as default};