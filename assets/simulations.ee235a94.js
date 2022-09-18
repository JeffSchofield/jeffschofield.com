import{e as v,m as g,f as c,g as d,A as x,u as h,o as r,c as u,a as e,K as _,L as m,F as b,l as y,i as S,h as j}from"./index.842c1b90.js";import{_ as B}from"./ProjectListItem.0737abd9.js";import"./ProjectImage.9a3ce32b.js";import"./_commonjsHelpers.b8add541.js";import"./index.61f96e6d.js";import"./index.3746861a.js";const k={class:"mx-auto w-full max-w-lg pointer-events-auto bg-neutral-900 rounded-t-1/2 border border-neutral-900"},w={class:"flex 0-sm:flex-col sm:items-center gap-1/2 p-1/2 bg-gradient-to-b from-neutral-850 to-neutral-875 sm:rounded-t-1/2"},D=e("h2",{class:"flex-1 sm:self-end font-bold text-3xl"},"Simulations",-1),T={class:"flex flex-col gap-1/4"},V=e("div",{class:"text-sm leading-1/2 font-bold"},"Sorting",-1),A={class:"flex items-center gap-1/2"},C=e("option",{value:"start_date"},"Start Date",-1),F=e("option",{value:"name"},"Name",-1),L=[C,F],N=e("option",{value:"asc"},"Ascending",-1),U=e("option",{value:"desc"},"Descending",-1),E=[N,U],H={class:"flex flex-col gap-1/2 px-1 py-1-1/2 bg-gradient-feather-y"},G=v({__name:"simulations",setup(J){const{simulation_projects:p}=g(),a=c("start_date"),n=c("desc"),l=d(()=>n.value=="asc"?1:-1);x(a,o=>{o=="start_date"&&(n.value="desc"),o=="name"&&(n.value="asc")});const f=d(()=>[...p].sort((o,s)=>{const t=o[a.value],i=s[a.value];return typeof t=="object"&&typeof t.getTime=="function"?o.start_date?s.start_date?(o.start_date.getTime()-s.start_date.getTime())*l.value:1*l.value:-1*l.value:typeof t=="string"?t.localeCompare(i,"en")*l.value:t-i}));return h({meta:[{name:"description",content:"Simulations by Jeff Schofield."}]}),(o,s)=>(r(),u("section",k,[e("div",w,[D,e("div",T,[V,e("div",A,[_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[0]||(s[0]=t=>a.value=t)},L,512),[[m,a.value]]),_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[1]||(s[1]=t=>n.value=t)},E,512),[[m,n.value]])])])]),e("div",H,[(r(!0),u(b,null,y(j(f),t=>(r(),S(B,{project:t},null,8,["project"]))),256))])]))}});export{G as default};