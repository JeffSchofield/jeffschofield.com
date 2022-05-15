import{a as v,r as i,q as d,P as g,C as x,c as l,l as u,v as e,U as _,V as f,F as h,x as b,e as y,f as S}from"./vendor.fd246b83.js";import{u as w}from"./index.de39d3cb.js";import{_ as j}from"./ProjectListItem.4411ff36.js";import"./ProjectVideo.108bb0aa.js";import"./index.a64afb2b.js";import"./index.fb1087b0.js";const B={class:"w-full max-w-lg"},V={class:"flex 0-sm:flex-col sm:items-center gap-1/2 p-1/2 bg-gradient-to-b from-neutral-850 to-neutral-875"},k=e("h2",{class:"flex-1 font-bold text-3xl"},"Software",-1),C={class:"flex flex-col gap-1/4"},D=e("div",{class:"text-sm leading-1/2 font-bold"},"Sorting",-1),T={class:"flex items-center gap-1/2"},U=e("option",{value:"start_date"},"Start Date",-1),F=e("option",{value:"name"},"Name",-1),N=[U,F],P=e("option",{value:"asc"},"Ascending",-1),q=e("option",{value:"desc"},"Descending",-1),A=[P,q],E={class:"flex flex-col gap-1/2 px-1 py-1-1/2 bg-gradient-feather-y"},I=v({setup(H){const{software_projects:p}=w(),a=i("start_date"),r=i("desc"),n=d(()=>r.value=="asc"?1:-1);g(a,o=>{o=="start_date"&&(r.value="desc"),o=="name"&&(r.value="asc")});const m=d(()=>[...p].sort((o,s)=>{const t=o[a.value],c=s[a.value];return typeof t=="object"&&typeof t.getTime=="function"?o.start_date?s.start_date?(o.start_date.getTime()-s.start_date.getTime())*n.value:1*n.value:-1*n.value:typeof t=="string"?t.localeCompare(c,"en")*n.value:t-c}));return x({meta:[{name:"description",content:"Software by Jeff Schofield."}]}),(o,s)=>(l(),u("section",B,[e("div",V,[k,e("div",C,[D,e("div",T,[_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[0]||(s[0]=t=>a.value=t)},N,512),[[f,a.value]]),_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[1]||(s[1]=t=>r.value=t)},A,512),[[f,r.value]])])])]),e("div",E,[(l(!0),u(h,null,b(S(m),t=>(l(),y(j,{project:t},null,8,["project"]))),256))])]))}});export{I as default};
