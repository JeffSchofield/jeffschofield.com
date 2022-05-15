import{a as v,r as i,q as d,P as g,C as x,c as n,l as u,v as e,U as _,V as p,F as h,x as b,e as y,f as S}from"./vendor.fd246b83.js";import{u as j}from"./index.de39d3cb.js";import{_ as w}from"./ProjectListItem.4411ff36.js";import"./ProjectVideo.108bb0aa.js";import"./index.a64afb2b.js";import"./index.fb1087b0.js";const B={class:"w-full max-w-lg"},V={class:"flex 0-sm:flex-col sm:items-center gap-1/2 p-1/2 bg-gradient-to-b from-neutral-850 to-neutral-875"},k=e("h2",{class:"flex-1 font-bold text-3xl"},"Software",-1),C={class:"flex flex-col gap-1/4"},D=e("div",{class:"text-sm leading-1/2 font-bold"},"Sorting",-1),T={class:"flex items-center gap-1/2"},U=e("option",{value:"start_date"},"Start Date",-1),A=e("option",{value:"name"},"Name",-1),F=[U,A],N=e("option",{value:"asc"},"Ascending",-1),P=e("option",{value:"desc"},"Descending",-1),q=[N,P],E={class:"flex flex-col gap-1/2 px-1 py-1-1/2 bg-gradient-feather-y"},I=v({setup(H){const{projects:f}=j(),a=i("start_date"),l=i("desc"),r=d(()=>l.value=="asc"?1:-1);g(a,o=>{o=="start_date"&&(l.value="desc"),o=="name"&&(l.value="asc")});const m=d(()=>[...f].sort((o,s)=>{const t=o[a.value],c=s[a.value];return typeof t=="object"&&typeof t.getTime=="function"?o.start_date?s.start_date?(o.start_date.getTime()-s.start_date.getTime())*r.value:1*r.value:-1*r.value:typeof t=="string"?t.localeCompare(c,"en")*r.value:t-c}));return x({meta:[{name:"description",content:"All portfolio projects by Jeff Schofield."}]}),(o,s)=>(n(),u("section",B,[e("div",V,[k,e("div",C,[D,e("div",T,[_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[0]||(s[0]=t=>a.value=t)},F,512),[[p,a.value]]),_(e("select",{class:"px-1/4 py-1/8 rounded-1/4 bg-neutral-800","onUpdate:modelValue":s[1]||(s[1]=t=>l.value=t)},q,512),[[p,l.value]])])])]),e("div",E,[(n(!0),u(h,null,b(S(m),t=>(n(),y(w,{project:t},null,8,["project"]))),256))])]))}});export{I as default};
