import{P as t}from"./index.7fc06961.js";import{_ as T,a as V}from"./ProjectImage.96af3af6.js";import{f as $}from"./index.a64afb2b.js";import{a as w,q as r,b as f,c as l,l as d,g as i,w as m,e as p,Q as a,f as o,k as u,v as s,I as g,D as c}from"./vendor.7fb21631.js";const W={key:0},U={class:"font-semibold text-xl leading-3/4"},F={key:0,class:"text-sm leading-1/2 text-neutral-400"},L=c("Released "),G={class:"font-semibold text-neutral-200"},K={key:1,class:"text-sm leading-1/2 text-neutral-400"},q=c("Started "),z={class:"font-semibold text-neutral-200"},H={class:"flex-1 flex flex-col gap-1/2 p-3/4 bg-neutral-925"},Q={class:"leading-1-1/4"},Y=s("div",{class:"flex-1"},null,-1),J={class:"p-1/2 flex gap-1/2"},X=c(" Repo"),Z=s("span",{class:"sr-only"},"sitory",-1),ee=c(" Visit"),te=c("Read More"),ie=w({props:{project:null,card:{type:Boolean,default:!1}},setup(e){const n=e,h=r(()=>`/portfolio/${n.project.path}${n.project.slug&&n.project.slug!="index"?`/${n.project.slug}`:""}`),x=$("MMMM yyyy"),v="flex sm:grid md:flex lg:grid flex-col grid-cols-3 0-sm:divide-y sm-md:divide-x md-lg:divide-y lg:divide-x",j="0-sm:rounded-t-inherit sm-md:rounded-l-inherit md-lg:rounded-t-inherit lg:rounded-l-inherit",A="0-sm:rounded-b-inherit sm-md:rounded-r-inherit md-lg:rounded-b-inherit lg:rounded-r-inherit",E="sm-md:rounded-tr-inherit lg:rounded-tr-inherit",S="0-sm:rounded-b-inherit sm-md:rounded-br-inherit md-lg:rounded-b-inherit lg:rounded-br-inherit",k="flex flex-col",R="rounded-t-inherit",I="rounded-b-inherit",B="rounded-b-inherit",C=r(()=>n.card?k:v),b=r(()=>n.card?R:j),N=r(()=>n.card?I:A),O=r(()=>n.card?"":E),M=r(()=>n.card?B:S),P=r(()=>n.card?"line-clamp-2":"line-clamp-3 sm:line-clamp-2 md:line-clamp-3 lg:line-clamp-2");return t.APP+"",t.ARTWORK+"",t.BACKEND+"",t.DEVOPS+"",t.FRONTEND+"",t.GAME+"",t.MUSIC+"",t.OPEN_SOURCE+"",t.SHADER+"",t.SIMULATION+"",t.SOFTWARE+"",t.USER_INTERFACE+"",t.VIDEO+"",t.WEBSITE+"",t.WEB_APPLICATION+"",(se,oe)=>{const D=f("RouterLink"),y=f("SvgIcon"),_=f("AbstractButton");return l(),d("div",{class:a([o(C),"rounded-1/2 border border-neutral-950 divide-neutral-950"])},[i(D,{to:o(h),class:a(["col-span-1 flex items-center justify-center",o(b)])},{default:m(()=>[e.card&&e.project.video?(l(),d("div",W,[i(T,{src:e.project.video,class:"rounded-t-inherit w-full aspect-video"},null,8,["src"])])):e.project.image?(l(),p(V,{key:1,class:a(["w-full h-full aspect-[4/3] flex",o(b)]),image:e.project.image,alt:`Preview image of ${e.project.name}`,"img-class":"object-cover","img-style":`object-position: ${e.project.image_focus[0]}% ${e.project.image_focus[1]}%;`},null,8,["image","class","alt","img-style"])):u("",!0)]),_:1},8,["to","class"]),s("div",{class:a(["col-span-2 flex flex-col",o(N)])},[s("div",{class:a(["p-3/4 flex flex-col gap-1/4 bg-neutral-875",o(O)])},[s("div",U,g(e.project.name),1),e.project.release_date?(l(),d("div",F,[L,s("span",G,g(o(x)(e.project.release_date)),1)])):e.project.start_date?(l(),d("div",K,[q,s("span",z,g(o(x)(e.project.start_date)),1)])):u("",!0)],2),s("div",H,[s("div",Q,[s("span",{class:a(o(P))},g(e.project.short_description),3)])]),s("div",{class:a(["flex-1 flex bg-neutral-875",o(M)])},[Y,s("div",J,[e.project.repo?(l(),p(_,{key:0,external:e.project.repo,class:"flex items-center gap-1/2 rounded-1/4 leading-3/4 px-1/2 py-1/2 bg-neutral-650 hover:bg-neutral-600"},{default:m(()=>[i(y,{name:"github",class:"h-[1em] fill-current"}),X,Z]),_:1},8,["external"])):u("",!0),e.project.url?(l(),p(_,{key:1,external:e.project.url,class:"flex items-center gap-1/2 rounded-1/4 leading-3/4 px-1/2 py-1/2 bg-primary-400 hover:bg-primary-350"},{default:m(()=>[i(y,{name:"external-link",class:"h-[1em] fill-current"}),ee]),_:1},8,["external"])):u("",!0),i(_,{to:o(h),class:"flex items-center gap-1/2 rounded-1/4 leading-3/4 px-3/4 py-1/2 bg-primary-500 hover:bg-primary-450"},{default:m(()=>[te]),_:1},8,["to"])])],2)],2)],2)}}});export{ie as _};