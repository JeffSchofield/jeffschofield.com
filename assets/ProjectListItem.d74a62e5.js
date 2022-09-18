import{e as V,g as l,P as t,r as _,o as r,c,b as i,w as m,i as p,n as a,h as s,j as f,a as o,d,t as u,F as $,l as L}from"./index.6f3d243b.js";import{_ as W,a as U}from"./ProjectImage.e38fc971.js";import{f as z}from"./index.61f96e6d.js";const H={key:0},G={key:0,class:"text-sm leading-1/2 text-neutral-400"},K=d("Released "),Y={class:"font-semibold text-neutral-200"},q={key:1,class:"text-sm leading-1/2 text-neutral-400"},J=d("Started "),Q={class:"font-semibold text-neutral-200"},X={class:"flex-1 flex flex-col gap-1/2 p-3/4 bg-neutral-925"},Z={class:"leading-1-1/4"},ee={class:"flex flex-wrap gap-1/2 h-1 overflow-hidden"},te={class:"rounded-1/4 px-1/4 py-1/4 text-xs leading-1/2 whitespace-nowrap text-primary-100/60 bg-primary-600/50"},se=o("div",{class:"flex-1"},null,-1),oe={class:"p-1/2 flex gap-1/2"},ne=d(" Repo"),re=o("span",{class:"sr-only"},"sitory",-1),le=d(" Visit"),ae=d("Read More"),_e=V({__name:"ProjectListItem",props:{project:null,card:{type:Boolean,default:!1},video:{type:Boolean,default:!1}},setup(e){const n=e,g=l(()=>`/portfolio/${n.project.path}${n.project.slug&&n.project.slug!="index"?`/${n.project.slug}`:""}`),x=z("MMMM yyyy"),v="flex sm:grid md:flex lg:grid flex-col grid-cols-3 0-sm:divide-y sm-md:divide-x md-lg:divide-y lg:divide-x",y="0-sm:rounded-t-inherit sm-md:rounded-l-inherit md-lg:rounded-t-inherit lg:rounded-l-inherit",E="0-sm:rounded-b-inherit sm-md:rounded-r-inherit md-lg:rounded-b-inherit lg:rounded-r-inherit",A="sm-md:rounded-tr-inherit lg:rounded-tr-inherit",S="0-sm:rounded-b-inherit sm-md:rounded-br-inherit md-lg:rounded-b-inherit lg:rounded-br-inherit",R="flex flex-col",k="rounded-t-inherit",B="rounded-b-inherit",C="rounded-b-inherit",I=l(()=>n.card?R:v),b=l(()=>n.card?k:y),w=l(()=>n.card?B:E),N=l(()=>n.card?"":A),O=l(()=>n.card?C:S),D=l(()=>n.card?"line-clamp-2":"line-clamp-3 sm:line-clamp-2 md:line-clamp-3 lg:line-clamp-2"),M={[t.APP]:"App",[t.ARTWORK]:"Artwork",[t.BACKEND]:"Backend",[t.DEVOPS]:"DevOps",[t.FRONTEND]:"Frontend",[t.GAME]:"Game",[t.MUSIC]:"Music",[t.OPEN_SOURCE]:"Open Source",[t.SHADER]:"Shader",[t.SIMULATION]:"Simulation",[t.SOFTWARE]:"Software",[t.HARDWARE]:"Hardware",[t.ELECTRONICS]:"Electronics",[t.EMBEDDED]:"Embedded",[t.USER_INTERFACE]:"User Interface",[t.VIDEO]:"Video",[t.WEBSITE]:"Website",[t.WEB_APPLICATION]:"Web Application"};return(ce,ie)=>{const P=_("RouterLink"),T=_("AbstractButton"),j=_("SvgIcon"),h=_("FlatButton");return r(),c("div",{class:a([s(I),"rounded-1/2 border border-neutral-950 divide-neutral-950"])},[i(P,{to:s(g),class:a(["col-span-1 flex items-center justify-center",s(b)])},{default:m(()=>[e.video&&e.project.video?(r(),c("div",H,[i(W,{src:e.project.video,class:"rounded-t-inherit w-full aspect-video"},null,8,["src"])])):e.project.image?(r(),p(U,{key:1,class:a(["w-full h-full aspect-[4/3] flex",s(b)]),image:e.project.image,alt:`Preview image of ${e.project.name}`,"img-class":"object-cover","img-style":`object-position: ${e.project.image_focus[0]}% ${e.project.image_focus[1]}%;`},null,8,["image","class","alt","img-style"])):f("",!0)]),_:1},8,["to","class"]),o("div",{class:a(["col-span-2 flex flex-col",s(w)])},[o("div",{class:a(["p-3/4 flex flex-col gap-1/4 bg-neutral-875",s(N)])},[i(T,{to:s(g),class:"font-semibold text-xl leading-3/4"},{default:m(()=>[d(u(e.project.name),1)]),_:1},8,["to"]),e.project.release_date?(r(),c("div",G,[K,o("span",Y,u(s(x)(e.project.release_date)),1)])):e.project.start_date?(r(),c("div",q,[J,o("span",Q,u(s(x)(e.project.start_date)),1)])):f("",!0)],2),o("div",X,[o("div",Z,[o("span",{class:a(s(D))},u(e.project.short_description),3)]),o("div",ee,[(r(!0),c($,null,L(e.project.categories,F=>(r(),c("div",te,u(M[F]),1))),256))])]),o("div",{class:a(["flex bg-neutral-875",s(O)])},[se,o("div",oe,[e.project.repo?(r(),p(h,{key:0,href:e.project.repo,color:"neutral",size:"xs",class:"flex items-center gap-1/4"},{default:m(()=>[i(j,{name:"github",class:"h-[1em] fill-current"}),ne,re]),_:1},8,["href"])):f("",!0),e.project.url?(r(),p(h,{key:1,href:e.project.url,color:"sky",size:"xs",class:"flex items-center gap-1/4"},{default:m(()=>[i(j,{name:"external-link",class:"h-[1em] fill-current"}),le]),_:1},8,["href"])):f("",!0),i(h,{to:s(g),size:"xs",class:"flex items-center gap-1/4"},{default:m(()=>[ae]),_:1},8,["to"])])],2)],2)],2)}}});export{_e as _};
