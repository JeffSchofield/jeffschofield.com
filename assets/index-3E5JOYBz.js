import{p as m,v as y,o as i,w as o,a as n,u,d as a,c as b,s as d,T as f,R as k,S as _,N as z,U as B,A as C,W as $,q as V,r as E,b as t,C as N}from"./index-CtIPOu-Y.js";const P=["href","onClick"],R={key:0,class:"absolute inset-0 flex items-center justify-center"},S={key:0,class:"absolute rounded-inherit inset-0 border-gradient-to-b from-neutral-650/80 to-neutral-600/70"},c=m({__name:"GhostButton",props:{to:{},href:{},is:{},color:{default:"primary"},size:{default:"base"},rounded:{type:Boolean,default:!1},disabled:{type:Boolean},loading:{type:Boolean}},setup(w){const v={primary:"bg-primary-500/50 border-primary-500 can-hover:hover:border-primary-450 can-hover:hover:bg-primary-500 can-hover:active:border-primary-350 active:border-primary-350 can-hover:active:bg-primary-350 active:bg-primary-350 can-hover:hover:text-white",secondary:"bg-secondary-550/50 border-secondary-550 can-hover:hover:border-secondary-500 can-hover:hover:bg-secondary-500 can-hover:active:border-secondary-350 active:border-secondary-350 can-hover:active:bg-secondary-350 active:bg-secondary-350 can-hover:hover:text-white",neutral:"bg-neutral-500/50 border-neutral-500 can-hover:hover:border-neutral-450 can-hover:hover:bg-neutral-500 can-hover:active:border-neutral-350 active:border-neutral-350 can-hover:active:bg-neutral-350 active:bg-neutral-350 can-hover:hover:text-white",info:"bg-info-500/50 border-info-500 can-hover:hover:border-info-450 can-hover:hover:bg-info-500 can-hover:active:border-info-350 active:border-info-350 can-hover:active:bg-info-350 active:bg-info-350 can-hover:hover:text-white",success:"bg-success-500/50 border-success-500 can-hover:hover:border-success-450 can-hover:hover:bg-success-500 can-hover:active:border-success-350 active:border-success-350 can-hover:active:bg-success-350 active:bg-success-350 can-hover:hover:text-white",warning:"bg-warning-500/50 border-warning-500 can-hover:hover:border-warning-450 can-hover:hover:bg-warning-500 can-hover:active:border-warning-350 active:border-warning-350 can-hover:active:bg-warning-350 active:bg-warning-350 can-hover:hover:text-white",alert:"bg-alert-500/50 border-alert-500 can-hover:hover:border-alert-450 can-hover:hover:bg-alert-500 can-hover:active:border-alert-350 active:border-alert-350 can-hover:active:bg-alert-350 active:bg-alert-350 can-hover:hover:text-white",link:"bg-link-500/50 border-link-500 can-hover:hover:border-link-450 can-hover:hover:bg-link-500 can-hover:active:border-link-350 active:border-link-350 can-hover:active:bg-link-350 active:bg-link-350 can-hover:hover:text-white",slate:"bg-slate-500/50 border-slate-500 can-hover:hover:border-slate-450 can-hover:hover:bg-slate-500 can-hover:active:border-slate-350 active:border-slate-350 can-hover:active:bg-slate-350 active:bg-slate-350 can-hover:hover:text-white",gray:"bg-gray-500/50 border-gray-500 can-hover:hover:border-gray-450 can-hover:hover:bg-gray-500 can-hover:active:border-gray-350 active:border-gray-350 can-hover:active:bg-gray-350 active:bg-gray-350 can-hover:hover:text-white","gray-neutral":"bg-gray-neutral-550/50 border-gray-neutral-550 can-hover:hover:border-gray-neutral-500 can-hover:hover:bg-gray-neutral-500 can-hover:active:border-gray-neutral-350 active:border-gray-neutral-350 can-hover:active:bg-gray-neutral-350 active:bg-gray-neutral-350 can-hover:hover:text-white",zinc:"bg-zinc-500/50 border-zinc-500 can-hover:hover:border-zinc-450 can-hover:hover:bg-zinc-500 can-hover:active:border-zinc-350 active:border-zinc-350 can-hover:active:bg-zinc-350 active:bg-zinc-350 can-hover:hover:text-white",stone:"bg-stone-500/50 border-stone-500 can-hover:hover:border-stone-450 can-hover:hover:bg-stone-500 can-hover:active:border-stone-350 active:border-stone-350 can-hover:active:bg-stone-350 active:bg-stone-350 can-hover:hover:text-white",red:"bg-red-500/50 border-red-500 can-hover:hover:border-red-450 can-hover:hover:bg-red-500 can-hover:active:border-red-350 active:border-red-350 can-hover:active:bg-red-350 active:bg-red-350 can-hover:hover:text-white",orange:"bg-orange-500/50 border-orange-500 can-hover:hover:border-orange-450 can-hover:hover:bg-orange-500 can-hover:active:border-orange-350 active:border-orange-350 can-hover:active:bg-orange-350 active:bg-orange-350 can-hover:hover:text-white",amber:"bg-amber-500/50 border-amber-500 can-hover:hover:border-amber-450 can-hover:hover:bg-amber-500 can-hover:active:border-amber-350 active:border-amber-350 can-hover:active:bg-amber-350 active:bg-amber-350 can-hover:hover:text-white",yellow:"bg-yellow-500/50 border-yellow-500 can-hover:hover:border-yellow-450 can-hover:hover:bg-yellow-500 can-hover:active:border-yellow-350 active:border-yellow-350 can-hover:active:bg-yellow-350 active:bg-yellow-350 can-hover:hover:text-white",lime:"bg-lime-500/50 border-lime-500 can-hover:hover:border-lime-450 can-hover:hover:bg-lime-500 can-hover:active:border-lime-350 active:border-lime-350 can-hover:active:bg-lime-350 active:bg-lime-350 can-hover:hover:text-white",green:"bg-green-500/50 border-green-500 can-hover:hover:border-green-450 can-hover:hover:bg-green-500 can-hover:active:border-green-350 active:border-green-350 can-hover:active:bg-green-350 active:bg-green-350 can-hover:hover:text-white",emerald:"bg-emerald-500/50 border-emerald-500 can-hover:hover:border-emerald-450 can-hover:hover:bg-emerald-500 can-hover:active:border-emerald-350 active:border-emerald-350 can-hover:active:bg-emerald-350 active:bg-emerald-350 can-hover:hover:text-white",teal:"bg-teal-500/50 border-teal-500 can-hover:hover:border-teal-450 can-hover:hover:bg-teal-500 can-hover:active:border-teal-350 active:border-teal-350 can-hover:active:bg-teal-350 active:bg-teal-350 can-hover:hover:text-white",cyan:"bg-cyan-500/50 border-cyan-500 can-hover:hover:border-cyan-450 can-hover:hover:bg-cyan-500 can-hover:active:border-cyan-350 active:border-cyan-350 can-hover:active:bg-cyan-350 active:bg-cyan-350 can-hover:hover:text-white",sky:"bg-sky-500/50 border-sky-500 can-hover:hover:border-sky-450 can-hover:hover:bg-sky-500 can-hover:active:border-sky-350 active:border-sky-350 can-hover:active:bg-sky-350 active:bg-sky-350 can-hover:hover:text-white",blue:"bg-blue-500/50 border-blue-500 can-hover:hover:border-blue-450 can-hover:hover:bg-blue-500 can-hover:active:border-blue-350 active:border-blue-350 can-hover:active:bg-blue-350 active:bg-blue-350 can-hover:hover:text-white",indigo:"bg-indigo-500/50 border-indigo-500 can-hover:hover:border-indigo-450 can-hover:hover:bg-indigo-500 can-hover:active:border-indigo-350 active:border-indigo-350 can-hover:active:bg-indigo-350 active:bg-indigo-350 can-hover:hover:text-white",violet:"bg-violet-500/50 border-violet-500 can-hover:hover:border-violet-450 can-hover:hover:bg-violet-500 can-hover:active:border-violet-350 active:border-violet-350 can-hover:active:bg-violet-350 active:bg-violet-350 can-hover:hover:text-white",purple:"bg-purple-500/50 border-purple-500 can-hover:hover:border-purple-450 can-hover:hover:bg-purple-500 can-hover:active:border-purple-350 active:border-purple-350 can-hover:active:bg-purple-350 active:bg-purple-350 can-hover:hover:text-white",fuchsia:"bg-fuchsia-500/50 border-fuchsia-500 can-hover:hover:border-fuchsia-450 can-hover:hover:bg-fuchsia-500 can-hover:active:border-fuchsia-350 active:border-fuchsia-350 can-hover:active:bg-fuchsia-350 active:bg-fuchsia-350 can-hover:hover:text-white",pink:"bg-pink-500/50 border-pink-500 can-hover:hover:border-pink-450 can-hover:hover:bg-pink-500 can-hover:active:border-pink-350 active:border-pink-350 can-hover:active:bg-pink-350 active:bg-pink-350 can-hover:hover:text-white",rose:"bg-rose-500/50 border-rose-500 can-hover:hover:border-rose-450 can-hover:hover:bg-rose-500 can-hover:active:border-rose-350 active:border-rose-350 can-hover:active:bg-rose-350 active:bg-rose-350 can-hover:hover:text-white"},p={primary:"!border-primary-500 !bg-primary-500",secondary:"!border-secondary-500 !bg-secondary-500",neutral:"!border-neutral-450 !bg-neutral-500",info:"!border-info-450 !bg-info-500",success:"!border-success-450 !bg-success-500",warning:"!border-warning-450 !bg-warning-500",alert:"!border-alert-450 !bg-alert-500",link:"!border-link-450 !bg-link-500",slate:"!border-slate-450 !bg-slate-500",gray:"!border-gray-450 !bg-gray-500","gray-neutral":"!border-gray-neutral-500 !bg-gray-neutral-500",zinc:"!border-zinc-450 !bg-zinc-500",stone:"!border-stone-450 !bg-stone-500",red:"!border-red-450 !bg-red-500",orange:"!border-orange-450 !bg-orange-500",amber:"!border-amber-450 !bg-amber-500",yellow:"!border-yellow-450 !bg-yellow-500",lime:"!border-lime-450 !bg-lime-500",green:"!border-green-450 !bg-green-500",emerald:"!border-emerald-450 !bg-emerald-500",teal:"!border-teal-450 !bg-teal-500",cyan:"!border-cyan-450 !bg-cyan-500",sky:"!border-sky-450 !bg-sky-500",blue:"!border-blue-450 !bg-blue-500",indigo:"!border-indigo-450 !bg-indigo-500",violet:"!border-violet-450 !bg-violet-500",purple:"!border-purple-450 !bg-purple-500",fuchsia:"!border-fuchsia-450 !bg-fuchsia-500",pink:"!border-pink-450 !bg-pink-500",rose:"!border-rose-450 !bg-rose-500"},l={xs:"initial:rounded-1/4 initial:text-sm initial:px-3/8 initial:py-1/4",sm:"initial:rounded-3/8 initial:text-sm initial:px-1/2 initial:py-1/2",base:"initial:rounded-1/2 initial:px-1 initial:py-3/4",lg:"initial:rounded-5/8 initial:text-lg initial:px-1-1/2 initial:py-1"},s={xs:"h-1/2",sm:"h-3/4",base:"h-1",lg:"h-1-1/2"};return(e,r)=>(i(),y(B,z({custom:""},{to:e.to,is:e.is,href:e.href,disabled:e.disabled}),{default:o(({isExactActive:h,href:g,navigate:x})=>[n("a",{class:u(["block relative cursor-pointer disabled:cursor-not-allowed border-[2px] transition-color duration-100 focus-visible:z-10 focus-visible:outline-none focus-visible:ring-1/8",{[e.$attrs.class]:!0,[v[e.color]]:!0,[l[e.size]]:!0,[p[e.color]]:h}]),href:g,onClick:x},[a(f,{"enter-to-class":"opacity-100"},{default:o(()=>[e.loading?(i(),b("div",R,[a(k,{name:"loader",class:u(["fill-current animate-[spin_500ms_linear_infinite]",{[s[e.size]]:!0}])},null,8,["class"])])):d("",!0)]),_:1}),e.disabled?(i(),b("div",S)):d("",!0),n("div",{class:u(["trim-both",{"opacity-0":e.loading}])},[_(e.$slots,"default")],2)],10,P)]),_:3},16))}}),T={class:"flex-1"},A={key:0,class:"p-1 prose prose-invert mx-auto w-full max-w-lg flex flex-col sm:h-10"},j={key:1,class:"fixed z-10 bottom-[calc(2*var(--twgl-current-base)-1px)] inset-x-0 md:relative md:bottom-0 md:mx-auto md:w-full md:max-w-lg mb-1 bg-gradient-to-t from-neutral-900/70 to-neutral-925/90 border-t border-neutral-800/40 backdrop-blur-[3px] sm:backdrop-blur-none sm:border-none md:bg-none"},M={class:"p-1/2 pb-3/4 md:p-1 md:py-0 bg-masked-x flex md:flex-wrap gap-1/2 overflow-x-auto"},q=m({__name:"index",setup(w){const v=C(),{onAfterPageEnter:p}=$(),l=V(()=>v.matched[v.matched.length-1].meta.project_frontmatter?v.meta.hasModal:!0);function s(){const e=document.getElementById("app");e==null||e.scrollTo({top:0,left:0,behavior:"smooth"})}return(e,r)=>{const h=E("RouterView");return i(),b("div",T,[l.value?(i(),b("div",A,r[0]||(r[0]=[n("div",{class:"hidden sm:block flex-1"},null,-1),n("div",null,[n("h1",{class:"inline mb-0 whitespace-nowrap pointer-events-auto"},[n("span",{class:"text-secondary-250"},"My"),t(" Portfolio")])],-1)]))):d("",!0),l.value?(i(),b("div",j,[n("div",M,[a(c,{class:"shrink-0 pointer-events-auto select-none px-3/4 py-3/4 md:py-1/4 min-w-3 text-center backdrop-blur-md",to:"/portfolio"},{default:o(()=>r[1]||(r[1]=[t("Highlights")])),_:1}),a(c,{class:"shrink-0 pointer-events-auto select-none px-3/4 py-3/4 md:py-1/4 min-w-3 text-center backdrop-blur-md",to:"/portfolio/simulations"},{default:o(()=>r[2]||(r[2]=[t("Simulations")])),_:1}),a(c,{class:"shrink-0 pointer-events-auto select-none px-3/4 py-3/4 md:py-1/4 min-w-3 text-center backdrop-blur-md",to:"/portfolio/software"},{default:o(()=>r[3]||(r[3]=[t("Software")])),_:1}),a(c,{class:"shrink-0 pointer-events-auto select-none px-3/4 py-3/4 md:py-1/4 min-w-3 text-center backdrop-blur-md",to:"/portfolio/web"},{default:o(()=>r[4]||(r[4]=[t("Web")])),_:1}),a(c,{class:"shrink-0 pointer-events-auto select-none px-3/4 py-3/4 md:py-1/4 min-w-3 text-center backdrop-blur-md",to:"/portfolio/video"},{default:o(()=>r[5]||(r[5]=[t("Video")])),_:1}),a(c,{class:"shrink-0 pointer-events-auto select-none px-3/4 py-3/4 md:py-1/4 min-w-3 text-center backdrop-blur-md",to:"/portfolio/all"},{default:o(()=>r[6]||(r[6]=[t("All")])),_:1})])])):d("",!0),a(h,null,{default:o(({Component:g})=>[a(f,{mode:"out-in","enter-active-class":"duration-100 ease-out","enter-from-class":"transform translate-y-1 opacity-0","enter-to-class":"opacity-100","leave-active-class":"duration-50 ease-in","leave-from-class":"opacity-100","leave-to-class":"transform translate-y-1/2 opacity-0",onBeforeEnter:s},{default:o(()=>[(i(),y(N(g)))]),_:2},1024)]),_:1})])}}});export{q as default};
