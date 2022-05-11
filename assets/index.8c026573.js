import{_ as V,u as B}from"./Masthead.0e13d929.js";import{a as S,b as C,c as M,e as R,w as d,s as L,B as F,l as H,g as r,v as e,H as J,f as O,C as t}from"./vendor.0830267b.js";import{_ as P}from"./Fireflies.4b2b1ada.js";import{r as N,t as f,d as U,g as y,c as $}from"./index.fb1087b0.js";import"./SvgLogo.9ec7e2d7.js";import"./index.6f690fd2.js";import"./utils.ad40f7b8.js";function W(o,a){N(2,arguments);var s=f(o),n=f(a),h=s.getTime()-n.getTime();return h<0?-1:h>0?1:h}function Y(o,a){if(o==null)throw new TypeError("assign requires that input parameter not be null or undefined");a=a||{};for(var s in a)Object.prototype.hasOwnProperty.call(a,s)&&(o[s]=a[s]);return o}function G(o){return Y({},o)}var k=1e3*60,x=60*24,I=x*30,D=x*365;function Q(o,a){var s=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{};N(2,arguments);var n=s.locale||U;if(!n.formatDistance)throw new RangeError("locale must contain localize.formatDistance property");var h=W(o,a);if(isNaN(h))throw new RangeError("Invalid time value");var l=G(s);l.addSuffix=Boolean(s.addSuffix),l.comparison=h;var m,g;h>0?(m=f(a),g=f(o)):(m=f(o),g=f(a));var v=s.roundingMethod==null?"round":String(s.roundingMethod),c;if(v==="floor")c=Math.floor;else if(v==="ceil")c=Math.ceil;else if(v==="round")c=Math.round;else throw new RangeError("roundingMethod must be 'floor', 'ceil' or 'round'");var b=g.getTime()-m.getTime(),u=b/k,A=y(g)-y(m),p=(b-A)/k,i;if(s.unit==null?u<1?i="second":u<60?i="minute":u<x?i="hour":p<I?i="day":p<D?i="month":i="year":i=String(s.unit),i==="second"){var T=c(b/1e3);return n.formatDistance("xSeconds",T,l)}else if(i==="minute"){var E=c(u);return n.formatDistance("xMinutes",E,l)}else if(i==="hour"){var q=c(u/60);return n.formatDistance("xHours",q,l)}else if(i==="day"){var j=c(p/x);return n.formatDistance("xDays",j,l)}else if(i==="month"){var w=c(p/I);return w===12&&s.unit!=="month"?n.formatDistance("xYears",1,l):n.formatDistance("xMonths",w,l)}else if(i==="year"){var z=c(p/D);return n.formatDistance("xYears",z,l)}throw new RangeError("unit must be 'second', 'minute', 'hour', 'day', 'month' or 'year'")}var K=$(Q,2),X="/assets/jeff-schofield.8d973196.jpg",Z="/assets/vue-logo.2cbec2c1.svg",ee="/assets/tailwind-logo.da161fa0.svg",te="/assets/vite-logo.8ce09b94.svg",se="/assets/graphql-logo.ad6a04a1.svg",oe="/assets/nestjs-logo.48bd018d.svg",ae="/assets/docker-logo.6b41650e.svg";const _=S({props:{to:null,href:null},setup(o){return(a,s)=>{const n=C("AbstractButton");return M(),R(n,{to:o.to,external:o.href,class:"underline font-bold transition-color duration-50 text-blue-100 hover:text-white"},{default:d(()=>[L(a.$slots,"default")]),_:3},8,["to","external"])}}}),ne={class:"bg-neutral-200 dark:bg-neutral-925"},ie={class:""},le={class:"grid grid-stack bg-gradient-to-b from-primary-1000 to-neutral-950 dark:to-neutral-925 rounded-b-1/2"},re=e("div",{class:"z-[20] flex flex-col h-[55vh] md:h-1/2-screen md:min-h-25 pointer-events-none"},[e("div",{class:"flex-1"}),e("div",{class:"h-8 bg-gradient-to-b from-transparent to-neutral-950"})],-1),ce={class:"z-[20] px-1 py-1/2 space-y-2 pointer-events-none"},de=e("div",{class:"h-[55vh] md:h-1/2-screen md:min-h-25 flex flex-col -mb-4"},[e("div",{class:"flex-1"}),e("h1",{class:"text-5xl leading-2-1/2 lg:text-6xl lg:leading-3-1/4 font-black pointer-events-auto pb-4 text-white"},[e("span",{class:"text-primary-200"},"Full Stack"),t(),e("br"),t("Web Design &"),e("br"),t(" Development")])],-1),he={class:""},_e=e("img",{src:X,alt:"Photo of Jeff",class:"w-[50%] xs:max-w-17 md:w-[60%] md:max-w-20 -mr-1 -mt-1 lg:-mt-3 pl-1 float-right",style:{"clip-path":"circle(54% at 73% 50%)","shape-outside":"circle(59% at 70% 45%)"}},null,-1),fe={class:"text-xl leading-1-1/2 text-neutral-100 xs:text-justify text-last-left mb-4"},ue=t("Hello! My name is "),pe=e("b",null,"Jeff Schofield",-1),me=t(". I am a "),ge=e("b",null,"soft\xADware developer",-1),xe=t(" from "),ve=e("b",null,"Peter\xADborough, Ontario",-1),be=t(" who special\xADizes in application develop\xADment for the web. I have "),we=t(" designing and con\xADstructing websites, web applications, as well as the server software and infra\xADstructure that powers it all."),ye={class:"px-1 pb-2 space-y-2 !-mt-[10%] md:!-mt-[12%]"},ke=e("h2",{class:"text-5xl font-black"},[t("Front, back, and "),e("br"),t("everything in between.")],-1),Ie=e("p",{class:"text-xl text-neutral-250"},"I like working on unique projects. I am comfortable designing simple static websites, developing prototypes for more dynamic web applications, and working on existing codebases to add new features or update existing ones.",-1),De={class:"text-xl text-neutral-250"},Se=t(" Check out "),Me=t("my work"),Ne=t(" and "),Ae=t("email me"),Te=t(" if you're looking for a web developer. "),Ee={class:"flex py-2 bg-gradient-to-b from-primary-600/30 to-transparent"},qe=e("div",{class:"hidden lg:block flex-1"},null,-1),je={class:"lg:w-70% px-1 space-y-2"},ze=e("div",{class:"space-y-1"},[e("h2",{class:"text-4xl font-bold text-primary-100"},"Beautiful and functional interfaces"),e("p",{class:"text-lg text-neutral-150"},"I use modern reactive frameworks and systems of design to create responsive and accessible application interfaces. I also love working with WebGL to create next-level visualizations that leave a lasting impression.")],-1),Ve={class:"space-y-2"},Be={class:"flex gap-1 lg:pl-1"},Ce=e("div",null,[e("img",{src:Z,class:"h-2-1/2 aspect-square"})],-1),Re={class:"flex-1 space-y-1"},Le=e("h3",{class:"text-3xl font-bold"},"Experienced with Vue",-1),Fe={class:"text-neutral-200"},He=t("I keep coming back to "),Je=t("Vue 3"),Oe=t(" as the foundation for web sites and applications of all shapes and sizes. Vue has a mature ecosystem with established patterns that allow it to fit almost any niche."),Pe={class:"flex gap-1 lg:pl-1"},Ue=e("div",null,[e("img",{src:ee,class:"h-2-1/2 aspect-square"})],-1),$e={class:"flex-1 space-y-1"},We=e("h3",{class:"text-3xl font-bold"},"Design system champion",-1),Ye={class:"text-neutral-200"},Ge=t(" I have suffered the pain of updating branding colors within a loose collection of stylesheets. Never again. I champion design systems based on utility CSS classes. I've returned to "),Qe=t("Tailwind"),Ke=t(" projects after years and made updates effortlessly. "),Xe={class:"flex gap-1 lg:pl-1"},Ze=e("div",null,[e("img",{src:te,class:"h-2-1/2 aspect-square"})],-1),et={class:"flex-1 space-y-1"},tt=e("h3",{class:"text-3xl font-bold"},"Rapid UI prototyping",-1),st={class:"text-neutral-200"},ot=t("Vite"),at=t(" is an exceptional multi-tool that is truly the work horse of my front end projects. With first class support for Vue and Tailwind, Vite lets me scaffold a project quickly "),nt=e("b",null,"and",-1),it=t(" get instant updates on save so I can work with clients in real time!"),lt={class:"flex py-2 bg-gradient-to-b from-primary-800/30 to-transparent"},rt={class:"lg:w-70% px-1 space-y-2"},ct=e("div",{class:"space-y-1"},[e("h2",{class:"text-4xl font-bold text-secondary-250"},"Versatile server software architecture"),e("p",{class:"text-lg text-neutral-250"})],-1),dt={class:"space-y-2"},ht={class:"flex gap-1 lg:pl-1"},_t=e("div",null,[e("img",{src:se,class:"h-2-1/2 aspect-square"})],-1),ft={class:"flex-1 space-y-1"},ut=e("h3",{class:"text-3xl font-bold"},"Interfacing with graphs",-1),pt={class:"text-neutral-200"},mt=t("I prefer to both produce and consume "),gt=t("GraphQL"),xt=t(" APIs. Graphs are a natural way to represent complex data models, and federation allows me to keep services physically separate while still contributing to the same interface!"),vt={class:"flex gap-1 lg:pl-1"},bt=e("div",null,[e("img",{src:oe,class:"h-2-1/2 aspect-square"})],-1),wt={class:"flex-1 space-y-1"},yt=e("h3",{class:"text-3xl font-bold"},"Bag of software architecture",-1),kt={class:"text-neutral-200"},It=t("For Node servers, "),Dt=t("NestJS"),St=t(" is an incredible backend framework that offers so many powerful architectural patterns right out of the box. I use the principles of domain driven design to build up modules for a given bounded context."),Mt={class:"flex gap-1 lg:pl-1"},Nt=e("div",null,[e("img",{src:ae,class:"h-2-1/2 aspect-square"})],-1),At={class:"flex-1 space-y-1"},Tt=e("h3",{class:"text-3xl font-bold"},[t("Contain "),e("i",null,"everything")],-1),Et={class:"text-neutral-200"},qt=t("Making versioned "),jt=t("Docker"),zt=t(" container images of the server software I write has drastically improved reliability and predictability in production. No more risky unicorn servers that could take down a whole operation if they break!"),Vt=e("div",{class:"hidden lg:block flex-1"},null,-1),Ot=S({setup(o){F({title:"Web Design and Development | Jeff Schofield",meta:[{name:"description",content:"The portfolio and resume of full stack web designer and developer Jeff Schofield."}]});const{title:a,start_date:s}=B(),n=K(s,Date.now());return(h,l)=>(M(),H("div",null,[r(V,{class:"flex md:hidden"}),e("div",ne,[e("section",ie,[e("div",le,[r(P,{class:"h-[55vh] md:h-1/2-screen md:min-h-25"}),re,e("div",ce,[de,e("div",he,[_e,e("p",fe,[ue,pe,me,ge,xe,ve,be,e("b",null,J(O(n))+" of pro\xADfessional exp\xADerience",1),we])])])])]),e("section",ye,[ke,Ie,e("p",De,[Se,r(_,{to:"/portfolio"},{default:d(()=>[Me]),_:1}),Ne,r(_,{href:"mailto:contact@jeffschofield.com"},{default:d(()=>[Ae]),_:1}),Te])]),e("section",Ee,[qe,e("div",je,[ze,e("ul",Ve,[e("li",Be,[Ce,e("div",Re,[Le,e("p",Fe,[He,r(_,{external:"https://vuejs.org"},{default:d(()=>[Je]),_:1}),Oe])])]),e("li",Pe,[Ue,e("div",$e,[We,e("p",Ye,[Ge,r(_,{external:"https://tailwindcss.com/"},{default:d(()=>[Qe]),_:1}),Ke])])]),e("li",Xe,[Ze,e("div",et,[tt,e("p",st,[r(_,{external:"https://vitejs.dev/"},{default:d(()=>[ot]),_:1}),at,nt,it])])])])])]),e("section",lt,[e("div",rt,[ct,e("ul",dt,[e("li",ht,[_t,e("div",ft,[ut,e("p",pt,[mt,r(_,{external:"https://graphql.org/"},{default:d(()=>[gt]),_:1}),xt])])]),e("li",vt,[bt,e("div",wt,[yt,e("p",kt,[It,r(_,{external:"https://nestjs.com/"},{default:d(()=>[Dt]),_:1}),St])])]),e("li",Mt,[Nt,e("div",At,[Tt,e("p",Et,[qt,r(_,{external:"https://docker.com/"},{default:d(()=>[jt]),_:1}),zt])])])])]),Vt])])]))}});export{Ot as default};
