import{_ as J}from"./SvgLogo.f85a793d.js";import{u as U}from"./about.42cb4485.js";import{e as Y,u as $,o as j,c as q,a as s,b as u,t as P,h as V,w as h,d as t,r as B}from"./index.7fdbdf08.js";import{r as A,t as _,d as F,a as M,g as G,c as W}from"./index.3746861a.js";function K(n,i){A(2,arguments);var e=_(n),d=_(i),r=e.getTime()-d.getTime();return r<0?-1:r>0?1:r}function E(n,i){if(n==null)throw new TypeError("assign requires that input parameter not be null or undefined");for(var e in i)Object.prototype.hasOwnProperty.call(i,e)&&(n[e]=i[e]);return n}function Q(n){return E({},n)}var k=1e3*60,w=60*24,N=w*30,T=w*365;function X(n,i,e){var d,r,x;A(2,arguments);var l=G(),c=(d=(r=e==null?void 0:e.locale)!==null&&r!==void 0?r:l.locale)!==null&&d!==void 0?d:F;if(!c.formatDistance)throw new RangeError("locale must contain localize.formatDistance property");var b=K(n,i);if(isNaN(b))throw new RangeError("Invalid time value");var f=E(Q(e),{addSuffix:Boolean(e==null?void 0:e.addSuffix),comparison:b}),p,g;b>0?(p=_(i),g=_(n)):(p=_(n),g=_(i));var y=String((x=e==null?void 0:e.roundingMethod)!==null&&x!==void 0?x:"round"),a;if(y==="floor")a=Math.floor;else if(y==="ceil")a=Math.ceil;else if(y==="round")a=Math.round;else throw new RangeError("roundingMethod must be 'floor', 'ceil' or 'round'");var D=g.getTime()-p.getTime(),m=D/k,O=M(g)-M(p),v=(D-O)/k,S=e==null?void 0:e.unit,o;if(S?o=String(S):m<1?o="second":m<60?o="minute":m<w?o="hour":v<N?o="day":v<T?o="month":o="year",o==="second"){var L=a(D/1e3);return c.formatDistance("xSeconds",L,f)}else if(o==="minute"){var R=a(m);return c.formatDistance("xMinutes",R,f)}else if(o==="hour"){var z=a(m/60);return c.formatDistance("xHours",z,f)}else if(o==="day"){var H=a(v/w);return c.formatDistance("xDays",H,f)}else if(o==="month"){var I=a(v/N);return I===12&&S!=="month"?c.formatDistance("xYears",1,f):c.formatDistance("xMonths",I,f)}else if(o==="year"){var C=a(v/T);return c.formatDistance("xYears",C,f)}throw new RangeError("unit must be 'second', 'minute', 'hour', 'day', 'month' or 'year'")}var Z=W(X,2);const ee=s("section",{class:"mx-auto max-w-44 px-1 pb-2 pointer-events-none"},[s("div",{class:"h-[45vh] md:min-h-5 flex flex-col -mb-4"},[s("div",{class:"flex-1"}),s("h1",{class:"text-5xl leading-2-1/2 lg:text-6xl lg:leading-3-1/4 font-black pointer-events-auto pb-4 text-white sr-only"},"About Jeff Schofield")])],-1),te={class:"mx-auto max-w-44 px-1 pb-2 space-y-2 pointer-events-auto"},oe={class:"text-xl leading-1-1/2 text-neutral-100 text-last-left"},se=t("Hello! Welcome to the personal website of "),ne=s("b",{class:"text-secondary-250"},"Jeff Schofield",-1),ie=t(". I am a "),ae=s("b",{class:"text-secondary-250"},"soft\xADware designer and developer",-1),re=t(" from "),le=s("b",{class:"text-secondary-250"},"Peterborough, Ontario",-1),ce=t(" who special\xADizes in application develop\xADment for the web. I have "),de={class:"text-secondary-250"},fe=t(" designing and con\xADstructing websites, web applications, as well as the server software and system infra\xADstructure that powers it all."),ue={class:"text-xl text-neutral-100"},he=t("I am currently working as part of the innovation team with the talented people at "),_e=t("DesignStripe"),me=t(", where we're building design tools for the world and exploring the possibilities of AI-assisted design in the browser."),ve={class:"text-xl text-neutral-100"},xe=t("I also publish my open source work as "),pe=t("ShiftLimits on GitHub"),ge=t(", where I have a suite of frontend design tooling that I use across my own work available for anyone to use."),we={class:"text-xl text-neutral-100"},be=t("I like working on unique projects. Check out "),ye=t("my work"),De=t(", my personal "),Se=t("GitHub"),Ie=t(", and "),Me=t("email me"),ke=t(" or get in touch through my "),Ne=t("LinkedIn"),Te=t(" if you would like to work together."),Re=Y({__name:"index",setup(n){$({title:"Web Design and Development | Jeff Schofield",meta:[{name:"description",content:"The portfolio and resume of full stack web designer and developer Jeff Schofield."}]});const{title:i,start_date:e}=U(),d=Z(e,Date.now());return(r,x)=>{const l=B("TextLink");return j(),q("div",null,[ee,s("section",te,[u(J,{class:"sm:hidden h-3"}),s("p",oe,[se,ne,ie,ae,re,le,ce,s("b",de,P(V(d))+" of professional experience",1),fe]),s("p",ue,[he,u(l,{href:"//designstripe.com"},{default:h(()=>[_e]),_:1}),me]),s("p",ve,[xe,u(l,{href:"//github.com/ShiftLimits"},{default:h(()=>[pe]),_:1}),ge]),s("p",we,[be,u(l,{to:"/portfolio"},{default:h(()=>[ye]),_:1}),De,u(l,{href:"//github.com/JeffSchofield"},{default:h(()=>[Se]),_:1}),Ie,u(l,{href:"mailto:contact@jeffschofield.com"},{default:h(()=>[Me]),_:1}),ke,u(l,{href:"//linkedin.com/in/jeff-schofield-76555b163/"},{default:h(()=>[Ne]),_:1}),Te])])])}}});export{Re as default};
