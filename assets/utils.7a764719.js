var F=Object.defineProperty;var y=(e,t,r)=>t in e?F(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r;var f=(e,t,r)=>(y(e,typeof t!="symbol"?t+"":t,r),r);import{O as S}from"./vendor.169cfbe7.js";var B=Object.defineProperty,M=Object.defineProperties,D=Object.getOwnPropertyDescriptors,w=Object.getOwnPropertySymbols,L=Object.prototype.hasOwnProperty,O=Object.prototype.propertyIsEnumerable,T=(e,t,r)=>t in e?B(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r,x=(e,t)=>{for(var r in t||(t={}))L.call(t,r)&&T(e,r,t[r]);if(w)for(var r of w(t))O.call(t,r)&&T(e,r,t[r]);return e},A=(e,t)=>M(e,D(t));function I(e,t,r){let a=e.createProgram();if(!a)throw new Error("Unable to create program");if(e.attachShader(a,t),e.attachShader(a,r),e.linkProgram(a),!e.getProgramParameter(a,e.LINK_STATUS))throw new Error("Unable to create program");return a}function R(e,t,r){let a=e.createShader(t);if(!a)throw new Error("Unable to create shader");if(e.shaderSource(a,r),e.compileShader(a),!e.getShaderParameter(a,e.COMPILE_STATUS))throw console.log(r),console.error(e.getShaderInfoLog(a)),new Error("Unable to compile shader");return a}function b(e,t,r,a){const i=e.getUniformLocation(t,a);return function(...s){if(!!i)return e.useProgram(t),e[`uniform${r}`](i,...s)}}function X(e,t,r){return e.getAttribLocation(t,r)}var c=new Map;function N(e,t){let r=c.get(e)||[];c.set(e,[...r,t])}function $(e){for(let t of c.get(e)||[])e.deleteBuffer(t);c.set(e,[])}function k(e,t){let r=c.get(e)||[],a=r.indexOf(t);a!=-1&&(e.deleteBuffer(t),r.splice(a,1),c.set(e,r))}function z(e,t,r=e.STATIC_DRAW){const a=e.createBuffer();if(!a)throw new Error("Unable to create buffer");return e.bindBuffer(e.ARRAY_BUFFER,a),e.bufferData(e.ARRAY_BUFFER,t,r),N(e,a),a}function H(e,t,r,a,i){e.bindBuffer(e.ARRAY_BUFFER,t),e.drawArrays(r,a,i)}var p=new Map;function C(e,t){let r=p.get(e)||[];p.set(e,[...r,t])}function K(e){for(let t of p.get(e)||[])e.deleteFramebuffer(t);p.set(e,[])}function J(e){let t=e.createFramebuffer();if(!t)throw new Error("Unable to create framebuffer");return C(e,t),t}function Q(e,t,r,a){e.bindFramebuffer(e.FRAMEBUFFER,t),e.framebufferTexture2D(e.FRAMEBUFFER,r,e.TEXTURE_2D,a,0)}var l=new Map;function j(e,t){let r=l.get(e)||new Set;r.add(t),l.set(e,r)}function Z(e){for(let t of l.get(e)||new Set)e.deleteTexture(t);l.set(e,new Set)}function g(e,t,r,a,i,o,s=null){let n=e.createTexture();if(!n)throw new Error("Unable to create texture");return e.bindTexture(e.TEXTURE_2D,n),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_S,e.REPEAT),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_WRAP_T,e.REPEAT),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MIN_FILTER,e.NEAREST),e.texParameteri(e.TEXTURE_2D,e.TEXTURE_MAG_FILTER,e.NEAREST),e.texImage2D(e.TEXTURE_2D,0,a,t,r,0,i,o,s),j(e,n),n}function ee(e,t,r){e.activeTexture(t),e.bindTexture(e.TEXTURE_2D,r)}var q=(e,...t)=>e.map((r,a)=>`${r}${t[a]||""}`).join(""),G=q`#version 300 es

precision highp float;

void main() {
	float x = float((gl_VertexID & 1) << 2);
	float y = float((gl_VertexID & 2) << 1);
	gl_Position = vec4(x - 1.0, y - 1.0, 0, 1);
}
`,W=()=>({devicePixelRatio:window&&window.devicePixelRatio||1,context:{premultipliedAlpha:!1}});function P(...e){const t={};for(let r of e)for(let a in r){let i=r[a];typeof i=="object"&&i!==null?(t[a]=t[a]||{},P(t[a],r[a])):t[a]=i}return t}function te(e,t,r,a={}){const{devicePixelRatio:i,context:o}=P(W(),a),s=e.getContext("webgl2",o);if(!s)throw new Error("Unable to get WebGL context.");const n=v(s,t,r),{setResolution:u}=n;function _(d=!0){if(!s)return{};s.bindFramebuffer(s.FRAMEBUFFER,null),s.clearColor(0,0,0,0),s.clear(s.COLOR_BUFFER_BIT);const E=e.getBoundingClientRect(),m=e.width=E.width*i,h=e.height=E.height*i;return s.viewport(0,0,m,h),u([m,h]),d&&s.drawArrays(s.TRIANGLE_FAN,0,3),{width:m,height:h,ratio:m/h}}function U(){if(!s)return;let d=s.getExtension("WEBGL_lose_context");d&&d.loseContext()}return A(x({},n),{render:_,destroy:U})}function v(e,t,r){const a=R(e,e.VERTEX_SHADER,t),i=R(e,e.FRAGMENT_SHADER,r),o=I(e,a,i);if(!o)throw new Error("Unable to create shader program");const s=b(e,o,"2fv","resolution");return e.useProgram(o),{gl:e,program:o,vertex_shader:a,fragment_shader:i,createUniform:(n,u)=>b(e,o,n,u),createAttribute:n=>X(e,o,n),setResolution:s}}function re(e,t){return A(x({},v(e,G,t)),{draw(){e.drawArrays(e.TRIANGLE_FAN,0,3)}})}var ae=class{constructor(){f(this,"program_states",new Map);f(this,"program_inits",new Map);f(this,"program_updates",new Map);f(this,"program_renders",new Map);f(this,"program_destroys",new Map);f(this,"state",0);f(this,"last_loop_time",0);f(this,"update_speed",1);f(this,"update_time_step",16);f(this,"max_updates",200);f(this,"animation_frame");f(this,"accumulated_frame_time",0)}linkProgram(e){if(this.program_states.set(e,0),e.init){const t=async()=>{let r=e.init();(!r||!r.then)&&(r=Promise.resolve(r)),await r,this.program_states.set(e,1),e.update&&this.program_updates.set(e,e.update),e.render&&this.program_renders.set(e,e.render)};this.program_inits.set(e,t),[1,2].includes(this.state)&&t()}e.destroy&&this.program_destroys.set(e,()=>{this.program_states.get(e)==1&&(this.program_updates.delete(e),this.program_renders.delete(e),e.destroy(),this.program_states.set(e,2))})}unlinkProgram(e){this.program_states.delete(e),this.program_inits.delete(e),this.program_updates.delete(e),this.program_renders.delete(e),this.program_destroys.delete(e)}async init(){if([0,3].includes(this.state)){this.last_loop_time=performance.now(),this.state=1;try{await Promise.all(Array.from(this.program_inits.values()).map(e=>e()))}catch(e){throw console.log(e),new Error("Error initializing")}this.requestLoop()}}requestLoop(){this.animation_frame=requestAnimationFrame(this.loop.bind(this))}cancelLoop(){this.animation_frame&&cancelAnimationFrame(this.animation_frame)}loop(e){if([0,3].includes(this.state))return;if(this.state==1)for(this.accumulated_frame_time+=e-this.last_loop_time,Math.floor(this.accumulated_frame_time/this.update_time_step)>this.max_updates&&(this.accumulated_frame_time=this.update_time_step);this.accumulated_frame_time>=this.update_time_step;)this.update(this.update_time_step),this.accumulated_frame_time-=this.update_time_step;let t=this.accumulated_frame_time/this.update_time_step;this.render(t),this.last_loop_time=e,this.requestLoop()}update(e){this.program_updates.forEach(t=>t(e*this.update_speed))}render(e){this.program_renders.forEach(t=>t(e))}};function se(e,t,r,a,i,o){const s=Array.from({length:16}),n=1/(e-t),u=1/(r-a),_=1/(i-o);return s[0]=-2*n,s[1]=0,s[2]=0,s[3]=0,s[4]=0,s[5]=-2*u,s[6]=0,s[7]=0,s[8]=0,s[9]=0,s[10]=2*_,s[11]=0,s[12]=(e+t)*n,s[13]=(a+r)*u,s[14]=(o+i)*_,s[15]=1,s}function ie(e){let t=Math.floor(Math.sqrt(e));for(let r of Array.from({length:t},(a,i)=>t-i))if(e%r==0)return[r,Math.floor(e/r)]}function oe(e,t){let r=e.value;e.value=t.value,t.value=r}function ne(e,t,{flush:r="sync",deep:a=!1,immediate:i=!0}={}){return S(e,o=>{t(o)},{flush:r,deep:a,immediate:i})}export{ae as L,re as a,ee as b,z as c,H as d,ie as e,Q as f,q as g,v as h,Z as i,K as j,g as k,J as l,oe as m,k as n,se as o,te as p,$ as r,ne as s};
