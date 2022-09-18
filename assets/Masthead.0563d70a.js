import{e as $,f as F,o as C,i as M,w,a,G as K,R as H,n as Y,C as G,I as J,S as Q,v as W,x as X,B as Z,A as T,U as ee,c as I,h as te,j as oe,b as f,d as L,r as O}from"./index.842c1b90.js";import{_ as se}from"./SvgLogo.0a15b55c.js";import{g as U,n as re,a as B,o as ae,p as N}from"./utils.46b868cd.js";import"./about.bca9c54f.js";const ne=["href","onClick","data-active"],E=$({__name:"MastheadNavButton",setup(n){const{afterEach:e}=Q(),t=F(!1);function o(s){t.value=!0,s()}return e(()=>t.value=!1),(s,l)=>(C(),M(J,G(s.$attrs,{custom:""}),{default:w(({href:_,navigate:m,isActive:i})=>[a("a",{href:_,onClick:H(h=>o(m),["prevent"]),class:Y(["transition-all duration-35 ease-in-out flex-1 relative before:z-behind before:absolute before:inset-0 before:rounded-1/2 h-3-1/2 md:h-3 lg:h-2-1/2 xs:px-1/2 xs:min-w-4 flex items-center justify-center text-xs xs:text-sm sm:text-base text-neutral-50 text-shadow font-bold bg-gradient-to-br from-neutral-750/0 to-neutral-800/20 can-hover:hover:bg-neutral-0/10 border border-neutral-875/70 border-b-neutral-850/40 border-t-neutral-850/70 no-select",{"before:shadow-sm border-b mt-px md:mt-0 translate-y-1/8 h-3-1/4 md:h-2-3/4 lg:h-2-1/4 before:bottom-0":i||t.value,"before:shadow-lg active:before:shadow-sm border-b-1/4 md:border-b-1/4 active:border-b active:mt-px md:active:mt-0 active:translate-y-1/8 active:h-3-1/4 md:active:h-2-3/4 lg:active:h-2-1/4 before:-bottom-1/4 md:before:-bottom-1/4 active:before:bottom-0 drop backdrop-blur-[1px]":!i&&!t.value,[s.$attrs.class]:s.$attrs.class}]),"data-active":i},[K(s.$slots,"default")],10,ne)]),_:3},16))}}),ie=6,ce=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),le=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),P=(n,...e)=>n.map((t,o)=>`${t}${e[o]||""}`).join(""),V=P`
#define PI 3.1415926535897932384626433832795
#define TAU PI*2.0

uniform vec2 resolution;
`;P`
#define EPSILON 0.000001

bool approximately(float a, float b) {
	return abs(a - b) <= EPSILON;
}
`;const j=P`
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
vec3 cct2rgb(float kelvin) {
	kelvin = clamp(kelvin, 1000., 40000.) / 100.;
	return vec3(
		kelvin <= 66. ? 1.0 : clamp(329.698727446 * (pow(kelvin - 60., -0.1332047592)), 0., 255.) / 255.,
		kelvin <= 66. ?	clamp(99.4708025861 * log(kelvin) - 161.1195681661, 0., 255.) / 255. : clamp(288.1221695283 * (pow(kelvin - 60., -0.0755148492)), 0., 255.) / 255.,
		kelvin >= 66. ? 1.0 : kelvin <= 19. ? 0. : clamp(138.5177312231 * log(kelvin - 10.) - 305.0447927307, 0., 255.) / 255.
	);
}
`;P`
// Evaluates the parametric equation of a quadratic bezier
// ---> a(1-t)^2 + 2b(1-t)t + ct^2
vec2 posBezier(in vec2 a, in vec2 b, in vec2 c, in float t) {
	float tInv = 1.0 - t;
	return a * tInv * tInv + b * 2.0 * t * tInv + c * t * t;
}

// vec2(shortest distance, parameter of closest point)
// clampRes flags whether the results should be held in the range [tmin, tmax]
float dot2(in vec2 v) { return dot(v, v); }
vec2 mapBezier(in vec2 p, in vec2 v1, in vec2 v2, in vec2 v3, in float tmin, in float tmax, in bool clampRes) {
    vec2 c1 = p - v1;
    vec2 c2 = 2.0 * v2 - v3 - v1;
    vec2 c3 = v1 - v2;

    // Cubic coefficients ---> t3*t^3 + t2*t^2 + t1*t + t0*t^0
    float t3 = dot(c2, c2);
    float t2 = dot(c3, c2) * 3.0;
    float t1 = dot(c1, c2) + 2.0 * dot(c3, c3);
    float t0 = dot(c1, c3);

    // Reduce by dividing by leading coefficient
    // This simplifies out a lot of things
    t2 /= t3, t1 /= t3, t0 /= t3;

    // Depressed cubic coefficients (p and q) and precomputation
    float t22 = t2 * t2;
    vec2 pq = vec2(t1 - t22 / 3.0, t22 * t2 / 13.5 - t2 * t1 / 3.0 + t0);
    float ppp = pq.x * pq.x * pq.x, qq = pq.y * pq.y;

    float p2 = abs(pq.x);
    float r1 = 1.5 / pq.x * pq.y;

    // Solutions and details gathered from here: https://en.wikipedia.org/wiki/Cubic_equation
    if (qq * 0.25 + ppp / 27.0 > 0.0) { // One real root, use hyperbolic trig
        float r2 = r1 * sqrt(3.0 / p2), root;
        if (pq.x < 0.0) root = sign(pq.y) * cosh(acosh(r2 * -sign(pq.y)) / 3.0);
        else root = sinh(asinh(r2) / 3.0);
        root = -2.0 * sqrt(p2 / 3.0) * root - t2 / 3.0;
        if (clampRes) root = clamp(root, tmin, tmax);
        return vec2(length(p - posBezier(v1, v2, v3, root)), root);
    }

    else { // Three real roots (only need to use two), use "normal" trig
        float ac = acos(r1 * sqrt(-3.0 / pq.x)) / 3.0; // 4pi/3 goes here --v
        vec2 roots = 2.0 * sqrt(-pq.x / 3.0) * cos(vec2(ac, ac - 4.18879020479)) - t2 / 3.0;
        if (clampRes) roots = clamp(roots, tmin, tmax);
        float d1 = dot2(p - posBezier(v1, v2, v3, roots.x));
        float d2 = dot2(p - posBezier(v1, v2, v3, roots.y));
        return d1 < d2 ? vec2(sqrt(d1), roots.x) : vec2(sqrt(d2), roots.t);
    }
}
`;var ve=U`#version 300 es

precision highp float;

${V}
${j}

in vec4 a_position;
in vec4 a_dimensions;
in vec4 a_radius;
in vec4 a_state;
in vec2 a_texcoord;
in vec4 a_color;
out vec4 v_dimensions;
out vec4 v_radius;
out vec4 v_state;
out vec2 v_texcoord;
out vec4 v_color;
flat out int instance_id;

uniform float time;
uniform float aspect_ratio;
uniform mat4 projection;
mat4 view = mat4(1.0);

uniform int n_buttons;

void main() {
	int id = gl_InstanceID;

	float f_n_buttons = float(n_buttons);
	float pos = float(id) / (f_n_buttons - 1.);

	vec2 button_center = vec2(
		((a_dimensions.x + a_dimensions.z / 2.) / resolution.x * 2. - 1.) * aspect_ratio,
		((a_dimensions.y + a_dimensions.w / 2.) / resolution.y) * -2. + 1.
	);

	mat4 position_matrix = mat4(
		1,															0,														 0, 0,
		0,															1,														 0, 0,
		0, 															0, 														 1, 0,
		button_center.x, button_center.y, 0, 1
	);

	mat4 scale_matrix = mat4(
		(a_dimensions.z) / resolution.x * aspect_ratio, 0,			0, 0,
		0, 		 (a_dimensions.w) / resolution.y, 0, 0,
		0, 		 0,			1, 0,
		0, 		 0, 		0, 1
	);

  gl_Position = projection * view * position_matrix * scale_matrix * a_position;

	// float slow_time = 0.0;//time * 0.25;
	// vec3 color = mix(vec3(0. + slow_time, 1.0, 1.0), vec3(0.7 + slow_time, 0.75, 0.75), pos);
	// color.x = mod(color.x, 1.0);
	vec3 color = vec3(1.0, 0.0, 1.0);

	// Set vertex color
	v_color = vec4(color, 1.0); //a_color;
	v_texcoord = a_texcoord;
	v_dimensions = a_dimensions;
	v_radius = a_radius;
	v_state = a_state;
	instance_id = id;
}
`,de=U`#version 300 es

precision highp float;

${V}
${j}
in vec4 v_dimensions;
in vec4 v_radius;
in vec4 v_state;
in vec4 v_color;
in vec2 v_texcoord;
flat in int instance_id;
out vec4 out_color;

uniform float time;
uniform float aspect_ratio;

float random (in vec2 _st) {
  return fract(sin(dot(_st.xy,
      vec2(12.9898,78.233)))*
      43758.5453123);
}

// Simplex 2D noise
//
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float rectSDF(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + vec2(r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;
}

float sdRoundBox( in vec2 p, in vec2 b, in vec4 r )
{
    r.xy = (p.x>0.0)?r.xy : r.zw;
    r.x  = (p.y>0.0)?r.x  : r.y;
    vec2 q = abs(p)-b+r.x;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
}

void addObject(inout float dist, inout vec3 color, float d, vec3 c) {
  if (dist > d) {
      dist = d;
      color = c;
  }
}

void scene(in vec2 pos, out vec3 color, out float dist, in vec2 uv) {
  dist = 1e9; color = vec3(0,0,0);
  vec3 c = hsv2rgb(v_color.rgb);
  float p = 1.5;
  float scalar = 1.1;
  float iid = float(1 + instance_id);
  float base = 0.7 + snoise(vec2(0., 0.) + time * 3.) * 0.05;
  if (v_state.x == 1.) {
    // base *= 0.9 + snoise(pos * 2.0 + float(instance_id) + time * 2.0) * 0.8;
  // } else {
  //   base = 0.7 + snoise(vec2(iid, iid*iid) + time * 1.) * 0.1;
  }
  c = vec3(pow(base + scalar * c.r, p), pow(base + scalar * c.g, p), pow(base + scalar * c.b, p));
  addObject(dist, color, rectSDF(pos - vec2(0.0, 0.25), 	vec2(0.33, 0.075), 0.05), c);
}

void trace(vec2 p, vec2 dir, out vec3 c, in vec2 uv) {
  for (int j = 0; j < 8; j++) {
      float d;
      scene(p, c, d, uv);
      if (d < 1e-3) {
          //c = vec3(0,.1,0);
          return;
      }
      if (d > 1e1) break;
      p -= dir * d;
  }
  c = vec3(0,0,0);
}

#define SAMPLES 128

void main() {
  // out_color = vec4(1.0, 0.0, 0.0, 1.0);

  vec2 uv = v_texcoord;
  float aspect = v_dimensions.z/v_dimensions.w;
  vec2 p = (2.0*uv*v_dimensions.zw-v_dimensions.zw)/v_dimensions.w;

  // vec3 col = vec3(0,0,0);
  // for (int i = 0; i < SAMPLES; i++) {
  //     float t = (float(i) + random(uv+float(i))) / float(SAMPLES) * 2. * 3.1415;
  //     vec3 c;
  //     trace(p, vec2(cos(t), sin(t)), c, uv);
  //     col += c;
  // }
  // col /= float(SAMPLES);

  // vec3 hsv = rgb2hsv(col);

  // // Output to screen
  // // if (resolution.x >= 1024.) {
  float d = rectSDF(p - vec2(0.0, 0.25), vec2(0.75, 0.01), 0.1);
  float d2 = d;
  float noise_value = snoise(vec2(0., 0.) + time * 3.);

  d += 0.7 + noise_value * 0.033;
  d2 += 0.7 + (1.0-noise_value) * 0.015;
  // if (v_state.x == 1.) {
  //   d = 1.0 - sqrt(smoothstep(0.5, 2.0, d));
  //   vec3 c = vec3(0.262, 0.215, 0.607);
  //   out_color = vec4(sqrt(c * 4.), d);
  // } else {
    d = (1.0 - sqrt(smoothstep(smoothstep(0.0, 2.0, 1.0-v_state.y), 3.0, d)));
    d2 = (1.0 - sqrt(smoothstep(smoothstep(2., 0.5, 1.0-v_state.y), 4.0, d2)));
    // vec3 lit_hsv = hsv2rgb(0.0, 1.0, 1.0);
    d = (d * v_state.y) + (d2 * (1.0 - v_state.y));// + snoise(uv + vec2(time * 1., 0.0)) * (0.5 - v_state.y);


    float r = sdRoundBox(p, vec2(aspect, 1.0), v_radius/24.);
    vec4 c = vec4(hsv2rgb(vec3(0.7, sqrt(1.0 - d)/2., pow(d, 0.8))), 0.1 + d / 1.);
    out_color = (r > 0.0) ? vec4(0.0) : c;//smoothstep(vec4(1.0), vec4(vec3(0.0), 1.0), r);//vec4(hsv2rgb(vec3(0.7, sqrt(1.0 - d)/2., pow(d, 0.8))), 0.5 + d / 1.);
  // }
  //   // out_color = vec4(col*4.0, mix(0., hsv.z, smoothstep(0.78, .78, 1.0 - d)));
  //   // out_color = vec4(col*4.0, hsv.z);//mix(0., hsv.z, smoothstep(0.4, 1.0, 1.0 - d)));
  //   out_color = vec4(col, hsv.z);
  // // } else {
  // //   out_color = vec4(col*4.0, hsv.z);
  // // }
}
`;function ue(n){const{gl:e,createUniform:t,createAttribute:o,render:s,destroy:l}=re(n,ve,de,{devicePixelRatio:1});e.enable(e.BLEND),e.blendFuncSeparate(e.SRC_ALPHA,e.ONE,e.SRC_ALPHA_SATURATE,e.ONE_MINUS_SRC_ALPHA);const _=o("a_state"),m=o("a_dimensions"),i=o("a_radius"),h=o("a_position"),r=o("a_texcoord"),d=t("1f","time"),v=t("1f","aspect_ratio"),p=t("1i","n_buttons"),q=t("Matrix4fv","projection"),R=B(e,ce),b=B(e,le);e.bindBuffer(e.ARRAY_BUFFER,b),e.enableVertexAttribArray(r),e.vertexAttribPointer(r,2,e.FLOAT,!1,0,0);let S=0,g=0,k=0,x,y,u;return{setNButtons(c){k=c,p(c)},setButtonDimensions(c){x&&N(e,x),x=B(e,Float32Array.from(c)),e.bindBuffer(e.ARRAY_BUFFER,x),e.enableVertexAttribArray(m),e.vertexAttribPointer(m,4,e.FLOAT,!1,0,0),e.vertexAttribDivisor(m,1)},setButtonRadius(c){y&&N(e,y),y=B(e,Float32Array.from(c)),e.bindBuffer(e.ARRAY_BUFFER,y),e.enableVertexAttribArray(i),e.vertexAttribPointer(i,4,e.FLOAT,!1,0,0),e.vertexAttribDivisor(i,1)},setButtonStates(c){u&&N(e,u),u=B(e,Float32Array.from(c)),e.bindBuffer(e.ARRAY_BUFFER,u),e.enableVertexAttribArray(_),e.vertexAttribPointer(_,4,e.FLOAT,!1,0,0),e.vertexAttribDivisor(_,1)},render(){const{height:c,width:D,ratio:A}=s(!1);if(g==0)g=Date.now();else{let z=Date.now();S+=(z-g)*.001,g=z}d(S),v(A),q(!1,ae(-(A||1),A||1,-1,1,-1,1)),e.bindBuffer(e.ARRAY_BUFFER,R),e.enableVertexAttribArray(h),e.vertexAttribPointer(h,2,e.FLOAT,!1,0,0),e.drawArraysInstanced(e.TRIANGLES,0,ie,k)},destroy:l}}function fe(n,e){let t,o=0,s=1;const l=F(!1);function _(r){if(!!t)if(r==null||n.value==null)t.setNButtons(0),t.setButtonDimensions([]);else{const{x:d,y:v}=n.value.getBoundingClientRect(),p=[],q=[],R=[];for(let b of Array.from(r.children))if(b instanceof HTMLElement){const{x:S,y:g,width:k,height:x}=b.getBoundingClientRect();p.push(S-d,g-v,k,x)-4;const y=(x-o)/s;R.push(b.dataset.active=="true"?1:0,y,b.matches(":hover")?1:0,0);const u=window.getComputedStyle(b),c=u.getPropertyValue("border-top-left-radius"),D=u.getPropertyValue("border-top-right-radius"),A=u.getPropertyValue("border-bottom-left-radius"),z=u.getPropertyValue("border-bottom-right-radius");q.push(parseInt(D),parseInt(z),parseInt(A),parseInt(c))}t.setNButtons(r.children.length),t.setButtonDimensions(p),t.setButtonRadius(q),t.setButtonStates(R),t.render()}}function m(){if(e.value){const r=e.value.querySelector('[data-active="false"]'),d=e.value.querySelector('[data-active="true"]');if(!r)throw new Error("Unable to find unpressed button");o=r.getBoundingClientRect().height-6,d&&(o=d.getBoundingClientRect().height),s=r.getBoundingClientRect().height-o}}let i=!1;function h(){t&&(_(e.value),t.render()),i&&requestAnimationFrame(h)}return W(()=>X(()=>{i=!0;const r=new ResizeObserver(m),d="sync";T(n,v=>{if(t&&t.destroy(),v)try{t=ue(v),h(),l.value=!1}catch{l.value=!0}},{flush:d,immediate:!0}),T(e,(v,p)=>{v&&(m(),r.observe(v)),p&&r.unobserve(p),_(v)},{flush:d,immediate:!0})})),Z(()=>{i=!1,t&&t.destroy()}),{error:l}}const _e={class:""},me={key:0,class:"absolute inset-1 top-1-1/4 bg-gradient-to-b from-neutral-875 to-neutral-700"},pe=$({__name:"MastheadNavButtonBackground",props:{nav:null},setup(n){const e=n,t=F(),{error:o}=fe(t,ee(e,"nav"));return(s,l)=>(C(),I("div",_e,[a("canvas",{class:"w-full h-full",ref_key:"canvas",ref:t},null,512),te(o)?(C(),I("div",me)):oe("",!0)]))}}),be={class:"fixed z-20 inset-x-0 bottom-0 sm:relative flex lg:px-3/4"},xe=a("div",{class:"flex-1"},null,-1),he={class:"w-full flex max-w-lg"},ge=a("div",{class:"hidden sm:block flex-1"},null,-1),ye={class:"flex-1 sm:flex-none flex flex-col justify-center"},we={class:"flex items-center relative pointer-events-auto"},Ae=a("div",{id:"nav-base",class:"absolute inset-0 top-1/4 lg:rounded-2/3 bg-neutral-850/60 backdrop-blur-sm"},null,-1),Be=L("Home"),qe=L("Portfolio"),Re=L("Resum\xE9"),Se={class:"hidden bg-gradient-to-b from-neutral-825 to-neutral-800"},ke=a("div",{class:"flex-1"},null,-1),ze=a("div",{class:"flex-1"},null,-1),Ee=$({__name:"Masthead",setup(n){const e=F();return(t,o)=>{const s=O("SvgIcon"),l=O("AbstractButton");return C(),I("header",be,[xe,a("div",he,[f(se,{class:"hidden sm:block h-5"}),ge,a("div",ye,[a("div",we,[Ae,f(pe,{nav:e.value,class:"absolute inset-0 !supports-backdrop:hidden overflow-hidden"},null,8,["nav"]),a("nav",{ref_key:"nav",ref:e,class:"flex-1 flex items-center border-group rounded-group"},[f(E,{class:"sm:rounded-1/2",to:"/"},{default:w(()=>[Be]),_:1}),f(E,{class:"sm:rounded-1/2",to:"/portfolio"},{default:w(()=>[qe]),_:1}),f(E,{class:"sm:rounded-1/2",to:"/resume"},{default:w(()=>[Re]),_:1})],512)]),a("div",Se,[ke,f(l,{is:"a",href:"https://github.com/JeffSchofield",target:"_blank",rel:"noopener",class:"p-1/4"},{default:w(()=>[f(s,{name:"github",class:"h-1-1/2 fill-current"})]),_:1}),f(l,{is:"a",href:"https://twitter.com/JeffScript",target:"_blank",rel:"noopener",class:"p-1/4"},{default:w(()=>[f(s,{name:"twitter",class:"h-1-1/2 fill-current"})]),_:1})])])]),ze])}}});export{Ee as _,pe as a,E as b};
