import{a as F,u as $,r as k,c as B,e as D,w as O,v as E,s as I,W as T,R as U,t as j,K as M,L as K,O as S,o as V,X as H,l as z,f as Y,k as G}from"./vendor.169cfbe7.js";import{c as W}from"./index.dead2e77.js";import{g as N,n as C,c as w,o as X,p as J}from"./utils.7a764719.js";const Q=["href","onClick","data-active"],ue=F({setup(s){const{afterEach:e}=$(),t=k(!1);function o(r){t.value=!0,r()}return e(()=>t.value=!1),(r,u)=>(B(),D(W,j(r.$attrs,{custom:""}),{default:O(({href:l,navigate:v,isActive:i})=>[E("a",{href:l,onClick:T(p=>o(v),["prevent"]),class:U(["transition-all duration-35 ease-in-out flex-1 relative before:z-behind before:absolute before:inset-0 h-3-1/2 md:h-3 lg:h-2-1/2 xs:px-1/2 xs:min-w-4 flex items-center justify-center text-xs xs:text-sm sm:text-base text-neutral-50 text-shadow font-bold bg-gradient-to-br from-neutral-150/10 to-neutral-200/30 dark:from-neutral-750/0 dark:to-neutral-800/20 can-hover:hover:bg-neutral-0/20 can-hover:dark:hover:bg-neutral-0/10 border border-neutral-300/50 dark:border-neutral-875/70 border-b-neutral-350/40 dark:border-b-neutral-850/40 border-t-neutral-250/50 dark:border-t-neutral-850/70 no-select",{"before:shadow-sm border-b mt-px md:mt-0 translate-y-1/8 h-3-1/4 md:h-2-3/4 lg:h-2-1/4 before:bottom-0":i||t.value,"before:shadow-lg active:before:shadow-sm border-b-1/4 md:border-b-1/4 active:border-b active:mt-px md:active:mt-0 active:translate-y-1/8 active:h-3-1/4 md:active:h-2-3/4 lg:active:h-2-1/4 before:-bottom-1/4 md:before:-bottom-1/4 active:before:bottom-0 drop":!i&&!t.value,[r.$attrs.class]:r.$attrs.class}]),"data-active":i},[I(r.$slots,"default")],10,Q)]),_:3},16))}}),Z=6,ee=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),te=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),A=(s,...e)=>s.map((t,o)=>`${t}${e[o]||""}`).join(""),P=A`
#define PI 3.1415926535897932384626433832795
#define TAU PI*2.0

uniform vec2 resolution;
`;A`
#define EPSILON 0.000001

bool approximately(float a, float b) {
	return abs(a - b) <= EPSILON;
}
`;const L=A`
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
`;A`
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
`;var oe=N`#version 300 es

precision highp float;

${P}
${L}

in vec4 a_position;
in vec4 a_dimensions;
in vec4 a_state;
in vec2 a_texcoord;
in vec4 a_color;
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
	v_state = a_state;
	instance_id = id;
}
`,re=N`#version 300 es

precision highp float;

${P}
${L}
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
  addObject(dist, color, rectSDF(pos - vec2(0.5, 0.55), 	vec2(0.33, 0.075), 0.05), c);
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

  // vec3 col = vec3(0,0,0);
  // for (int i = 0; i < SAMPLES; i++) {
  //     float t = (float(i) + random(uv+float(i))) / float(SAMPLES) * 2. * 3.1415;
  //     vec3 c;
  //     trace(uv, vec2(cos(t), sin(t)), c, uv);
  //     col += c;
  // }
  // col /= float(SAMPLES);

  // vec3 hsv = rgb2hsv(col);

  // // Output to screen
  // // if (resolution.x >= 1024.) {
  float d = rectSDF(uv - vec2(0.5, 0.75), vec2(0.2, 0.01), 0.01);
  float d2 = d;
  float noise_value = snoise(vec2(0., 0.) + time * 3.);

  d += 0.7 + noise_value * 0.033;
  d2 += 0.7 + (1.0-noise_value) * 0.015;
  // if (v_state.x == 1.) {
  //   d = 1.0 - sqrt(smoothstep(0.5, 2.0, d));
  //   vec3 c = vec3(0.262, 0.215, 0.607);
  //   out_color = vec4(sqrt(c * 4.), d);
  // } else {
    d = (1.0 - sqrt(smoothstep(smoothstep(0.0, 2.0, 1.0-v_state.y), 2.0, d)));
    d2 = (1.0 - sqrt(smoothstep(smoothstep(2., 0.5, 1.0-v_state.y), 4.0, d2)));
    // vec3 lit_hsv = hsv2rgb(0.0, 1.0, 1.0);
    d = (d * v_state.y) + (d2 * (1.0 - v_state.y));// + snoise(uv + vec2(time * 1., 0.0)) * (0.5 - v_state.y);


    out_color = vec4(hsv2rgb(vec3(0.7, sqrt(1.0 - d)/2., pow(d, 0.8))), 0.8 + d / 8.);
  // }
  //   // out_color = vec4(col*4.0, mix(0., hsv.z, smoothstep(0.78, .78, 1.0 - d)));
  //   // out_color = vec4(col*4.0, hsv.z);//mix(0., hsv.z, smoothstep(0.4, 1.0, 1.0 - d)));
  //   out_color = vec4(col, hsv.z);
  // // } else {
  // //   out_color = vec4(col*4.0, hsv.z);
  // // }
}
`;function ae(s){const{gl:e,createUniform:t,createAttribute:o,render:r,destroy:u}=J(s,oe,re,{devicePixelRatio:1});e.enable(e.BLEND),e.blendFuncSeparate(e.SRC_ALPHA,e.ONE,e.SRC_ALPHA_SATURATE,e.ONE_MINUS_SRC_ALPHA);const l=o("a_state"),v=o("a_dimensions"),i=o("a_position"),p=o("a_texcoord"),a=t("1f","time"),c=t("1f","aspect_ratio"),n=t("1i","n_buttons"),d=t("Matrix4fv","projection"),h=w(e,ee),b=w(e,te);e.bindBuffer(e.ARRAY_BUFFER,b),e.enableVertexAttribArray(p),e.vertexAttribPointer(p,2,e.FLOAT,!1,0,0);let g=0,x=0,y=0,f,_;return{setNButtons(m){y=m,n(m)},setButtonDimensions(m){f&&C(e,f),f=w(e,Float32Array.from(m)),e.bindBuffer(e.ARRAY_BUFFER,f),e.enableVertexAttribArray(v),e.vertexAttribPointer(v,4,e.FLOAT,!1,0,0),e.vertexAttribDivisor(v,1)},setButtonStates(m){_&&C(e,_),_=w(e,Float32Array.from(m)),e.bindBuffer(e.ARRAY_BUFFER,_),e.enableVertexAttribArray(l),e.vertexAttribPointer(l,4,e.FLOAT,!1,0,0),e.vertexAttribDivisor(l,1)},render(){const{height:m,width:ce,ratio:q}=r(!1);if(x==0)x=Date.now();else{let R=Date.now();g+=(R-x)*.001,x=R}a(g),c(q),d(!1,X(-(q||1),q||1,-1,1,-1,1)),e.bindBuffer(e.ARRAY_BUFFER,h),e.enableVertexAttribArray(i),e.vertexAttribPointer(i,2,e.FLOAT,!1,0,0),e.drawArraysInstanced(e.TRIANGLES,0,Z,y)},destroy:u}}function se(s,e){let t,o=0,r=1;const u=k(!1);function l(a){if(!!t)if(a==null||s.value==null)t.setNButtons(0),t.setButtonDimensions([]);else{const{x:c,y:n}=s.value.getBoundingClientRect(),d=[],h=[];for(let b of Array.from(a.children))if(b instanceof HTMLElement){const{x:g,y:x,width:y,height:f}=b.getBoundingClientRect();d.push(g-c,x-n,y,f)-4;const _=(f-o)/r;h.push(b.dataset.active=="true"?1:0,_,0,0)}t.setNButtons(a.children.length),t.setButtonDimensions(d),t.setButtonStates(h),t.render()}}function v(){if(e.value){const a=e.value.querySelector('[data-active="false"]'),c=e.value.querySelector('[data-active="true"]');if(!a||!c)throw new Error("Unable to find an unpressed button.");o=c.getBoundingClientRect().height,r=a.getBoundingClientRect().height-o}}let i=!1;function p(){t&&(l(e.value),t.render()),i&&requestAnimationFrame(p)}return M(()=>K(()=>{i=!0;const a=new ResizeObserver(v),c="sync";S(s,n=>{if(t&&t.destroy(),n)try{t=ae(n),p(),u.value=!1}catch{u.value=!0}},{flush:c,immediate:!0}),S(e,(n,d)=>{n&&(v(),a.observe(n)),d&&a.unobserve(d),l(n)},{flush:c,immediate:!0})})),V(()=>{i=!1,t&&t.destroy()}),{error:u}}const ie={class:""},ne={key:0,class:"absolute inset-1 top-1-1/4 bg-gradient-to-b from-neutral-875 to-neutral-700"},fe=F({props:{nav:null},setup(s){const e=s,t=k(),{error:o}=se(t,H(e,"nav"));return(r,u)=>(B(),z("div",ie,[E("canvas",{class:"w-full h-full",ref_key:"canvas",ref:t},null,512),Y(o)?(B(),z("div",ne)):G("",!0)]))}});export{fe as _,ue as a};
