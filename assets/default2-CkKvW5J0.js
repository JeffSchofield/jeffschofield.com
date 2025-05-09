import{_ as G,a as O}from"./Masthead.vue_vue_type_script_setup_true_lang-BGFdxI5K.js";import{u as W}from"./about-DNYk4ovx.js";import{g as $,n as Q,c as S,o as X,p as T}from"./utils-B_gNnKp6.js";import{e as j,D as Z,E as ee,g as U,M as te,p as K,a8 as oe,c as v,o as u,a as e,s as re,x,r as V,d as n,w as g,b as R,F as z,y as N,B as I,a9 as D,S as se}from"./index-RO_jvqb9.js";import"./SvgLogo.vue_vue_type_script_setup_true_lang-DaCc2Ctd.js";const ae=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),ie=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),L=(f,...t)=>f.map((r,i)=>`${r}${t[i]||""}`).join(""),Y=L`
#define PI 3.1415926535897932384626433832795
#define TAU PI*2.0

uniform vec2 resolution;
`;L`
#define EPSILON 0.000001

bool approximately(float a, float b) {
	return abs(a - b) <= EPSILON;
}
`;const H=L`
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
`;L`
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
`;const ne=$`#version 300 es

precision highp float;

${Y}
${H}

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
`,le=$`#version 300 es

precision highp float;

${Y}
${H}
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
`;function ce(f){const{gl:t,createUniform:r,createAttribute:i,render:h,destroy:o}=Q(f,ne,le,{devicePixelRatio:1});t.enable(t.BLEND),t.blendFuncSeparate(t.SRC_ALPHA,t.ONE,t.SRC_ALPHA_SATURATE,t.ONE_MINUS_SRC_ALPHA);const b=i("a_state"),d=i("a_dimensions"),l=i("a_radius"),s=i("a_position"),a=i("a_texcoord"),m=r("1f","time"),p=r("1f","aspect_ratio"),_=r("1i","n_buttons"),C=r("Matrix4fv","projection"),F=S(t,ae),y=S(t,ie);t.bindBuffer(t.ARRAY_BUFFER,y),t.enableVertexAttribArray(a),t.vertexAttribPointer(a,2,t.FLOAT,!1,0,0);let E=0,A=0,k,w,B;return{setNButtons(c){_(c)},setButtonDimensions(c){k&&T(t,k),k=S(t,Float32Array.from(c)),t.bindBuffer(t.ARRAY_BUFFER,k),t.enableVertexAttribArray(d),t.vertexAttribPointer(d,4,t.FLOAT,!1,0,0),t.vertexAttribDivisor(d,1)},setButtonRadius(c){w&&T(t,w),w=S(t,Float32Array.from(c)),t.bindBuffer(t.ARRAY_BUFFER,w),t.enableVertexAttribArray(l),t.vertexAttribPointer(l,4,t.FLOAT,!1,0,0),t.vertexAttribDivisor(l,1)},setButtonStates(c){B&&T(t,B),B=S(t,Float32Array.from(c)),t.bindBuffer(t.ARRAY_BUFFER,B),t.enableVertexAttribArray(b),t.vertexAttribPointer(b,4,t.FLOAT,!1,0,0),t.vertexAttribDivisor(b,1)},render(){const{height:c,width:M,ratio:q}=h(!1);if(A==0)A=Date.now();else{let P=Date.now();E+=(P-A)*.001,A=P}m(E),p(q),C(!1,X(-(q||1),q||1,-1,1,-1,1)),t.bindBuffer(t.ARRAY_BUFFER,F),t.enableVertexAttribArray(s),t.vertexAttribPointer(s,2,t.FLOAT,!1,0,0)},destroy:o}}function de(f,t){let r,i=0,h=1;const o=j(!1);function b(a){if(r)if(a==null||f.value==null)r.setNButtons(0),r.setButtonDimensions([]);else{const{x:m,y:p}=f.value.getBoundingClientRect(),_=[],C=[],F=[];for(let y of Array.from(a.children))if(y instanceof HTMLElement){const{x:E,y:A,width:k,height:w}=y.getBoundingClientRect();_.push(E-m,A-p,k,w)-4;const B=(w-i)/h;F.push(y.dataset.active=="true"?1:0,B,y.matches(":hover")?1:0,0);const c=window.getComputedStyle(y),M=c.getPropertyValue("border-top-left-radius"),q=c.getPropertyValue("border-top-right-radius"),P=c.getPropertyValue("border-bottom-left-radius"),J=c.getPropertyValue("border-bottom-right-radius");C.push(parseInt(q),parseInt(J),parseInt(P),parseInt(M))}r.setNButtons(a.children.length),r.setButtonDimensions(_),r.setButtonRadius(C),r.setButtonStates(F),r.render()}}function d(){if(t.value){const a=t.value.querySelector('[data-active="false"]'),m=t.value.querySelector('[data-active="true"]');if(!a)throw new Error("Unable to find unpressed button");i=a.getBoundingClientRect().height-6,m&&(i=m.getBoundingClientRect().height),h=a.getBoundingClientRect().height-i}}let l=!1;function s(){r&&(b(t.value),r.render()),l&&requestAnimationFrame(s)}return Z(()=>ee(()=>{l=!0;const a=new ResizeObserver(d),m="sync";U(f,p=>{if(r&&r.destroy(),p)try{r=ce(p),s(),o.value=!1}catch{o.value=!0}},{flush:m,immediate:!0}),U(t,(p,_)=>{p&&(d(),a.observe(p)),_&&a.unobserve(_),b(p)},{flush:m,immediate:!0})})),te(()=>{l=!1,r&&r.destroy()}),{error:o}}const ve={class:""},ue={key:0,class:"absolute inset-1 top-1-1/4 bg-gradient-to-b from-neutral-875 to-neutral-700"},fe=K({__name:"MastheadNavButtonBackground",props:{nav:{}},setup(f){const t=f,r=j(),{error:i}=de(r,oe(t,"nav"));return(h,o)=>(u(),v("div",ve,[e("canvas",{class:"w-full h-full",ref_key:"canvas",ref:r},null,512),x(i)?(u(),v("div",ue)):re("",!0)]))}}),pe={class:"order-1 pointer-fine:order-2 lg:order-2 flex-1 w-full self-center flex flex-col md:flex-row max-w-xl"},me={class:"z-50 md:z-30 md:relative flex flex-col md:w-14 lg:w-16"},xe={class:"fixed bottom-0 h-full max-h-screen flex flex-col shadow-lg w-full md:w-14 lg:w-16"},be={class:"flex -mt-1/4"},_e={class:"flex-1 flex items-center relative"},ge={class:"hidden md:flex flex-col border-t border-neutral-300 dark:border-neutral-900 divide-y divide-neutral-300 dark:divide-neutral-900"},he={class:"divide-y divide-neutral-250 dark:divide-neutral-900"},ye={class:"flex divide-x divide-neutral-900"},we={class:"flex-1 hidden md:flex flex-col divide-y divide-neutral-300 dark:divide-neutral-900"},Ae={class:"flex-1 relative bg-neutral-850"},ke={class:"absolute inset-0 flex flex-col overflow-y-auto"},Be={class:"flex flex-col"},qe={class:"flex flex-wrap gap-1/2 max-h-4 overflow-hidden p-1/2 bg-neutral-950"},Se={class:"px-1/2 py-1/4 rounded-full whitespace-nowrap leading-3/4 bg-primary-450/50 hover:bg-primary-450"},Re={class:"flex flex-col"},ze={class:"flex flex-wrap gap-1/2 max-h-2-1/2 overflow-hidden p-1/2 bg-neutral-950"},Ce={class:"px-1/2 py-1/4 rounded-full whitespace-nowrap leading-1 bg-primary-450/50 hover:bg-primary-450"},Fe={class:"flex flex-col"},Ee={class:"flex flex-wrap gap-1/2 max-h-4 overflow-hidden p-1/2 bg-neutral-950"},Pe={class:"px-1/2 py-1/4 rounded-full whitespace-nowrap leading-3/4 bg-primary-450/50 hover:bg-primary-450"},Ne={class:"flex flex-col"},Ie={class:"flex flex-wrap gap-1/2 max-h-4 overflow-hidden p-1/2 bg-neutral-950"},De={class:"px-1/2 py-1/4 rounded-full whitespace-nowrap leading-3/4 bg-primary-450/50 hover:bg-primary-450"},Le={class:"flex flex-col"},Ue=K({__name:"default2",setup(f){const{title:t,abilities:r}=W(),i=j();return(h,o)=>{const b=V("ModalView"),d=V("SvgIcon"),l=V("AbstractButton");return u(),v(z,null,[n(b),o[20]||(o[20]=e("div",{class:"absolute top-0 inset-x-0 h-8 z-behind bg-gradient-to-b from-primary-200/10 to-transparent"},null,-1)),e("div",pe,[e("div",me,[e("div",xe,[n(G,{class:"hidden md:flex"}),e("div",be,[o[4]||(o[4]=e("div",{class:"flex-1 hidden sm:block md:hidden bg-neutral-850"},null,-1)),e("div",_e,[o[3]||(o[3]=e("div",{class:"absolute inset-0 top-1/4 bg-neutral-200/80 dark:bg-neutral-850/60"},null,-1)),n(fe,{nav:i.value,class:"absolute inset-0 !supports-backdrop:hidden"},null,8,["nav"]),e("nav",{ref_key:"nav",ref:i,class:"flex-1 flex items-center border-group rounded-group"},[n(O,{to:"/"},{default:g(()=>o[0]||(o[0]=[R("About")])),_:1}),n(O,{to:"/portfolio"},{default:g(()=>o[1]||(o[1]=[R("Portfolio")])),_:1}),n(O,{to:"/resume"},{default:g(()=>o[2]||(o[2]=[R("Resumé")])),_:1})],512)])]),e("div",ge,[o[12]||(o[12]=e("div",{class:"p-1/2 pt-1 bg-gradient-to-br from-neutral-150 to-neutral-100 dark:from-neutral-750 dark:via-neutral-775 dark:to-neutral-800"},[e("h2",{class:"leading-3/4 font-semibold text-sm text-secondary-300"}," Connect With Me ")],-1)),e("dl",he,[n(l,{is:"a",href:"https://github.com/JeffSchofield",target:"_blank",rel:"noopener",class:"flex items-center gap-1/2 p-1/2 even:bg-neutral-125 even:dark:bg-neutral-850 odd:bg-neutral-175 odd:dark:bg-neutral-875 even:hover:bg-primary-300/30 odd:hover:bg-primary-300/30"},{default:g(()=>[n(d,{name:"github",class:"h-1 fill-current"}),o[5]||(o[5]=e("div",{class:"flex-1 flex gap-1/2"},[e("dt",{class:"col-span-2 text-neutral-150"},"GitHub"),e("div",{class:"flex-1 leader text-neutral-700"}),e("dd",{class:"col-span-5 font-semibold"},"JeffSchofield")],-1))]),_:1}),n(l,{is:"a",href:"https://github.com/ShiftLimits",target:"_blank",rel:"noopener",class:"flex items-center gap-1/2 p-1/2 even:bg-neutral-125 even:dark:bg-neutral-850 odd:bg-neutral-175 odd:dark:bg-neutral-875 even:hover:bg-primary-300/30 odd:hover:bg-primary-300/30"},{default:g(()=>[n(d,{name:"github",class:"h-1 fill-current"}),o[6]||(o[6]=e("div",{class:"flex-1 flex gap-1/2"},[e("dt",{class:"col-span-2 text-neutral-150"},"Open Source"),e("div",{class:"flex-1 leader text-neutral-700"}),e("dd",{class:"col-span-5 font-semibold"},"ShiftLimits")],-1))]),_:1}),o[10]||(o[10]=e("div",{class:"px-1/2 py-1/4 bg-neutral-825"},[e("h3",{class:"leading-1/2 font-semibold text-xs text-primary-200"}," Contact ")],-1)),n(l,{is:"a",href:"mailto:contact@jeffschofield.com",target:"_blank",rel:"noopener",class:"flex items-center gap-1/2 p-1/2 even:bg-neutral-125 even:dark:bg-neutral-850 odd:bg-neutral-175 odd:dark:bg-neutral-875 even:hover:bg-primary-300/30 odd:hover:bg-primary-300/30"},{default:g(()=>[n(d,{name:"mail",class:"h-1 fill-current"}),o[7]||(o[7]=e("div",{class:"flex-1 flex gap-1/2"},[e("dt",{class:"col-span-2 text-neutral-150"},"Email"),e("div",{class:"flex-1 leader text-neutral-700"}),e("dd",{class:"col-span-5 font-semibold"}," contact@jeffschofield.com ")],-1))]),_:1}),o[11]||(o[11]=e("div",{class:"px-1/2 py-1/4 bg-neutral-825"},[e("h3",{class:"leading-1/2 font-semibold text-xs text-primary-200"}," Socials ")],-1)),e("div",ye,[n(l,{is:"a",href:"https://twitter.com/JeffScript",target:"_blank",rel:"noopener",class:"flex-1 flex items-center justify-center gap-1/2 p-1/2 bg-neutral-125 dark:bg-neutral-875 hover:bg-primary-300/30"},{default:g(()=>[n(d,{name:"twitter",class:"h-1 fill-current"}),o[8]||(o[8]=R(" Twitter "))]),_:1}),n(l,{is:"a",href:"https://www.linkedin.com/in/jeff-schofield-dev/",target:"_blank",rel:"noopener",class:"flex-1 flex items-center justify-center gap-1/2 p-1/2 bg-neutral-125 dark:bg-neutral-875 hover:bg-primary-300/30"},{default:g(()=>[n(d,{name:"linkedin",class:"h-1 fill-current"}),o[9]||(o[9]=R(" LinkedIn "))]),_:1})])])]),e("div",we,[o[17]||(o[17]=e("div",{class:"p-1/2 pt-1 border-t border-neutral-300 dark:border-neutral-900 bg-gradient-to-br from-neutral-150 to-neutral-100 dark:from-neutral-750 dark:via-neutral-775 dark:to-neutral-800"},[e("h2",{class:"leading-3/4 font-semibold text-sm text-secondary-300"}," Abilities ")],-1)),e("div",Ae,[e("div",ke,[e("div",Be,[o[13]||(o[13]=e("div",{class:"p-1/2 space-y-1/4 bg-neutral-900"},[e("div",{class:"leading-3/4 font-bold text-primary-300"}," Front End "),e("div",{class:"text-xs leading-1/2 text-neutral-300"}," Interfacing with humans. ")],-1)),e("div",qe,[(u(!0),v(z,null,N(x(r).filter(s=>s.categories.includes(x(D).FRONTEND)).sort((s,a)=>(a.importance||0)-(s.importance||0)).slice(0,5),s=>(u(),v("div",Se,I(s.name),1))),256))])]),e("div",Re,[o[14]||(o[14]=e("div",{class:"p-1/2 space-y-1/4 bg-neutral-900"},[e("div",{class:"leading-3/4 font-bold text-primary-300"}," Back End "),e("div",{class:"text-xs leading-1/2 text-neutral-300"}," Information structures for machines. ")],-1)),e("div",ze,[(u(!0),v(z,null,N(x(r).filter(s=>s.categories.includes(x(D).BACKEND)).sort((s,a)=>(a.importance||0)-(s.importance||0)).slice(0,5),s=>(u(),v("button",Ce,I(s.name),1))),256))])]),e("div",Fe,[o[15]||(o[15]=e("div",{class:"p-1/2 space-y-1/4 bg-neutral-900"},[e("div",{class:"leading-3/4 font-bold text-primary-300"}," DevOps "),e("div",{class:"text-xs leading-1/2 text-neutral-300"}," Infrastructure for the applications and the developers. ")],-1)),e("div",Ee,[(u(!0),v(z,null,N(x(r).filter(s=>s.categories.includes(x(D).DEVOPS)).sort((s,a)=>(a.importance||0)-(s.importance||0)).slice(0,5),s=>(u(),v("div",Pe,I(s.name),1))),256))])]),e("div",Ne,[o[16]||(o[16]=e("div",{class:"p-1/2 space-y-1/4 bg-neutral-900"},[e("div",{class:"leading-3/4 font-bold text-primary-300"}," Security "),e("div",{class:"text-xs leading-1/2 text-neutral-300"}," Protecting the precious. ")],-1)),e("div",Ie,[(u(!0),v(z,null,N(x(r).filter(s=>s.categories.includes(x(D).SECURITY)).sort((s,a)=>(a.importance||0)-(s.importance||0)).slice(0,5),s=>(u(),v("div",De,I(s.name),1))),256))])])])]),o[18]||(o[18]=e("div",{class:"h-2 bg-neutral-800"},null,-1))]),o[19]||(o[19]=e("div",{class:"p-1 bg-neutral-800"},"Copyright © 2022",-1))])]),e("div",Le,[se(h.$slots,"default")])])],64)}}});export{Ue as default};
