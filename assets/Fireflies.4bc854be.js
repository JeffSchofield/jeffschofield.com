import{L as Oe,g as o,c as Q,r as xe,a as ee,b as c,d as De,e as Me,f as y,s as f,h as ce,i as _e,j as ue,k as m,l as ve,m as h}from"./utils.ad40f7b8.js";import{K as Ne,L as Ie,O as Se,o as Ge,r as n,q as C,P as $e,a as Xe,c as ze,l as Ye,v as He}from"./vendor.0830267b.js";import{a as Ve}from"./index.89fcf377.js";const I=new Oe;I.update_time_step=16;function qe(t){let e;function l(){if(t.value)try{e=dt(t.value),I.linkProgram(e)}catch{}}const{isLandingPage:_,onAfterPageEnter:d}=Ve();Ne(()=>Ie(()=>{I.init(),Se(t,u=>{e&&(I.unlinkProgram(e),e.destroy()),_.value==!0?l():d(l)},{flush:"sync",immediate:!0})})),Ge(()=>{e&&(I.unlinkProgram(e),e.destroy())})}const S=o`
uniform float dt;
uniform float time;
uniform vec2 resolution;
uniform float aspect_ratio;
`,ke=o`
mat2 rotationMatrix(float a) {
	return mat2(cos(a), -sin(a), sin(a), cos(a));
}
`,Y=o`
#define PI 3.1415926535897932384626433832795
#define TAU 6.283185307179586476925286766559
#define PI180 57.295779513082320876798154814105

uniform float seed;
float random(vec2 st) {
	return fract(sin(dot(st * seed, vec2(81.121, 54.986))) * 81942.124);
}

float randomBetween(vec2 st, float start, float end) {
	return start + (random(st) * (end - start));
}

vec2 randomVector(vec2 st) {
	return vec2(random(st * 0.00135) * 2.0 - 1.0, random(st * 0.00731) * 2.0 - 1.0);
}

float randomGaussian(vec2 st, float mean, float stddev, float skewness) {
	float U, V, R, Z, Y, value;

	U = random(st * .01301 + time * .201);
	V = random(st * .08934 + time * .387);
	// R = random(st * .04611 + time * .461);

	float magnitude = sqrt(-2.0 * log(U));
	float direction = 2.0 * PI * V;
	Z = magnitude * cos(direction);
	Y = magnitude * sin(direction);
	// if (R < 0.5) Z = sqrt(-2.0 * log(U)) * sin(2.0 * PI * V);
	// else Z = sqrt(-2.0 * log(U)) * cos(2.0 * PI * V);

	if (skewness == 0.0) {
		// Apply the stddev and mean.
		value = Z * stddev + mean;
	} else {
		float correlation = skewness / sqrt(1. + skewness * skewness);
		float correlated = correlation * Z + sqrt(1. - correlation * correlation) * Y;
		float z = Z >= 0.0 ? correlated : -correlated;
		value = z * stddev + mean;
	}

	return value;
}

float noise (in vec2 st) {
	vec2 i = floor(st);
	vec2 f = fract(st);

	// Four corners in 2D of a tile
	float a = random(i);
	float b = random(i + vec2(1.0, 0.0));
	float c = random(i + vec2(0.0, 1.0));
	float d = random(i + vec2(1.0, 1.0));

	vec2 u = f * f * (3.0 - 2.0 * f);

	return mix(a, b, u.x) +
					(c - a)* u.y * (1.0 - u.x) +
					(d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm (in vec2 st) {
		// Initial values
		float value = 0.8;
		float amplitude = .2;
		//
		// Loop of octaves
		for (int i = 0; i < OCTAVES; i++) {
				value += amplitude * noise(st);
				st *= 2.;
				amplitude *= .5;
		}
		return value;
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
`,We=o`
	uniform bool pointer_down;
	uniform vec4 pointer_state;
`,ye=o`
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
`,je=o`
	struct Particle {
		int state;

		// Body
		float width;
		float height;
		float agility;

		// Physics
		vec2 position;
		vec2 tile;
		vec2 velocity;
		float current_speed;

		// Firefly
		float phase;
		float brightness;
	};
`,Ke=o`
	const float N_TILES = 1000.0;
	vec2 getParticleWorldPosition(vec2 tile, vec2 position) {
		vec2 tile_resolution = resolution / N_TILES;
		vec2 tile_uv = ((tile * tile_resolution) + (tile_resolution / 2.0)) / resolution;
		vec2 tile_world_position = tile_uv * vec2(aspect_ratio * 2.0, 2.0) - vec2(aspect_ratio, 1.0);

		return tile_world_position + (position / vec2(N_TILES / aspect_ratio, N_TILES));
	}

	void getParticleNextTilePosition(out vec2 tile, out vec2 position) {
		if (position.x > 1.0) {
			position.x = mod(position.x + 1.0, 2.0) - 1.0;
			tile.x += 1.0;
		} else if (position.x < -1.0) {
			position.x = mod(position.x + 1.0, 2.0) - 1.0;
			tile.x -= 1.0;
		}

		if (position.y > 1.0) {
			position.y = mod(position.y + 1.0, 2.0) - 1.0;
			tile.y += 1.0;
		} else if (position.y < -1.0) {
			position.y = mod(position.y + 1.0, 2.0) - 1.0;
			tile.y -= 1.0;
		}

		tile = mod(tile, N_TILES);
	}
`,Ze=o`
	const float PARTICLE_DISTANCE_PER_MS = 0.05;
	const float FIREFLY_PHASE_PER_MS = 0.0005;
	const float FIREFLY_BRIGHTNESS_DECAY_PER_MS = 0.0001;
`,Je=o`
	uniform ivec2 particle_data_dimensions;
	uniform sampler2D particle_physics1_read_texture;
	uniform sampler2D particle_physics2_read_texture;
	uniform sampler2D particle_body_read_texture;
	uniform sampler2D particle_firefly_read_texture;
`,Qe=o`
	vec2 particleXYFromId(int id) {
		int x = id % particle_data_dimensions.x;
		int y = id / particle_data_dimensions.x;
		return vec2(x, y);
	}

	int particleIdFromXY(vec2 xy) {
		return int(xy.x) + int(xy.y) * particle_data_dimensions.x;
	}

	int particleIdFromFragCoord(vec4 FragCoord) {
		return particleIdFromXY(FragCoord.xy - 0.5);
	}

	Particle readParticle(vec2 uv) {
		vec4 physics1 = texture(particle_physics1_read_texture, uv);
		vec4 physics2 = texture(particle_physics2_read_texture, uv);
		vec4 body = texture(particle_body_read_texture, uv);
		vec4 firefly = texture(particle_firefly_read_texture, uv);

		int state = int(body.x);

		float width = body.y;
		float height = body.z;
		float agility = body.w;

		vec2 position = physics1.xy;
		vec2 tile = physics1.zw;
		vec2 velocity = physics2.xy;
		float current_speed = length(velocity);

		float phase = firefly.x;
		float brightness = firefly.y;

		return Particle(state, width, height, agility, position, tile, velocity, current_speed, phase, brightness);
	}

	Particle getParticleByFragCoord(vec4 FragCoord) {
		vec2 uv = FragCoord.xy / vec2(particle_data_dimensions);
		return readParticle(uv);
	}

	Particle getParticleByXY(vec2 xy) {
		vec2 uv = (xy+0.5) / vec2(particle_data_dimensions);
		return readParticle(uv);
	}

	Particle getParticleById(int id) {
		vec2 xy = particleXYFromId(id);
		return getParticleByXY(xy);
	}
`,me=([t,e,l,_]=[0,1,2,3])=>o`
	layout(location = ${t.toString()}) out vec4 particle_physics1_write_texture;
	layout(location = ${e.toString()}) out vec4 particle_physics2_write_texture;
	layout(location = ${l.toString()}) out vec4 particle_body_write_texture;
	layout(location = ${_.toString()}) out vec4 particle_firefly_write_texture;
	void writeParticle(Particle particle) {
		particle_physics1_write_texture = vec4(particle.position, particle.tile);
		particle_physics2_write_texture = vec4(particle.velocity, 0.0, 0.0);
		particle_body_write_texture = vec4(particle.state, particle.width, particle.height, particle.agility);
		particle_firefly_write_texture = vec4(particle.phase, particle.brightness, 0.0, 0.0);
	}
`,H=o`
	${je}
	${Ze}
	${Je}
	${Qe}
	${Ke}
`;var et=o`#version 300 es

precision highp float;

${S}

${H}
${me()}

${Y}

void main() {
	int id = particleIdFromXY(gl_FragCoord.xy);

	int state = 1;

	float size = randomGaussian(gl_FragCoord.xy + 0.00654, 0.004, 0.0008, 0.0);
	float width = size;//randomGaussian(gl_FragCoord.xy + 0.00123, , , -50.0);
	float height = size;//randomGaussian(gl_FragCoord.xy + 0.00945, , , -0.1);
	float agility = randomGaussian(gl_FragCoord.xy + 0.00654, 1.0, 0.5, 0.0);

	vec2 position = vec2(random(gl_FragCoord.xy + 0.00033), random(gl_FragCoord.xy + 0.00081));
	vec2 tile = vec2(floor(random(gl_FragCoord.xy + 0.00683) * N_TILES), floor(random(gl_FragCoord.xy + 0.00182) * N_TILES));
	vec2 velocity = normalize(randomVector(gl_FragCoord.xy + 0.00219)) * agility;//normalize(randomVector(gl_FragCoord.xy + 0.00219)) * agility;
	float current_speed = length(velocity);

	vec2 particle_world_position = getParticleWorldPosition(tile, position);
	float phase = snoise(particle_world_position * 2.);
	float brightness = 0.0;

	writeParticle(Particle(state, width, height, agility, position, tile, velocity, current_speed, phase, brightness));
}
`;const tt=o`
	struct FireflyDeposit {
		vec2 location;
		float total_brightness;
		int n_particles;
	};
`,rt=o`
	uniform ivec2 firefly_deposit_texture_dimensions;
	uniform sampler2D firefly_deposit_texture;
`,it=o`
	FireflyDeposit readFireflyDeposit(vec2 uv) {
		vec4 firefly_deposit = texture(firefly_deposit_texture, uv);

		vec2 location = uv * vec2(firefly_deposit_texture_dimensions);
		float total_brightness = firefly_deposit.x;
		int n_particles = int(firefly_deposit.w);

		return FireflyDeposit(location, total_brightness, n_particles);
	}

	FireflyDeposit getFireflyDepositByLocation(vec2 location) {
		vec2 uv = location / vec2(firefly_deposit_texture_dimensions);
		return readFireflyDeposit(uv);
	}
`,ot=o`
	${tt}
	${rt}
	${it}
`;var at=o`#version 300 es

precision highp float;

${S}
${We}

${H}
${me()}

${ot}

${ke}
${Y}

void main() {
	int id = particleIdFromFragCoord(gl_FragCoord);
	Particle particle = getParticleByFragCoord(gl_FragCoord);
	vec2 particle_world_position = getParticleWorldPosition(particle.tile, particle.position);

	vec2 aspect = vec2(aspect_ratio, -1.0);
	vec2 normal_velocity = normalize(particle.velocity);
	vec2 desired_velocity = particle.agility * normal_velocity;

	vec2 world_uv = vec2((particle_world_position.x + aspect_ratio) / (aspect_ratio*2.0), (particle_world_position.y + 1.0) / 2.0) * vec2(1.0, -1.0);
	vec2 texture_xy = world_uv * resolution; // We need to convert to XY coords to get the closest value
	vec2 closest_xy = floor(texture_xy) + 0.5;
	vec2 closest_uv = closest_xy / resolution;
	FireflyDeposit firefly_deposit = readFireflyDeposit(closest_uv);

	// // Lookup all the neighboring deposits
	int range = 1;
	FireflyDeposit current_deposit;

	int total_neighbors = 0;
	float total_neighborhood_brightness = 0.0;
	float max_brightness = 0.0;
	vec2 brightest_force = vec2(0.0, 0.0);

	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
			if (x == 0 && y == 0) current_deposit = firefly_deposit;
			else current_deposit = getFireflyDepositByLocation(firefly_deposit.location + vec2(x, y));

			total_neighbors += current_deposit.n_particles;
			if (x == 0 && y == 0) total_neighbors -= 1;

			total_neighborhood_brightness += current_deposit.total_brightness;
			if (x == 0 && y == 0) total_neighborhood_brightness -= particle.brightness;

			if (current_deposit.total_brightness > max_brightness) {
				max_brightness = current_deposit.total_brightness;
				vec2 brightest_uv = current_deposit.location / vec2(firefly_deposit_texture_dimensions);
				vec2 brightest_world_position = vec2(brightest_uv.x * (2.0 * aspect_ratio) - aspect_ratio, -brightest_uv.y * 2.0 - 1.0);

				brightest_force = brightest_world_position - particle_world_position;
			}
		}
	}

	if (total_neighborhood_brightness > 0.9 && particle.phase > 0.5) {
		particle.phase += random(gl_FragCoord.xy + time + 0.0652) * 0.05;
	}

	particle.brightness = max(0.0, particle.brightness - 0.01);
	// if(firefly_deposit.n_particles == 1) particle.state = 2;
	// if(particle.state == 2) particle.brightness = 0.9;

	bool in_mouse = false;
	if (pointer_down) {
		vec2 pointer_xy = (pointer_state.xy / resolution * aspect * 2.0) - aspect;
		if (length(pointer_xy - particle_world_position) < 0.15 * min(aspect_ratio, 1.0)) {
			in_mouse = true;
			particle.phase = 0.0;
			particle.brightness = 0.9;
		}
	}

	if (!in_mouse) {
		particle.phase += 0.002 + random(gl_FragCoord.xy * 0.001) * 0.001;
		if (particle.phase > 1.0) {
			particle.phase = mod(particle.phase, 1.0);
			particle.brightness = 1.0;
		}
	}

	// vec2 drift_force = normalize(randomVector(vec2(1.3, 0.2)));
	vec2 drift_force = vec2(1.3, 0.2);
	desired_velocity += drift_force * random(gl_FragCoord.xy + time * 0.00123) * 0.008;

	vec2 wander_force = randomVector(gl_FragCoord.xy + time * 0.00416);
	desired_velocity += wander_force * random(gl_FragCoord.xy * 0.00123) * 0.05;

	desired_velocity += brightest_force * smoothstep(0.0, 5.0, max_brightness) * 5.0 * ((0.75 + sin(time * 0.0001)) / 2.);

	vec2 steering_force = desired_velocity - particle.velocity;
	particle.velocity += steering_force;

	particle.position += particle.velocity * (PARTICLE_DISTANCE_PER_MS * dt);
	getParticleNextTilePosition(particle.tile, particle.position);

	writeParticle(particle);
}
`,st=o`#version 300 es

precision highp float;

${S}
${ye}
${Y}
in vec4 v_color;
in vec2 v_texcoord;
in float random_offset;
in mat2 rotation;
out vec4 out_color;

mat2 createRotationMatrix(float angle) {
  float s = sin(angle), c = cos(angle);
  return mat2(
    c, -s,
    s, c
  );
}

float createFirefly(vec2 uv, float flare) {
  uv *= rotation;

  float d = length(uv);
  float m = smoothstep(0.5, 1.0, 0.05 / d);

  float rays = max(0.0, 1.0 - abs(uv.x * uv.y * 100.0));
  m += rays * flare;

  m *= smoothstep(0.5, 0.2, d);
  return m;
}

void main() {
  // float firefly = createFirefly(v_texcoord * 2. - 1., (v_color.a - 0.5) / 2.0);

  out_color = v_color; //vec4(v_color.rgb * firefly, v_color.a);
}
`,nt=o`#version 300 es

precision highp float;

${S}

in vec4 a_position;
in vec2 a_texcoord;
out vec4 v_color;
out vec2 v_texcoord;
out float random_offset;
out mat2 rotation;

uniform float interpolation;
uniform mat4 projection;
uniform mat4 view;

${H}

${Y}
${ye}

float smoothmin(float a, float b, float smoothing) {
	float transition = pow(max(smoothing - abs(a - b), 0.0), 3.0) / 2.0 * pow(smoothing, 2.0);
	return min(a, b) - transition;
}

float getBrightness(float b) {
	float x = sqrt(b * 1.03);
	float y = 1.0/exp((x-0.97)*200.0);
	return min(x, y);
}

void main() {
	int id = gl_InstanceID;
	Particle particle = getParticleById(id);

	float scale = 0.2 + min(aspect_ratio, 0.8);
	mat4 scale_matrix = mat4(
		scale * particle.width, 0,											 0, 0,
		0, 											scale * particle.height, 0, 0,
		0, 										  0,			  							 1, 0,
		0, 											0, 											 0, 1
	);

	// Direction
	vec2 a = normalize(particle.velocity);
	vec2 b = normalize(vec2(0, 1));
	mat4 direction_matrix = mat4(
		a.x * b.x + a.y * b.y, b.x * a.y - a.x * b.y, 0, 0,
		a.x * b.y - b.x * a.y, a.x * b.x + a.y * b.y, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	);

	// Particle position
	vec2 interpolated_position = getParticleWorldPosition(particle.tile, particle.position + (particle.velocity * (PARTICLE_DISTANCE_PER_MS * dt)) * interpolation);
	mat4 position_matrix = mat4(
		1, 0,					 0, 0,
		0, 1,					 0, 0,
		0, 0,					 1, 0,
		interpolated_position, 0, 1
	);

  gl_Position = projection * view * position_matrix * direction_matrix * scale_matrix * a_position;

	vec3 c = vec3(1.0, 1.0, 1.0);
	if (particle.state == 1) c = vec3(0.8, 0.2, 0.9);
	if (particle.state == 2) c = vec3(1.0, 0.8, 0.0);
	if (particle.state == 3) c = vec3(0.9, 0.4, 1.0);

	float next_brightness = max(0.0, particle.brightness - (FIREFLY_BRIGHTNESS_DECAY_PER_MS * dt));//particle.brightness * sqrt(particle.brightness * 0.8);
	float interpolated_brightness = mix(particle.brightness, next_brightness, interpolation);

	float animate_brightness = getBrightness(interpolated_brightness);
	float alpha = 0.2 + (animate_brightness / 1.125);

	c = hsv2rgb(vec3(0.75 + ((animate_brightness) / 8.), smoothstep(0.95, 0.0, pow(animate_brightness, 3.0)), 0.5 + (animate_brightness / 2.)));

  // Pass the vertex color to the fragment shader.
	v_color = vec4(c, sqrt(alpha));
	v_texcoord = a_texcoord;
	random_offset = float(id);
	rotation = mat2(
		a.x * b.x + a.y * b.y, b.x * a.y - a.x * b.y,
		a.x * b.y - b.x * a.y, a.x * b.x + a.y * b.y
	);
}
`,lt=o`#version 300 es

precision highp float;

in vec4 deposit_data;
// uniform vec2 deposit_texture_resolution;
// uniform sampler2D firefly_deposit_texture;
layout(location = 0) out vec4 deposit_data_texture;

void main() {
  // vec2 uv = gl_FragCoord.xy / vec2(deposit_texture_resolution);
  // vec4 deposit_data = texture(firefly_deposit_texture, uv);
  deposit_data_texture += deposit_data;
}
`,ct=o`#version 300 es

precision highp float;

${S}

out vec4 deposit_data;

uniform mat4 projection;
uniform mat4 view;
uniform vec2 deposit_texture_resolution;

${H}

void main() {
	int id = gl_VertexID;
	Particle particle = getParticleById(id);
	vec2 particle_world_position = getParticleWorldPosition(particle.tile, particle.position);

	vec2 ratio = deposit_texture_resolution / resolution;

	vec2 world_uv = vec2((particle_world_position.x + aspect_ratio) / (aspect_ratio*2.0), (particle_world_position.y + 1.0) / 2.0) * vec2(1.0, -1.0);
	vec2 texture_xy = world_uv * resolution; // We need to convert to XY coords to get the closest value
	vec2 closest_xy = floor(texture_xy) + 0.5;
	vec2 closest_uv = closest_xy / resolution;

  gl_Position = vec4((closest_uv * 2.0) - vec2(1.0, -1.0), 0, 1);//projection * view * vec4(0.0, 0.0, 0, 1);
	gl_PointSize = 1.0;

	deposit_data = vec4(particle.brightness > 0.88 && particle.brightness < 0.92 ? 1.0 : 0.0, 0.1, 0.0, 1.0);
}
`,_t=o`#version 300 es

precision highp float;

uniform vec2 resolution;
uniform sampler2D input_texture;
layout(location = 0) out vec4 color;

void main() {
	vec2 uv = vec2((gl_FragCoord.x-0.5) / (resolution.x-1.0), gl_FragCoord.y / -resolution.y);
	color = vec4(texture(input_texture, uv).rgb, 0.5);
}
`,de=typeof Float32Array!="undefined"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var t=0,e=arguments.length;e--;)t+=arguments[e]*arguments[e];return Math.sqrt(t)});function z(){var t=new de(16);return de!=Float32Array&&(t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0),t[0]=1,t[5]=1,t[10]=1,t[15]=1,t}function pe(t){return t[0]=1,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=1,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=1,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function ut(t,e,l,_,d,v,u){var g=1/(e-l),T=1/(_-d),p=1/(v-u);return t[0]=-2*g,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=-2*T,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=2*p,t[11]=0,t[12]=(e+l)*g,t[13]=(d+_)*T,t[14]=(u+v)*p,t[15]=1,t}var fe=ut;function vt(t){const e=n();function l(){e.value&&De(t,e.value,t.TRIANGLE_FAN,0,3)}const _=(()=>{let{program:d,createUniform:v,setResolution:u}=ee(t,_t);return v("1i","input_texture")(0),function(p,b,V){t.useProgram(d),t.bindFramebuffer(t.FRAMEBUFFER,null),t.viewport(0,0,b,V),u([b,V]),c(t,t.TEXTURE0,p),l()}})();return{reset(){e.value=Q(t,new Float32Array([-1,3,3,-1,-1,-1]))},destroy(){xe(t)},render(d,v,u){_(d,v,u)}}}function dt(t){const e=t.getContext("webgl2",{premultipliedAlpha:!1});if(!e)throw new Error("Unable to get WebGL context.");e.getExtension("EXT_color_buffer_float"),e.getExtension("EXT_float_blend");const l=window.devicePixelRatio||1,_=n(0),d=n(.001+Math.random()*.01),v=n(0),u=n(1),g=n(1),T=C(()=>[u.value,g.value]),p=C(()=>u.value/g.value);function b(){const i=t.getBoundingClientRect();u.value=t.width=Math.max(1,i.width*l),g.value=t.height=Math.max(1,i.height*l)}new ResizeObserver(b).observe(t),b();function G(i){f(v,i("1f","dt")),f(_,i("1f","time")),f(d,i("1f","seed")),f(T,i("2fv","resolution")),f(p,i("1f","aspect_ratio"))}function te(){e.drawArrays(e.TRIANGLE_FAN,0,3)}const s=$e({down:!1,x:0,y:0,last_x:0,last_y:0});function re(i){if(!s.down){i.preventDefault();const r=t.getBoundingClientRect();s.x=i.pageX-r.x,s.y=i.pageY-r.y,s.last_x=i.pageX-r.x,s.last_y=i.pageY-r.y,s.down=!0,document.addEventListener("pointermove",q),document.addEventListener("pointerup",k)}}function q(i){if(s.down){const r=t.getBoundingClientRect();s.last_x=s.x,s.last_y=s.y,s.x=i.pageX-r.x,s.y=i.pageY-r.y}}function k(){s.down&&(s.down=!1,document.removeEventListener("pointermove",q),document.removeEventListener("pointerup",k))}const ge=6,he=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),ie=n(),Te=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),be=Q(e,Te),W=2e4,a=C(()=>Me(W)||[1,1]);function j(i){f(a,i("2iv","particle_data_dimensions"))}const P=n(),U=n(),F=n(),L=n(),w=n(),B=n(),R=n(),O=n(),x=n(),Pe=n(12),E=C(()=>T.value.map(i=>Math.ceil(i/Pe.value))),$=n(),oe=n(),ae=vt(e),Fe=(()=>{let{program:i,createUniform:r}=ee(e,et);return G(r),function(){e.useProgram(i),e.viewport(0,0,a.value[0],a.value[1]),y(e,x.value,e.COLOR_ATTACHMENT0,U.value),y(e,x.value,e.COLOR_ATTACHMENT1,L.value),y(e,x.value,e.COLOR_ATTACHMENT2,B.value),y(e,x.value,e.COLOR_ATTACHMENT3,O.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3]),te()}})(),we=(()=>{let{program:i,createUniform:r}=ee(e,at);G(r),j(r);const A=r("1i","particle_physics1_read_texture"),D=r("1i","particle_physics2_read_texture"),M=r("1i","particle_body_read_texture"),N=r("1i","particle_firefly_read_texture");return A(0),D(1),M(2),N(3),f(E,r("2iv","firefly_deposit_texture_dimensions")),r("1i","firefly_deposit_texture")(4),f(C(()=>s.down),r("1i","pointer_down")),f(C(()=>[s.x*l,s.y*l,s.last_x*l,s.last_y*l]),r("4fv","pointer_state")),function(){e.useProgram(i),e.viewport(0,0,a.value[0],a.value[1]),c(e,e.TEXTURE0,P.value),c(e,e.TEXTURE1,F.value),c(e,e.TEXTURE2,w.value),c(e,e.TEXTURE3,R.value),c(e,e.TEXTURE4,$.value),y(e,x.value,e.COLOR_ATTACHMENT0,U.value),y(e,x.value,e.COLOR_ATTACHMENT1,L.value),y(e,x.value,e.COLOR_ATTACHMENT2,B.value),y(e,x.value,e.COLOR_ATTACHMENT3,O.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3]),te()}})(),Re=(()=>{let{program:i,createUniform:r,createAttribute:A}=ce(e,ct,lt);G(r),j(r),f(E,r("2fv","deposit_texture_resolution"));const D=r("1i","particle_physics1_read_texture"),M=r("1i","particle_physics2_read_texture"),N=r("1i","particle_body_read_texture"),X=r("1i","particle_firefly_read_texture");D(0),M(1),N(2),X(3),r("1i","firefly_deposit_texture")(4);const Z=r("Matrix4fv","projection");return r("Matrix4fv","view")(!1,pe(z())),function(){e.useProgram(i),e.viewport(0,0,E.value[0],E.value[1]),e.enable(e.BLEND),e.blendEquationSeparate(e.FUNC_ADD,e.FUNC_ADD),e.blendFuncSeparate(e.ONE,e.ONE,e.ONE,e.ONE),Z(!1,fe(z(),-p.value,p.value,-1,1,-1,1)),c(e,e.TEXTURE0,P.value),c(e,e.TEXTURE1,F.value),c(e,e.TEXTURE2,w.value),c(e,e.TEXTURE3,R.value),c(e,e.TEXTURE4,$.value),y(e,oe.value,e.COLOR_ATTACHMENT0,$.value),e.drawBuffers([e.COLOR_ATTACHMENT0]),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),e.drawArrays(e.POINTS,0,W),e.disable(e.BLEND)}})(),Ee=(()=>{let{program:i,createUniform:r,createAttribute:A}=ce(e,nt,st);G(r),j(r);const D=r("1i","particle_physics1_read_texture"),M=r("1i","particle_physics2_read_texture"),N=r("1i","particle_body_read_texture"),X=r("1i","particle_firefly_read_texture");D(0),M(1),N(2),X(3);const K=r("1f","interpolation"),Z=r("Matrix4fv","projection"),ne=r("Matrix4fv","view"),J=A("a_position"),le=A("a_texcoord");return e.bindBuffer(e.ARRAY_BUFFER,be),e.enableVertexAttribArray(le),e.vertexAttribPointer(le,2,e.FLOAT,!1,0,0),ne(!1,pe(z())),function(Be=0){e.useProgram(i),e.bindFramebuffer(e.FRAMEBUFFER,null),e.viewport(0,0,u.value,g.value),K(Be),Z(!1,fe(z(),-p.value,p.value,-1,1,-1,1)),c(e,e.TEXTURE0,P.value),c(e,e.TEXTURE1,F.value),c(e,e.TEXTURE2,w.value),c(e,e.TEXTURE3,R.value),e.bindBuffer(e.ARRAY_BUFFER,ie.value),e.enableVertexAttribArray(J),e.vertexAttribPointer(J,2,e.FLOAT,!1,0,0),e.drawArraysInstanced(e.TRIANGLES,0,ge,W)}})();function se(){_e(e),ue(e),P.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),U.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),F.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),L.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),w.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),B.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),R.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),O.value=m(e,a.value[0],a.value[1],e.RGBA32F,e.RGBA,e.FLOAT),x.value=ve(e),ie.value=Q(e,he),$.value=m(e,E.value[0],E.value[1],e.RGBA32F,e.RGBA,e.FLOAT),oe.value=ve(e),Fe(),h(P,U),h(F,L),h(w,B),h(R,O),ae.reset()}function Ae(){b(),t.addEventListener("pointerdown",re),se()}function Ce(){t.removeEventListener("pointerdown",re),document.removeEventListener("pointermove",q),document.removeEventListener("pointerup",k),_e(e),xe(e),ue(e),ae.destroy();let i=e.getExtension("WEBGL_lose_context");i&&i.loseContext()}function Ue(i){v.value=i,_.value+=i,Re(),we(),h(P,U),h(F,L),h(w,B),h(R,O)}function Le(i){e.bindFramebuffer(e.FRAMEBUFFER,null),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),Ee(i)}return{init:Ae,reset:se,update:Ue,render:Le,destroy:Ce}}const mt=Xe({setup(t){const e=n();return qe(e),(l,_)=>(ze(),Ye("div",null,[He("canvas",{ref_key:"canvas",ref:e,class:"w-full h-full","touch-action":"none"},null,512)]))}});export{mt as _};
