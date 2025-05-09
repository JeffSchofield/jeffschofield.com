import{L as Ke,g as a,c as ee,a as Q,b as n,r as Ue,d as Ze,e as Je,f as u,s as f,h as be,i as Fe,j as Re,k as w,l as p,m as pe}from"./utils-BrZyIiGL.js";import{e as i,D as Qe,E as et,g as tt,M as rt,q as D,O as Ae}from"./index-Cy704NKG.js";const A=new Ke;A.update_time_step=16;function Et(t,{n_fireflies:_=3e4}={}){let e;const l=i(_);function v(){if(t.value)try{console.log("Creating firefly program"),e=Tt(t.value,{n_fireflies:l.value}),A.linkProgram(e),console.log("Firefly program created")}catch(d){console.error(d)}}return Qe(()=>et(()=>{A.init(),tt(t,m=>{e&&(A.unlinkProgram(e),e.destroy()),v()},{flush:"sync",immediate:!0})})),rt(()=>{e&&(A.unlinkProgram(e),e.destroy())}),{setFireflies(d){l.value=d,e&&(e.state.n_particles.value=d,e.init())},pause(){A.cancelLoop()},play(){A.requestLoop()}}}const Y=a`
uniform float dt;
uniform float time;
uniform vec2 viewport_resolution;
uniform float viewport_aspect_ratio;
`,ot=a`
uniform bool pointer_down;
uniform vec4 pointer_state;
`,it=a`
mat2 rotationMatrix(float a) {
	return mat2(cos(a), -sin(a), sin(a), cos(a));
}
`,te=a`
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
`,at=a`
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
	vec2 desired_velocity;
	float current_speed; // computed
	float wander_theta;

	// Firefly
	float phase;
	float brightness;
};
`,st=a`
const float PARTICLE_DISTANCE_PER_MS = 0.05;
const float FIREFLY_PHASE_PER_MS = 0.0005;
const float FIREFLY_BRIGHTNESS_DECAY_PER_MS = 0.0001;
`,nt=a`
uniform ivec2 particle_data_dimensions;
uniform sampler2D particle_physics1_read_texture;
uniform sampler2D particle_physics2_read_texture;
uniform sampler2D particle_physics3_read_texture;
uniform sampler2D particle_body_read_texture;
uniform sampler2D particle_firefly_read_texture;
`,lt=a`
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
	vec4 physics3 = texture(particle_physics3_read_texture, uv);
	vec4 body = texture(particle_body_read_texture, uv);
	vec4 firefly = texture(particle_firefly_read_texture, uv);

	int state = int(body.x);

	float width = body.y;
	float height = body.z;
	float agility = body.w;

	vec2 position = physics1.xy;
	vec2 tile = physics1.zw;
	vec2 velocity = physics2.xy;
	vec2 desired_velocity = physics2.zw;
	float current_speed = length(velocity);
	float wander_theta = physics3.x;

	float phase = firefly.x;
	float brightness = firefly.y;

	return Particle(state, width, height, agility, position, tile, velocity, desired_velocity, current_speed, wander_theta, phase, brightness);
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
`,Le=([t,_,e,l,v]=[0,1,2,3,4])=>a`
layout(location = ${t.toString()}) out vec4 particle_physics1_write_texture;
layout(location = ${_.toString()}) out vec4 particle_physics2_write_texture;
layout(location = ${e.toString()}) out vec4 particle_physics3_write_texture;
layout(location = ${l.toString()}) out vec4 particle_body_write_texture;
layout(location = ${v.toString()}) out vec4 particle_firefly_write_texture;
void writeParticle(Particle particle) {
	particle_physics1_write_texture = vec4(particle.position, particle.tile);
	particle_physics2_write_texture = vec4(particle.velocity, particle.desired_velocity);
	particle_physics3_write_texture = vec4(particle.wander_theta, 0.0, 0.0, 0.0);
	particle_body_write_texture = vec4(particle.state, particle.width, particle.height, particle.agility);
	particle_firefly_write_texture = vec4(particle.phase, particle.brightness, 0.0, 0.0);
}
`,re=a`
${at}
${st}
${nt}
${lt}
`,oe=a`
uniform vec2 world_resolution;
uniform float world_aspect_ratio;
uniform sampler2D world_level_texture;
`,ie=a`
const float N_TILES = 1000.0;
vec2 getUVFromWorldCoords(vec2 position) { // vec2(0.0, 0.0);
	return vec2(
		(position.x + world_aspect_ratio) / (world_aspect_ratio*2.0), // (1.777) / 3.55 = 0.5
		(position.y + 1.0) / 2.0 // 1. / 2. = 0.5
	);
}

vec2 getWorldCoordsFromUV(vec2 uv) {
	return uv * vec2(world_aspect_ratio * 2.0, 2.0) - vec2(world_aspect_ratio, 1.0);
}

vec2 getWorldCoords(vec2 tile, vec2 position) { // (250, 250)
	vec2 tile_resolution = world_resolution / N_TILES; // 1.92
	vec2 tile_uv = ((tile * tile_resolution) + (tile_resolution / 2.0)) / world_resolution; // 0.0035
	vec2 tile_world_coords = getWorldCoordsFromUV(tile_uv);

	return tile_world_coords + (position / vec2(N_TILES / world_aspect_ratio, N_TILES));
}

void setNextWorldPosition(inout vec2 tile, inout vec2 position) {
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

vec4 readWorldLevel(vec2 uv) {
	return texture(world_level_texture, uv);
}
`,ct=a`#version 300 es

precision highp float;

${Y}
${oe}
${ie}

${re}
${Le()}

${te}

void main() {
	int id = particleIdFromXY(gl_FragCoord.xy);

	int state = 1;

	float size = randomGaussian(gl_FragCoord.xy + 0.00654, 0.004, 0.0008, 0.0);
	float width = size * 0.8;//randomGaussian(gl_FragCoord.xy + 0.00123, , , -50.0);
	float height = size;//randomGaussian(gl_FragCoord.xy + 0.00945, , , -0.1);
	float agility = randomGaussian(gl_FragCoord.xy + 0.00654, 0.6, 0.5, 0.0);

	vec2 position = vec2(random(gl_FragCoord.xy + 0.00033) * 2.0 - 1.0, random(gl_FragCoord.xy + 0.00081) * 2.0 - 1.0);
	vec2 tile = vec2(
		floor(random(gl_FragCoord.xy + 0.00683) * N_TILES),
		floor(random(gl_FragCoord.xy + 0.00182) * N_TILES)
	);
	vec2 velocity = normalize(randomVector(gl_FragCoord.xy + 0.00219)) * agility;//normalize(randomVector(gl_FragCoord.xy + 0.00219)) * agility;
	float current_speed = length(velocity);
	float wander_theta = 0.0;

	vec2 world_coords = getWorldCoords(tile, position);
	float phase = snoise(world_coords * 2.);
	float brightness = 0.0;

	writeParticle(Particle(state, width, height, agility, position, tile, velocity, velocity, current_speed, wander_theta, phase, brightness));
}
`,_t=a`
struct FireflyDeposit {
	vec2 xy;
	float total_brightness;
	int n_particles;
	vec2 mass_center;
	vec2 total_velocity;
};
`,dt=a`
uniform float firefly_deposit_bin_divisor;
uniform ivec2 firefly_deposit_texture_dimensions;
uniform sampler2D firefly_deposit_texture;
uniform sampler2D firefly_deposit2_texture;
`,vt=a`
vec2 fireflyDepositXYFromUV(vec2 uv) {
	return floor(uv * vec2(firefly_deposit_texture_dimensions));
}
vec2 fireflyDepositUVFromXY(vec2 xy) {
	return (xy+0.5) / vec2(firefly_deposit_texture_dimensions);
}

FireflyDeposit readFireflyDeposit(vec2 uv) {
	vec4 firefly_deposit = texture(firefly_deposit_texture, uv);
	vec4 firefly_deposit2 = texture(firefly_deposit2_texture, uv);

	vec2 xy = fireflyDepositXYFromUV(uv);
	float total_brightness = firefly_deposit.x;
	int n_particles = int(firefly_deposit.w);
	vec2 mass_center = firefly_deposit2.xy;
	vec2 total_velocity = firefly_deposit2.zw;

	return FireflyDeposit(xy, total_brightness, n_particles, mass_center, total_velocity);
}

FireflyDeposit getFireflyDepositByXY(vec2 xy) {
	vec2 uv = fireflyDepositUVFromXY(xy);
	return readFireflyDeposit(uv);
}
`,Oe=a`
${_t}
${dt}
${vt}
`,ut=a`#version 300 es

precision highp float;

${Y}
${oe}
${ie}
${ot}

${re}
${Le()}

${Oe}

${it}
${te}

void main() {
	Particle particle = getParticleByFragCoord(gl_FragCoord);
	vec2 particle_world_coords = getWorldCoords(particle.tile, particle.position);

	vec2 world_uv = getUVFromWorldCoords(particle_world_coords);
	FireflyDeposit firefly_deposit = readFireflyDeposit(world_uv);
	vec2 deposit_center_uv = fireflyDepositUVFromXY(firefly_deposit.xy);
	vec2 deposit_center_world_coords = getWorldCoordsFromUV(deposit_center_uv);
	vec2 own_distance_to_center = particle_world_coords - deposit_center_world_coords;

	int range = 1;
	FireflyDeposit current_deposit;

	int total_neighbors = 0;
	float total_deposits = 0.0;

	float total_neighborhood_brightness = 0.0;
	float max_brightness = 0.0;
	vec2 brightest_force = vec2(0.0, 0.0);
	vec2 separation_force = vec2(0.0, 0.0);
	// vec2 cohesion_force = vec2(0.0, 0.0);
	vec2 alignment_force = vec2(0.0, 0.0);

	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
			if (x == 0 && y == 0) current_deposit = firefly_deposit;
			else current_deposit = getFireflyDepositByXY(firefly_deposit.xy + vec2(x, y));

			int n_particles = current_deposit.n_particles;
			if (x == 0 && y == 0) n_particles -= 1; // Don't include self in particle count

			if (n_particles > 0) {
				total_deposits += 1.0;
				total_neighbors += n_particles;
				float n_particles_float = float(n_particles);

				total_neighborhood_brightness += current_deposit.total_brightness;
				if (x == 0 && y == 0) total_neighborhood_brightness -= particle.brightness; // Don't include self on center tile

				vec2 deposit_center_uv = fireflyDepositUVFromXY(current_deposit.xy);
				vec2 deposit_center_world_coords = getWorldCoordsFromUV(deposit_center_uv);
				vec2 avg_mass_center = ((x == 0 && y == 0) ? (current_deposit.mass_center - own_distance_to_center) : current_deposit.mass_center) / n_particles_float;
				vec2 deposit_mass_coords = deposit_center_world_coords + avg_mass_center;

				if (current_deposit.total_brightness > max_brightness) {
					max_brightness = current_deposit.total_brightness;
					brightest_force = deposit_center_world_coords - particle_world_coords;
				}

				alignment_force += current_deposit.total_velocity / n_particles_float;

				vec2 separation_direction = particle_world_coords - deposit_mass_coords;
				if (length(separation_direction) < 0.05) {
					separation_force += separation_direction;
				}

			}
		}
	}

	// Influence brightness
	if (total_neighborhood_brightness > 0.0 && particle.phase > 0.5) {
		// particle.phase += random(gl_FragCoord.xy + time + 0.00652) * 0.101 * pow(total_neighborhood_brightness * 0.1, 2.2);
		particle.phase = 1.0;
	}

	// Decay brightness
	particle.brightness = max(0.0, particle.brightness - 0.01);

	// Mouse interaction
	bool in_mouse = false;
	vec2 aspect = vec2(world_aspect_ratio, 1.0);
	if (pointer_down) {
		vec2 pointer_xy = (pointer_state.xy / world_resolution * aspect * 2.0) - aspect;
		if (length(pointer_xy - particle_world_coords) < 0.15 * min(world_aspect_ratio, 1.0)) {
			in_mouse = true;
			particle.phase = 0.0;
			particle.brightness = 0.9;
		}
	}

	// Progress oscillator + flash
	if (!in_mouse) {
		particle.phase += 0.002 + random(gl_FragCoord.xy * 0.001) * 0.0001;
		if (particle.phase > 1.0) {
			particle.phase = mod(particle.phase, 1.0);
			particle.brightness = 1.0;
		}
	}

	// Build Forces
	vec2 desired_velocity = particle.desired_velocity;
	float n_desired_forces = 0.0;

	// Debug - To Center Deposit
	// vec2 deposit_center_uv = fireflyDepositUVFromXY(firefly_deposit.xy);
	// vec2 deposit_center_world_coords = getWorldCoordsFromUV(deposit_center_uv);

	// desired_velocity += (deposit_center_world_coords - particle_world_coords) * 5.;
	// n_desired_forces += 1.0;

	// Wandering around
	float wander_radius = 10.0;
	vec2 wander_point = particle_world_coords + normalize(particle.velocity) * 100.0;
	float theta = particle.wander_theta + atan(particle.velocity.y, particle.velocity.x);
	wander_point += vec2(
		cos(theta) * wander_radius,
		sin(theta) * wander_radius
	);

	float wander_displacement = 0.0125;
	particle.wander_theta += random(gl_FragCoord.xy + time + 0.00782) * (wander_displacement*2.0) - wander_displacement;

	desired_velocity += normalize(wander_point - particle_world_coords) * particle.agility * (sin(time * 10.) * 0.25);
	n_desired_forces += 1.0;

	// Attract to brightness
	desired_velocity += brightest_force * smoothstep(0.0, 5.0, max_brightness) * 10. * particle.agility;
	n_desired_forces += 1.0;

	// Align with neighbors
	if (alignment_force != vec2(0.0, 0.0)) {
		alignment_force /= total_deposits;
		desired_velocity += normalize(alignment_force) * 0.75;
		n_desired_forces += 1.0;
	}

	// Separate from neighbors
	if (separation_force != vec2(0.0, 0.0)) {
		separation_force /= total_deposits;
		desired_velocity += separation_force * 0.5;
		n_desired_forces += 1.0;
	}

	if (n_desired_forces > 0.0) {
		desired_velocity += normalize(desired_velocity / n_desired_forces) * particle.agility * n_desired_forces;

		particle.desired_velocity = desired_velocity / (n_desired_forces + 1.0);
	}

	vec2 steering_force = particle.desired_velocity - particle.velocity;
	particle.velocity += steering_force;

	particle.position += particle.velocity * (PARTICLE_DISTANCE_PER_MS * dt);
	setNextWorldPosition(particle.tile, particle.position);

	writeParticle(particle);
}
`,De=a`
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
`,pt=a`#version 300 es

precision highp float;

${Y}
${De}
${te}
in vec4 v_color;
in vec2 v_texcoord;
in float random_offset;
in mat2 rotation;
out vec4 out_color;

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

  // out_color = vec4(v_color.rgb * firefly, v_color.a);
  out_color = v_color;
}
`,ft=a`#version 300 es

precision highp float;

${Y}
${oe}
${ie}

in vec4 a_position;
in vec2 a_texcoord;
out vec4 v_color;
out vec2 v_texcoord;
out float random_offset;
out mat2 rotation;

uniform float interpolation;
uniform mat4 projection;
uniform mat4 view;

${re}

${te}
${De}

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

	// Size
	float scale = 0.5;
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

	// Particle world position
	vec2 interpolated_coords = getWorldCoords(particle.tile, particle.position + (particle.velocity * (PARTICLE_DISTANCE_PER_MS * dt)) * interpolation) * vec2(1.0, -1.0);
	mat4 position_matrix = mat4(
		1, 0,					 0, 0,
		0, 1,					 0, 0,
		0, 0,					 1, 0,
		interpolated_coords, 0, 1
	);

	mat4 world_matrix = position_matrix * direction_matrix * scale_matrix;

  gl_Position = projection * view * world_matrix * a_position;

	vec3 c = vec3(1.0, 1.0, 1.0);
	if (particle.state == 1) c = vec3(0.8, 0.2, 0.9);
	if (particle.state == 2) c = vec3(1.0, 0.8, 0.0);
	if (particle.state == 3) c = vec3(0.9, 0.4, 1.0);

	float next_brightness = max(0.0, particle.brightness - (FIREFLY_BRIGHTNESS_DECAY_PER_MS * dt));
	float interpolated_brightness = mix(particle.brightness, next_brightness, interpolation);

	float animate_brightness = getBrightness(interpolated_brightness);
	float alpha = 0.2 + (animate_brightness / 1.125);

	c = hsv2rgb(vec3(0.75 + ((animate_brightness) / 8.), smoothstep(0.95, 0.0, pow(animate_brightness, 3.0)), 0.5 + (animate_brightness / 2.)));
	// c = vec3(1.0);

  // Pass the vertex color to the fragment shader.
	v_color = vec4(c, sqrt(alpha));
	v_texcoord = a_texcoord;
	random_offset = float(id);
	rotation = mat2(
		a.x * b.x + a.y * b.y, b.x * a.y - a.x * b.y,
		a.x * b.y - b.x * a.y, a.x * b.x + a.y * b.y
	);
}
`,yt=a`#version 300 es

precision highp float;

in vec4 deposit_data;
in vec4 deposit2_data;
layout(location = 0) out vec4 deposit_data_texture;
layout(location = 1) out vec4 deposit2_data_texture;

void main() {
  deposit_data_texture += deposit_data;
  deposit2_data_texture += deposit2_data;
}
`,xt=a`#version 300 es

precision highp float;

${Y}
${oe}
${ie}

in vec4 a_dummy;
out vec4 deposit_data;
out vec4 deposit2_data;

uniform mat4 projection;
uniform mat4 view;

${re}
${Oe}

void main() {
	int id = gl_VertexID;
	Particle particle = getParticleById(id);
	vec2 particle_world_coords = getWorldCoords(particle.tile, particle.position);

	vec2 world_uv = getUVFromWorldCoords(particle_world_coords);
  gl_Position = vec4((world_uv * vec2(2.0, 2.0)) - vec2(1.0, 1.0), 0, 1);
	gl_PointSize = 1.0;

	vec2 deposit_xy = fireflyDepositXYFromUV(world_uv);
	vec2 deposit_uv = fireflyDepositUVFromXY(deposit_xy);
	vec2 deposit_world_coords = getWorldCoordsFromUV(deposit_uv);
	vec2 distance_to_deposit_center = particle_world_coords - deposit_world_coords;

	float c = 0.0;
	if (particle.state != 1) c = 1.0;

	deposit_data = vec4(particle.brightness > 0.88 && particle.brightness < 0.92 ? 1.0 : 0.0, 0.1, c, 1.0);
	deposit2_data = vec4(distance_to_deposit_center, particle.velocity);
}
`,mt=a`#version 300 es

precision highp float;

uniform float world_aspect_ratio;
uniform vec2 world_resolution;
uniform vec4 nav_base_dimensions;
layout(location = 0) out vec4 level_texture;


float sdRoundBox( in vec2 p, in vec2 b, in vec4 r )
{
    r.xy = (p.x>0.0)?r.xy : r.zw;
    r.x  = (p.y>0.0)?r.x  : r.y;
    vec2 q = abs(p)-b+r.x;
    return min(max(q.x,q.y),0.0) + length(max(q,0.0)) - r.x;
}

void main() {
	vec3 c = vec3(0.0);
	vec2 p = (2.0*gl_FragCoord.xy-world_resolution.xy)/world_resolution.y;

	vec2 pos = (((nav_base_dimensions.xy+nav_base_dimensions.zw/2.0)/world_resolution.xy) * 2.0 - 1.0) * vec2(world_aspect_ratio, 1.0); //vec2(((nav_base_dimensions.x/world_resolution.x)*2.0 - 1.0) * world_aspect_ratio, 0.0);

	c = vec3(1.0) * ((sdRoundBox(p - pos, nav_base_dimensions.zw/world_resolution.xy*vec2(world_aspect_ratio, 1.0), vec4(0.0375)) > 0.0) ? 0.0 : 1.0);
	level_texture = vec4(c, 1.0);
}
`;a`#version 300 es

precision highp float;

uniform vec2 resolution;
uniform sampler2D input_texture;
layout(location = 0) out vec4 color;

void main() {
	vec2 uv = vec2((gl_FragCoord.x-0.5) / (resolution.x-1.0), gl_FragCoord.y / -resolution.y);
	color = vec4(texture(input_texture, uv).rgb, 0.5);
}
`;var Ee=typeof Float32Array<"u"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var t=0,_=arguments.length;_--;)t+=arguments[_]*arguments[_];return Math.sqrt(t)});function J(){var t=new Ee(16);return Ee!=Float32Array&&(t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0),t[0]=1,t[5]=1,t[10]=1,t[15]=1,t}function Pe(t){return t[0]=1,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=1,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=1,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function gt(t,_,e,l,v,d,m){var g=1/(_-e),h=1/(l-v),b=1/(d-m);return t[0]=-2*g,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=-2*h,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=2*b,t[11]=0,t[12]=(_+e)*g,t[13]=(v+l)*h,t[14]=(m+d)*b,t[15]=1,t}var Ce=gt;const ht=a`#version 300 es

precision highp float;

uniform vec2 resolution;
uniform sampler2D input_texture;
layout(location = 0) out vec4 color;

void main() {
	vec2 uv = vec2((gl_FragCoord.x-0.5) / (resolution.x-1.0), gl_FragCoord.y / -resolution.y);
	color = vec4(texture(input_texture, uv).rgb, 0.5);
}
`;function wt(t){const _=i(ee(t,new Float32Array([-1,3,3,-1,-1,-1])));function e(){_.value&&Ze(t,_.value,t.TRIANGLE_FAN,0,3)}const l=(()=>{let{program:v,createUniform:d,setResolution:m}=Q(t,ht);return d("1i","input_texture")(0),function(b,T,E){t.useProgram(v),t.bindFramebuffer(t.FRAMEBUFFER,null),t.viewport(0,0,T,E),m([T,E]),n(t,t.TEXTURE0,b),e()}})();return{reset(){_.value=ee(t,new Float32Array([-1,3,3,-1,-1,-1]))},destroy(){Ue(t)},render(v,d,m){l(v,d,m)}}}function Tt(t,{n_fireflies:_=3e4}={}){const e=t.getContext("webgl2",{premultipliedAlpha:!1});if(!e)throw new Error("Unable to get WebGL context.");if(!e.getExtension("EXT_color_buffer_float"))throw new Error("EXT_color_buffer_float not supported!");if(!e.getExtension("EXT_float_blend"))throw new Error("EXT_float_blend not supported!");const l=window.devicePixelRatio||1,v=i(0),d=i(.001+Math.random()*.01),m=i(0),g=i(1),h=i(1),b=D(()=>[g.value,h.value]),T=D(()=>g.value/h.value);function E(){const o=t.getBoundingClientRect();g.value=t.width=Math.max(1,o.width*l),h.value=t.height=Math.max(1,o.height*l)}new ResizeObserver(E).observe(t),E();function B(o){f(m,o("1f","dt")),f(v,o("1f","time")),f(d,o("1f","seed")),f(b,o("2fv","viewport_resolution")),f(T,o("1f","viewport_aspect_ratio"))}function M(o){f(b,o("2fv","world_resolution")),f(T,o("1f","world_aspect_ratio"))}function ae(){e.drawArrays(e.TRIANGLE_FAN,0,3)}const c=Ae({down:!1,x:0,y:0,last_x:0,last_y:0});function fe(o){if(!c.down){o.preventDefault();const r=t.getBoundingClientRect();c.x=o.pageX-r.x,c.y=o.pageY-r.y,c.last_x=o.pageX-r.x,c.last_y=o.pageY-r.y,c.down=!0,document.addEventListener("pointermove",se),document.addEventListener("pointerup",ne)}}function se(o){if(c.down){const r=t.getBoundingClientRect();c.last_x=c.x,c.last_y=c.y,c.x=o.pageX-r.x,c.y=o.pageY-r.y}}function ne(){c.down&&(c.down=!1,document.removeEventListener("pointermove",se),document.removeEventListener("pointerup",ne))}const Be=6,Me=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),ye=i(),Ne=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),Ge=ee(e,Ne),H=i(_),s=D(()=>Je(H.value)||[2,2]);function le(o){f(s,o("2iv","particle_data_dimensions"))}const P=i(),N=i(),C=i(),G=i(),U=i(),S=i(),L=i(),I=i(),O=i(),X=i(),y=i(),xe=i(6),R=D(()=>b.value.map(o=>Math.ceil(o/xe.value)));function me(o){f(xe,o("1f","firefly_deposit_bin_divisor")),f(R,o("2iv","firefly_deposit_texture_dimensions"))}const W=i(),q=i(),ce=i(),_e=i(),ge=i(),de=Ae([0,0,0,0]),k=document.getElementById("nav-base");function he(){if(k){const{x:o,y:r,width:x,height:F}=k.getBoundingClientRect();de.length=0,de.push(o*l,r*l,x*l,F*l)}}const Se=new ResizeObserver(he);k&&Se.observe(k),he();const Ie=wt(e),Xe=(()=>{let{program:o,createUniform:r}=Q(e,ct);return B(r),M(r),function(){e.useProgram(o),e.viewport(0,0,s.value[0],s.value[1]),u(e,y.value,e.COLOR_ATTACHMENT0,N.value),u(e,y.value,e.COLOR_ATTACHMENT1,G.value),u(e,y.value,e.COLOR_ATTACHMENT2,S.value),u(e,y.value,e.COLOR_ATTACHMENT3,I.value),u(e,y.value,e.COLOR_ATTACHMENT4,X.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3,e.COLOR_ATTACHMENT4]),ae()}})(),$e=(()=>{let{program:o,createUniform:r}=Q(e,ut);B(r),M(r),le(r);const x=r("1i","particle_physics1_read_texture"),F=r("1i","particle_physics2_read_texture"),$=r("1i","particle_physics3_read_texture"),z=r("1i","particle_body_read_texture"),V=r("1i","particle_firefly_read_texture");x(0),F(1),$(2),z(3),V(4),me(r);const j=r("1i","firefly_deposit_texture"),K=r("1i","firefly_deposit2_texture");return j(5),K(6),r("1i","world_level_texture")(7),f(D(()=>c.down),r("1i","pointer_down")),f(D(()=>[c.x*l,c.y*l,c.last_x*l,c.last_y*l]),r("4fv","pointer_state")),function(){e.useProgram(o),e.viewport(0,0,s.value[0],s.value[1]),n(e,e.TEXTURE0,P.value),n(e,e.TEXTURE1,C.value),n(e,e.TEXTURE2,U.value),n(e,e.TEXTURE3,L.value),n(e,e.TEXTURE4,O.value),n(e,e.TEXTURE5,W.value),n(e,e.TEXTURE6,q.value),n(e,e.TEXTURE7,_e.value),u(e,y.value,e.COLOR_ATTACHMENT0,N.value),u(e,y.value,e.COLOR_ATTACHMENT1,G.value),u(e,y.value,e.COLOR_ATTACHMENT2,S.value),u(e,y.value,e.COLOR_ATTACHMENT3,I.value),u(e,y.value,e.COLOR_ATTACHMENT4,X.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3,e.COLOR_ATTACHMENT4]),ae()}})(),ze=(()=>{let{program:o,createUniform:r}=be(e,xt,yt);B(r),M(r),le(r),me(r);const x=r("1i","particle_physics1_read_texture"),F=r("1i","particle_physics2_read_texture"),$=r("1i","particle_physics3_read_texture"),z=r("1i","particle_body_read_texture"),V=r("1i","particle_firefly_read_texture");x(0),F(1),$(2),z(3),V(4),r("1i","firefly_deposit_texture")(5),r("1i","firefly_deposit2_texture")(6);const Z=r("Matrix4fv","projection");return r("Matrix4fv","view")(!1,Pe(J())),function(){e.useProgram(o),e.viewport(0,0,R.value[0],R.value[1]),e.enable(e.BLEND),e.blendEquationSeparate(e.FUNC_ADD,e.FUNC_ADD),e.blendFuncSeparate(e.ONE,e.ONE,e.ONE,e.ONE),Z(!1,Ce(J(),-T.value,T.value,-1,1,-1,1)),n(e,e.TEXTURE0,P.value),n(e,e.TEXTURE1,C.value),n(e,e.TEXTURE2,U.value),n(e,e.TEXTURE3,L.value),n(e,e.TEXTURE4,O.value),n(e,e.TEXTURE5,W.value),n(e,e.TEXTURE6,q.value),u(e,ce.value,e.COLOR_ATTACHMENT0,W.value),u(e,ce.value,e.COLOR_ATTACHMENT1,q.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1]),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),e.drawArrays(e.POINTS,0,H.value),e.disable(e.BLEND)}})(),Ve=(()=>{let{program:o,createUniform:r,createAttribute:x}=be(e,ft,pt);B(r),M(r),le(r);const F=r("1i","particle_physics1_read_texture"),$=r("1i","particle_physics2_read_texture"),z=r("1i","particle_physics3_read_texture"),V=r("1i","particle_body_read_texture"),j=r("1i","particle_firefly_read_texture");F(0),$(1),z(2),V(3),j(4);const K=r("1f","interpolation"),Z=r("Matrix4fv","projection"),ve=r("Matrix4fv","view"),ue=x("a_position"),Te=x("a_texcoord");return e.bindBuffer(e.ARRAY_BUFFER,Ge),e.enableVertexAttribArray(Te),e.vertexAttribPointer(Te,2,e.FLOAT,!1,0,0),ve(!1,Pe(J())),function(je=0){e.useProgram(o),e.bindFramebuffer(e.FRAMEBUFFER,null),e.viewport(0,0,g.value,h.value),K(je),Z(!1,Ce(J(),-T.value,T.value,-1,1,-1,1)),n(e,e.TEXTURE0,P.value),n(e,e.TEXTURE1,C.value),n(e,e.TEXTURE2,U.value),n(e,e.TEXTURE3,L.value),n(e,e.TEXTURE4,O.value),e.bindBuffer(e.ARRAY_BUFFER,ye.value),e.enableVertexAttribArray(ue),e.vertexAttribPointer(ue,2,e.FLOAT,!1,0,0),e.drawArraysInstanced(e.TRIANGLES,0,Be,H.value)}})(),Ye=(()=>{let{program:o,createUniform:r}=Q(e,mt);B(r),M(r);const x=r("4fv","nav_base_dimensions");return function(){e.useProgram(o),e.viewport(0,0,g.value,h.value),x(de),u(e,ge.value,e.COLOR_ATTACHMENT0,_e.value),e.drawBuffers([e.COLOR_ATTACHMENT0]),ae()}})();function we(){Fe(e),Re(e),P.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),N.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),C.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),G.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),U.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),S.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),L.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),I.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),O.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),X.value=p(e,s.value[0],s.value[1],e.RGBA32F,e.RGBA,e.FLOAT),y.value=pe(e),ye.value=ee(e,Me),W.value=p(e,R.value[0],R.value[1],e.RGBA32F,e.RGBA,e.FLOAT),q.value=p(e,R.value[0],R.value[1],e.RGBA32F,e.RGBA,e.FLOAT),ce.value=pe(e),_e.value=p(e,g.value,h.value,e.RGBA32F,e.RGBA,e.FLOAT),ge.value=pe(e),Xe(),Ye(),w(P,N),w(C,G),w(U,S),w(L,I),w(O,X)}function He(){E(),t.addEventListener("pointerdown",fe),we()}function We(){t.removeEventListener("pointerdown",fe),document.removeEventListener("pointermove",se),document.removeEventListener("pointerup",ne),Fe(e),Ue(e),Re(e),Ie.destroy();let o=e.getExtension("WEBGL_lose_context");o&&o.loseContext()}function qe(o){m.value=o,v.value+=o,ze(),$e(),w(P,N),w(C,G),w(U,S),w(L,I),w(O,X)}function ke(o){e.bindFramebuffer(e.FRAMEBUFFER,null),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),Ve(o)}return{init:He,reset:we,update:qe,render:ke,destroy:We,state:{n_particles:H}}}export{Et as u};
