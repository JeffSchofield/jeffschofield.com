import{L as Ke,g as s,c as Z,b as n,r as Le,a as fe,d as Ze,e as Je,f as v,s as u,h as Fe,i as Re,j as Ae,k as h,l as p,m as ue}from"./utils-C0Fl8xI1.js";import{D as Qe,E as et,g as tt,M as rt,e as a,q as O,O as Pe,p as ot,c as it,o as at,a as st}from"./index-CtIPOu-Y.js";const R=new Ke;R.update_time_step=16;function nt(t){let e;function l(){if(t.value)try{console.log("Creating firefly program"),e=Pt(t.value),R.linkProgram(e),console.log("Firefly program created")}catch(_){console.error(_)}}return Qe(()=>et(()=>{R.init(),tt(t,f=>{e&&(R.unlinkProgram(e),e.destroy()),l()},{flush:"sync",immediate:!0})})),rt(()=>{e&&(R.unlinkProgram(e),e.destroy())}),{pause(){R.cancelLoop()},play(){R.requestLoop()}}}const Y=s`
uniform float dt;
uniform float time;
uniform vec2 viewport_resolution;
uniform float viewport_aspect_ratio;
`,lt=s`
uniform bool pointer_down;
uniform vec4 pointer_state;
`,ct=s`
mat2 rotationMatrix(float a) {
	return mat2(cos(a), -sin(a), sin(a), cos(a));
}
`,J=s`
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
`,_t=s`
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
`,dt=s`
const float PARTICLE_DISTANCE_PER_MS = 0.05;
const float FIREFLY_PHASE_PER_MS = 0.0005;
const float FIREFLY_BRIGHTNESS_DECAY_PER_MS = 0.0001;
`,vt=s`
uniform ivec2 particle_data_dimensions;
uniform sampler2D particle_physics1_read_texture;
uniform sampler2D particle_physics2_read_texture;
uniform sampler2D particle_physics3_read_texture;
uniform sampler2D particle_body_read_texture;
uniform sampler2D particle_firefly_read_texture;
`,pt=s`
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
`,Oe=([t,e,l,_,f]=[0,1,2,3,4])=>s`
layout(location = ${t.toString()}) out vec4 particle_physics1_write_texture;
layout(location = ${e.toString()}) out vec4 particle_physics2_write_texture;
layout(location = ${l.toString()}) out vec4 particle_physics3_write_texture;
layout(location = ${_.toString()}) out vec4 particle_body_write_texture;
layout(location = ${f.toString()}) out vec4 particle_firefly_write_texture;
void writeParticle(Particle particle) {
	particle_physics1_write_texture = vec4(particle.position, particle.tile);
	particle_physics2_write_texture = vec4(particle.velocity, particle.desired_velocity);
	particle_physics3_write_texture = vec4(particle.wander_theta, 0.0, 0.0, 0.0);
	particle_body_write_texture = vec4(particle.state, particle.width, particle.height, particle.agility);
	particle_firefly_write_texture = vec4(particle.phase, particle.brightness, 0.0, 0.0);
}
`,Q=s`
${_t}
${dt}
${vt}
${pt}
`,ee=s`
uniform vec2 world_resolution;
uniform float world_aspect_ratio;
uniform sampler2D world_level_texture;
`,te=s`
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

void setNextWorldPosition(out vec2 tile, out vec2 position) {
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
`,ut=s`#version 300 es

precision highp float;

${Y}
${ee}
${te}

${Q}
${Oe()}

${J}

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
`,ft=s`
struct FireflyDeposit {
	vec2 xy;
	float total_brightness;
	int n_particles;
	vec2 mass_center;
	vec2 total_velocity;
};
`,yt=s`
uniform float firefly_deposit_bin_divisor;
uniform ivec2 firefly_deposit_texture_dimensions;
uniform sampler2D firefly_deposit_texture;
uniform sampler2D firefly_deposit2_texture;
`,xt=s`
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
`,De=s`
${ft}
${yt}
${xt}
`,mt=s`#version 300 es

precision highp float;

${Y}
${ee}
${te}
${lt}

${Q}
${Oe()}

${De}

${ct}
${J}

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
`,Be=s`
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
`,gt=s`#version 300 es

precision highp float;

${Y}
${Be}
${J}
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
`,ht=s`#version 300 es

precision highp float;

${Y}
${ee}
${te}

in vec4 a_position;
in vec2 a_texcoord;
out vec4 v_color;
out vec2 v_texcoord;
out float random_offset;
out mat2 rotation;

uniform float interpolation;
uniform mat4 projection;
uniform mat4 view;

${Q}

${J}
${Be}

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
`,wt=s`#version 300 es

precision highp float;

in vec4 deposit_data;
in vec4 deposit2_data;
layout(location = 0) out vec4 deposit_data_texture;
layout(location = 1) out vec4 deposit2_data_texture;

void main() {
  deposit_data_texture += deposit_data;
  deposit2_data_texture += deposit2_data;
}
`,Tt=s`#version 300 es

precision highp float;

${Y}
${ee}
${te}

in vec4 a_dummy;
out vec4 deposit_data;
out vec4 deposit2_data;

uniform mat4 projection;
uniform mat4 view;

${Q}
${De}

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
`,bt=s`#version 300 es

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
`,Ft=s`#version 300 es

precision highp float;

uniform vec2 resolution;
uniform sampler2D input_texture;
layout(location = 0) out vec4 color;

void main() {
	vec2 uv = vec2((gl_FragCoord.x-0.5) / (resolution.x-1.0), gl_FragCoord.y / -resolution.y);
	color = vec4(texture(input_texture, uv).rgb, 0.5);
}
`;var Ee=typeof Float32Array<"u"?Float32Array:Array;Math.hypot||(Math.hypot=function(){for(var t=0,e=arguments.length;e--;)t+=arguments[e]*arguments[e];return Math.sqrt(t)});function K(){var t=new Ee(16);return Ee!=Float32Array&&(t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0),t[0]=1,t[5]=1,t[10]=1,t[15]=1,t}function Ce(t){return t[0]=1,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=1,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=1,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function Rt(t,e,l,_,f,w,d){var m=1/(e-l),b=1/(_-f),y=1/(w-d);return t[0]=-2*m,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=-2*b,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=2*y,t[11]=0,t[12]=(e+l)*m,t[13]=(f+_)*b,t[14]=(d+w)*y,t[15]=1,t}var Ue=Rt;function At(t){const e=a();function l(){e.value&&Ze(t,e.value,t.TRIANGLE_FAN,0,3)}const _=(()=>{let{program:f,createUniform:w,setResolution:d}=Z(t,Ft);return w("1i","input_texture")(0),function(y,A,re){t.useProgram(f),t.bindFramebuffer(t.FRAMEBUFFER,null),t.viewport(0,0,A,re),d([A,re]),n(t,t.TEXTURE0,y),l()}})();return{reset(){e.value=fe(t,new Float32Array([-1,3,3,-1,-1,-1]))},destroy(){Le(t)},render(f,w,d){_(f,w,d)}}}function Pt(t){const e=t.getContext("webgl2",{premultipliedAlpha:!1});if(!e)throw new Error("Unable to get WebGL context.");e.getExtension("EXT_color_buffer_float"),e.getExtension("EXT_float_blend"),console.log("HELLO");const l=window.devicePixelRatio||1,_=a(0),f=a(.001+Math.random()*.01),w=a(0),d=a(1),m=a(1),b=O(()=>[d.value,m.value]),y=O(()=>d.value/m.value);function A(){const o=t.getBoundingClientRect();d.value=t.width=Math.max(1,o.width*l),m.value=t.height=Math.max(1,o.height*l)}new ResizeObserver(A).observe(t),A();function D(o){u(w,o("1f","dt")),u(_,o("1f","time")),u(f,o("1f","seed")),u(b,o("2fv","viewport_resolution")),u(y,o("1f","viewport_aspect_ratio"))}function B(o){u(b,o("2fv","world_resolution")),u(y,o("1f","world_aspect_ratio"))}function oe(){e.drawArrays(e.TRIANGLE_FAN,0,3)}const c=Pe({down:!1,x:0,y:0,last_x:0,last_y:0});function ye(o){if(!c.down){o.preventDefault();const r=t.getBoundingClientRect();c.x=o.pageX-r.x,c.y=o.pageY-r.y,c.last_x=o.pageX-r.x,c.last_y=o.pageY-r.y,c.down=!0,document.addEventListener("pointermove",ie),document.addEventListener("pointerup",ae)}}function ie(o){if(c.down){const r=t.getBoundingClientRect();c.last_x=c.x,c.last_y=c.y,c.x=o.pageX-r.x,c.y=o.pageY-r.y}}function ae(){c.down&&(c.down=!1,document.removeEventListener("pointermove",ie),document.removeEventListener("pointerup",ae))}const Me=6,Ne=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),xe=a(),Ge=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),Ie=fe(e,Ge),se=2e4,i=O(()=>Je(se)||[1,1]);function ne(o){u(i,o("2iv","particle_data_dimensions"))}const P=a(),M=a(),E=a(),N=a(),C=a(),G=a(),U=a(),I=a(),L=a(),S=a(),x=a(),me=a(6),F=O(()=>b.value.map(o=>Math.ceil(o/me.value)));function ge(o){u(me,o("1f","firefly_deposit_bin_divisor")),u(F,o("2iv","firefly_deposit_texture_dimensions"))}const H=a(),W=a(),le=a(),ce=a(),he=a(),_e=Pe([0,0,0,0]),q=document.getElementById("nav-base");function we(){if(q){const{x:o,y:r,width:g,height:T}=q.getBoundingClientRect();_e.length=0,_e.push(o*l,r*l,g*l,T*l)}}const Se=new ResizeObserver(we);q&&Se.observe(q),we();const Te=At(e),Xe=(()=>{let{program:o,createUniform:r}=Z(e,ut);return D(r),B(r),function(){e.useProgram(o),e.viewport(0,0,i.value[0],i.value[1]),v(e,x.value,e.COLOR_ATTACHMENT0,M.value),v(e,x.value,e.COLOR_ATTACHMENT1,N.value),v(e,x.value,e.COLOR_ATTACHMENT2,G.value),v(e,x.value,e.COLOR_ATTACHMENT3,I.value),v(e,x.value,e.COLOR_ATTACHMENT4,S.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3,e.COLOR_ATTACHMENT4]),oe()}})(),$e=(()=>{let{program:o,createUniform:r}=Z(e,mt);D(r),B(r),ne(r);const g=r("1i","particle_physics1_read_texture"),T=r("1i","particle_physics2_read_texture"),X=r("1i","particle_physics3_read_texture"),$=r("1i","particle_body_read_texture"),z=r("1i","particle_firefly_read_texture");g(0),T(1),X(2),$(3),z(4),ge(r);const V=r("1i","firefly_deposit_texture"),k=r("1i","firefly_deposit2_texture");return V(5),k(6),r("1i","world_level_texture")(7),u(O(()=>c.down),r("1i","pointer_down")),u(O(()=>[c.x*l,c.y*l,c.last_x*l,c.last_y*l]),r("4fv","pointer_state")),function(){e.useProgram(o),e.viewport(0,0,i.value[0],i.value[1]),n(e,e.TEXTURE0,P.value),n(e,e.TEXTURE1,E.value),n(e,e.TEXTURE2,C.value),n(e,e.TEXTURE3,U.value),n(e,e.TEXTURE4,L.value),n(e,e.TEXTURE5,H.value),n(e,e.TEXTURE6,W.value),n(e,e.TEXTURE7,ce.value),v(e,x.value,e.COLOR_ATTACHMENT0,M.value),v(e,x.value,e.COLOR_ATTACHMENT1,N.value),v(e,x.value,e.COLOR_ATTACHMENT2,G.value),v(e,x.value,e.COLOR_ATTACHMENT3,I.value),v(e,x.value,e.COLOR_ATTACHMENT4,S.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3,e.COLOR_ATTACHMENT4]),oe()}})(),ze=(()=>{let{program:o,createUniform:r,createAttribute:g}=Fe(e,Tt,wt);D(r),B(r),ne(r),ge(r);const T=r("1i","particle_physics1_read_texture"),X=r("1i","particle_physics2_read_texture"),$=r("1i","particle_physics3_read_texture"),z=r("1i","particle_body_read_texture"),V=r("1i","particle_firefly_read_texture");T(0),X(1),$(2),z(3),V(4),r("1i","firefly_deposit_texture")(5),r("1i","firefly_deposit2_texture")(6);const j=r("Matrix4fv","projection");return r("Matrix4fv","view")(!1,Ce(K())),function(){e.useProgram(o),e.viewport(0,0,F.value[0],F.value[1]),e.enable(e.BLEND),e.blendEquationSeparate(e.FUNC_ADD,e.FUNC_ADD),e.blendFuncSeparate(e.ONE,e.ONE,e.ONE,e.ONE),j(!1,Ue(K(),-y.value,y.value,-1,1,-1,1)),n(e,e.TEXTURE0,P.value),n(e,e.TEXTURE1,E.value),n(e,e.TEXTURE2,C.value),n(e,e.TEXTURE3,U.value),n(e,e.TEXTURE4,L.value),n(e,e.TEXTURE5,H.value),n(e,e.TEXTURE6,W.value),v(e,le.value,e.COLOR_ATTACHMENT0,H.value),v(e,le.value,e.COLOR_ATTACHMENT1,W.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1]),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),e.drawArrays(e.POINTS,0,se),e.disable(e.BLEND)}})(),Ve=(()=>{let{program:o,createUniform:r,createAttribute:g}=Fe(e,ht,gt);D(r),B(r),ne(r);const T=r("1i","particle_physics1_read_texture"),X=r("1i","particle_physics2_read_texture"),$=r("1i","particle_physics3_read_texture"),z=r("1i","particle_body_read_texture"),V=r("1i","particle_firefly_read_texture");T(0),X(1),$(2),z(3),V(4);const k=r("1f","interpolation"),de=r("Matrix4fv","projection"),j=r("Matrix4fv","view"),ve=g("a_position"),pe=g("a_texcoord");return e.bindBuffer(e.ARRAY_BUFFER,Ie),e.enableVertexAttribArray(pe),e.vertexAttribPointer(pe,2,e.FLOAT,!1,0,0),j(!1,Ce(K())),function(je=0){e.useProgram(o),e.bindFramebuffer(e.FRAMEBUFFER,null),e.viewport(0,0,d.value,m.value),k(je),de(!1,Ue(K(),-y.value,y.value,-1,1,-1,1)),n(e,e.TEXTURE0,P.value),n(e,e.TEXTURE1,E.value),n(e,e.TEXTURE2,C.value),n(e,e.TEXTURE3,U.value),n(e,e.TEXTURE4,L.value),e.bindBuffer(e.ARRAY_BUFFER,xe.value),e.enableVertexAttribArray(ve),e.vertexAttribPointer(ve,2,e.FLOAT,!1,0,0),e.drawArraysInstanced(e.TRIANGLES,0,Me,se)}})(),Ye=(()=>{let{program:o,createUniform:r}=Z(e,bt);D(r),B(r);const g=r("4fv","nav_base_dimensions");return function(){e.useProgram(o),e.viewport(0,0,d.value,m.value),g(_e),v(e,he.value,e.COLOR_ATTACHMENT0,ce.value),e.drawBuffers([e.COLOR_ATTACHMENT0]),oe()}})();function be(){Re(e),Ae(e),P.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),M.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),E.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),N.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),C.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),G.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),U.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),I.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),L.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),S.value=p(e,i.value[0],i.value[1],e.RGBA32F,e.RGBA,e.FLOAT),x.value=ue(e),xe.value=fe(e,Ne),H.value=p(e,F.value[0],F.value[1],e.RGBA32F,e.RGBA,e.FLOAT),W.value=p(e,F.value[0],F.value[1],e.RGBA32F,e.RGBA,e.FLOAT),le.value=ue(e),ce.value=p(e,d.value,m.value,e.RGBA32F,e.RGBA,e.FLOAT),he.value=ue(e),Xe(),Ye(),h(P,M),h(E,N),h(C,G),h(U,I),h(L,S),Te.reset(),console.log("Reset"),console.log(e.getError())}function He(){console.log("INIT"),A(),t.addEventListener("pointerdown",ye),be()}function We(){t.removeEventListener("pointerdown",ye),document.removeEventListener("pointermove",ie),document.removeEventListener("pointerup",ae),Re(e),Le(e),Ae(e),Te.destroy();let o=e.getExtension("WEBGL_lose_context");o&&o.loseContext()}function qe(o){w.value=o,_.value+=o,ze(),$e(),h(P,M),h(E,N),h(C,G),h(U,I),h(L,S)}function ke(o){e.bindFramebuffer(e.FRAMEBUFFER,null),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),Ve(o)}return{init:He,reset:be,update:qe,render:ke,destroy:We}}const Lt=ot({__name:"Fireflies",setup(t){const e=a();return nt(e),(l,_)=>(at(),it("div",null,[st("canvas",{ref_key:"canvas",ref:e,class:"w-full h-full","touch-action":"none"},null,512)]))}});export{Lt as _};
