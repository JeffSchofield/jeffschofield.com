import{e as i,H as wt,K as Tt,g as At,I as Rt,q as N,O as Fe,p as Et,c as bt,o as Ft,a as y,d as ke,Y as z,b as D,a2 as H,x as C,a4 as K,B as V,w as Ct,a3 as Pt}from"./index-NEwBzf1q.js";import{L as Ut,g as l,a as S,s as u,b as n,f as w,m as ae,l as h,k as P,n as je,i as Ce,r as Qe,j as Pe,c as Ue,d as Lt,e as St,h as Ke}from"./utils-CP7LEk5c.js";import{i as Je,c as fe,o as Ze}from"./mat4-CmMOcA9f.js";import{s as J}from"./index-B-nBI0Kh.js";import{_ as Ot}from"./SvgLogo.vue_vue_type_script_setup_true_lang-CCznGCSF.js";const X=new Ut;X.update_time_step=16;function Dt(t,{n_fireflies:c=3e4}={}){let e;const f=i(c),g=i(.05),A=i(.45),T=i(.01),p=i(10),v=i(40),b=i(.25);function R(){if(t.value)try{console.log("Creating firefly program"),e=or(t.value,{n_fireflies:f.value}),J(g,e.distance_per_ms),J(A,e.alignment_strength),J(T,e.cohesion_strength),J(p,e.separation_strength),J(v,e.wander_radius),J(b,e.wander_displacement),X.linkProgram(e),console.log("Firefly program created")}catch(m){console.error(m)}}return wt(()=>Tt(()=>{X.init(),At(t,a=>{e&&(X.unlinkProgram(e),e.destroy()),R()},{flush:"sync",immediate:!0})})),Rt(()=>{e&&(X.unlinkProgram(e),e.destroy())}),{distance_per_ms:g,alignment_strength:A,cohesion_strength:T,separation_strength:p,wander_radius:v,wander_displacement:b,setFireflies(m){f.value=m,e&&(e.n_particles.value=m,e.init())},pause(){X.cancelLoop()},play(){X.requestLoop()}}}const Z=l`
uniform float dt;
uniform float time;
uniform vec2 viewport_resolution;
uniform float viewport_aspect_ratio;
`,et=l`
uniform bool pointer_down;
uniform vec4 pointer_state;
uniform vec4 pointer_tool;
`,Nt=l`
mat2 rotationMatrix(float a) {
	return mat2(cos(a), -sin(a), sin(a), cos(a));
}
`,pe=l`
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
`,It=l`
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
	float refractory;
};
`,Bt=l`
// const float PARTICLE_DISTANCE_PER_MS = 0.05;
// const float PARTICLE_ALIGNMENT_STRENGTH = 0.45;
// const float PARTICLE_COHESION_STRENGTH = 0.01;
// const float PARTICLE_SEPARATION_STRENGTH = 10.0;
const float PARTICLE_ATTRACT_TO_CENTER_STRENGTH = 0.0;

const float FIREFLY_PHASE_PER_MS = 0.00005;
const float FIREFLY_BRIGHTNESS_DECAY_PER_MS = 0.001;
const float FIREFLY_REFRACTORY_PER_MS = 0.0005;
`,Gt=l`
uniform ivec2 particle_data_dimensions;
uniform sampler2D particle_physics1_read_texture;
uniform sampler2D particle_physics2_read_texture;
uniform sampler2D particle_physics3_read_texture;
uniform sampler2D particle_body_read_texture;
uniform sampler2D particle_firefly_read_texture;

uniform float PARTICLE_DISTANCE_PER_MS;
uniform float PARTICLE_ALIGNMENT_STRENGTH;
uniform float PARTICLE_COHESION_STRENGTH;
uniform float PARTICLE_SEPARATION_STRENGTH;
uniform float PARTICLE_WANDER_RADIUS;
uniform float PARTICLE_WANDER_DISPLACEMENT;
`,Mt=l`
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
	float refractory = firefly.z;

	return Particle(state, width, height, agility, position, tile, velocity, desired_velocity, current_speed, wander_theta, phase, brightness, refractory);
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
`,tt=([t,c,e,f,g]=[0,1,2,3,4])=>l`
layout(location = ${t.toString()}) out vec4 particle_physics1_write_texture;
layout(location = ${c.toString()}) out vec4 particle_physics2_write_texture;
layout(location = ${e.toString()}) out vec4 particle_physics3_write_texture;
layout(location = ${f.toString()}) out vec4 particle_body_write_texture;
layout(location = ${g.toString()}) out vec4 particle_firefly_write_texture;
void writeParticle(Particle particle) {
	particle_physics1_write_texture = vec4(particle.position, particle.tile);
	particle_physics2_write_texture = vec4(particle.velocity, particle.desired_velocity);
	particle_physics3_write_texture = vec4(particle.wander_theta, 0.0, 0.0, 0.0);
	particle_body_write_texture = vec4(particle.state, particle.width, particle.height, particle.agility);
	particle_firefly_write_texture = vec4(particle.phase, particle.brightness, particle.refractory, 0.0);
}
`,ve=l`
${It}
${Bt}
${Gt}
${Mt}
`,me=l`
uniform vec2 world_resolution;
uniform float world_aspect_ratio;
uniform sampler2D world_level_texture;
`,xe=l`
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
`,zt=l`#version 300 es

precision highp float;

${Z}
${me}
${xe}

${ve}
${tt()}

${pe}

void main() {
	int id = particleIdFromXY(gl_FragCoord.xy);

	int state = 1;

	float size = randomGaussian(gl_FragCoord.xy + 0.00654, 0.004, 0.0008, 0.0);
	float width = size * 0.8;//randomGaussian(gl_FragCoord.xy + 0.00123, , , -50.0);
	float height = size;//randomGaussian(gl_FragCoord.xy + 0.00945, , , -0.1);
	// Increase minimum agility to ensure all particles move more noticeably
	float agility = max(0.8, randomGaussian(gl_FragCoord.xy + 0.00654, 1.2, 0.3, 0.0));

	vec2 position = vec2(random(gl_FragCoord.xy + 0.00033) * 2.0 - 1.0, random(gl_FragCoord.xy + 0.00081) * 2.0 - 1.0);
	// Use highly concentrated Gaussian distribution to force most particles to center tiles
	vec2 tile = vec2(
		clamp(floor(randomGaussian(gl_FragCoord.xy + 0.00683, float(N_TILES) / 2.0, float(N_TILES) / 10.0, 0.0)), 0.0, float(N_TILES) - 1.0),
		clamp(floor(randomGaussian(gl_FragCoord.xy + 0.00182, float(N_TILES) / 2.0, float(N_TILES) / 10.0, 0.0)), 0.0, float(N_TILES) - 1.0)
	);
	vec2 velocity = normalize(randomVector(gl_FragCoord.xy + 0.00219)) * agility;//normalize(randomVector(gl_FragCoord.xy + 0.00219)) * agility;
	float current_speed = length(velocity);
	float wander_theta = 0.0;

	vec2 world_coords = getWorldCoords(tile, position);
	float phase = snoise(world_coords * 2.);
	float brightness = 0.0;
	float refractory = 0.0;

	writeParticle(Particle(state, width, height, agility, position, tile, velocity, velocity, current_speed, wander_theta, phase, brightness, refractory));
}
`,Ht=l`
struct FireflyDeposit {
	vec2 xy;
	float total_brightness;
	int n_particles;
	vec2 mass_center;
	vec2 total_velocity;
};
`,Vt=l`
uniform float firefly_deposit_bin_divisor;
uniform ivec2 firefly_deposit_texture_dimensions;
uniform sampler2D firefly_deposit_texture;
uniform sampler2D firefly_deposit2_texture;
`,Xt=l`
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
`,rt=l`
${Ht}
${Vt}
${Xt}
`,Wt=l`#version 300 es

precision highp float;

${Z}
${me}
${xe}
${et}

uniform sampler2D walls_sdf_texture;

${ve}
${tt()}

${rt}

${Nt}
${pe}

void main() {
	Particle particle = getParticleByFragCoord(gl_FragCoord);
	vec2 particle_world_coords = getWorldCoords(particle.tile, particle.position);

	vec2 world_uv = getUVFromWorldCoords(particle_world_coords);
	FireflyDeposit firefly_deposit = readFireflyDeposit(world_uv);
	vec2 deposit_center_uv = fireflyDepositUVFromXY(firefly_deposit.xy);
	vec2 deposit_center_world_coords = getWorldCoordsFromUV(deposit_center_uv);
	vec2 own_distance_to_center = particle_world_coords - deposit_center_world_coords;

	int range = 3; // Increased from 1 to 2 for wider interaction range
	FireflyDeposit current_deposit;

	int total_neighbors = 0;
	float total_deposits = 0.0;

	float total_neighborhood_brightness = 0.0;
	float max_brightness = 0.0;
	vec2 brightest_force = vec2(0.0, 0.0);
	vec2 separation_force = vec2(0.0, 0.0);
	vec2 cohesion_force = vec2(0.0, 0.0); // Uncommented for flocking behavior
	vec2 alignment_force = vec2(0.0, 0.0);
	
	// For weight normalization
	float total_weight = 0.0;
	float total_cohesion_weight = 0.0;
	float total_separation_weight = 0.0;

	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
			// Circular pattern check - skip tiles outside the circle
			float dist_sq = float(x*x + y*y);
			if (dist_sq > float(range*range)) continue;
			
			// Calculate distance-based weighting - closer tiles have stronger influence
			float dist_weight = 1.0 / (1.0 + dist_sq * 0.5);
			
			if (x == 0 && y == 0) current_deposit = firefly_deposit;
			else current_deposit = getFireflyDepositByXY(firefly_deposit.xy + vec2(x, y));

			int n_particles = current_deposit.n_particles;
			if (x == 0 && y == 0) n_particles -= 1; // Don't include self in particle count

			if (n_particles > 0) {
				total_deposits += 1.0;
				total_neighbors += n_particles;
				float n_particles_float = float(n_particles);
				
				// Combined weight based on particle count and distance
				float tile_weight = dist_weight * n_particles_float;
				total_weight += tile_weight;

				// Weight the brightness influence by distance
				total_neighborhood_brightness += current_deposit.total_brightness * dist_weight;
				if (x == 0 && y == 0) total_neighborhood_brightness -= particle.brightness * dist_weight; // Don't include self

				vec2 deposit_center_uv = fireflyDepositUVFromXY(current_deposit.xy);
				vec2 deposit_center_world_coords = getWorldCoordsFromUV(deposit_center_uv);
				vec2 avg_mass_center = ((x == 0 && y == 0) ? (current_deposit.mass_center - own_distance_to_center) : current_deposit.mass_center) / n_particles_float;
				vec2 deposit_mass_coords = deposit_center_world_coords + avg_mass_center;

				// Apply weight to brightness attraction
				if (current_deposit.total_brightness * dist_weight > max_brightness) {
					max_brightness = current_deposit.total_brightness * dist_weight;
					brightest_force = deposit_center_world_coords - particle_world_coords;
				}

				// Apply weight to alignment
				alignment_force += (current_deposit.total_velocity / n_particles_float) * tile_weight;
				
				// Add cohesion force (skip own tile to prevent self-attraction)
				if (!(x == 0 && y == 0)) {
					float cohesion_weight = tile_weight;
					cohesion_force += (deposit_mass_coords - particle_world_coords) * cohesion_weight;
					total_cohesion_weight += cohesion_weight;
				}

				// Separation with weighted influence
				vec2 separation_direction = particle_world_coords - deposit_mass_coords;
				float sep_distance = length(separation_direction);
				
				// Unique separation threshold per particle (prevents uniform distances)
				float unique_sep_threshold = 0.05 + sin(particle.phase * TAU + particle.tile.x) * 0.02;
				
				// Apply separation with non-linear falloff and distance weighting
				if (sep_distance < unique_sep_threshold) {
					// Exponential falloff creates more organic spacing
					float repulsion = exp(-sep_distance * 20.0) * pow(n_particles_float, 2.0) * dist_weight;
					// Add slight directional variation to prevent linear alignments
					float angle_jitter = sin(time * 0.5 + gl_FragCoord.x * 0.01) * 45.0;
					mat2 rotation = rotationMatrix(angle_jitter);
					
					separation_force += rotation * separation_direction * repulsion;
					total_separation_weight += repulsion;
				}
			}
		}
	}
	
	// Normalize forces by their respective weights
	if (total_weight > 0.0) {
		alignment_force /= total_weight;
	}

	if (total_separation_weight > 0.0) {
		separation_force /= total_separation_weight;
	}

	if (total_cohesion_weight > 0.0) {
		cohesion_force /= total_cohesion_weight;
	}

	// Biologically-inspired coupled oscillator model based on firefly research
	// Implementation based on the Kuramoto model and phase response curves observed in real fireflies
	if (total_neighborhood_brightness > 0.0) {
		// Phase response curve - sensitivity depends on current phase
		// Fireflies are most sensitive in mid-cycle (around phase 0.5-0.8)
		float sensitivity = sin(particle.phase * PI) * 0.25 + 0.5; // Creates a bell curve with peak at 0.5
		
		// Distance-weighted influence (brightness already accounts for spatial distribution)
		float influence = total_neighborhood_brightness * 0.005;
		
		// Biological noise factor (natural variability in response)
		float noise = random(gl_FragCoord.xy + time + 0.00652) * 0.002;
		
		// Gradual phase adjustment that increases as phase gets closer to 1.0
		// More "ready" fireflies are more influenced by neighbors' flashes
		if (particle.phase > 0.3) {
			particle.phase += sensitivity * influence + noise;
		}
		
		// Natural limit (don't allow phase to exceed 1.0)
		particle.phase = min(particle.phase, 1.0);
	}

	// Decay brightness with slight randomness for more natural appearance
	particle.brightness = max(0.0, particle.brightness - (FIREFLY_BRIGHTNESS_DECAY_PER_MS + random(gl_FragCoord.xy + 0.00123) * 0.002) * dt);
	particle.refractory = max(0.0, particle.refractory - (FIREFLY_REFRACTORY_PER_MS * dt));

	// Mouse interaction
	bool in_mouse = false;
	vec2 aspect = vec2(world_aspect_ratio, 1.0);
	if (pointer_tool.x == 1.0 && pointer_down) {
		vec2 pointer_xy = (pointer_state.xy / world_resolution * aspect * 2.0) - aspect;
		if (length(pointer_xy - particle_world_coords) < 0.15 * min(world_aspect_ratio, 1.0)) {
			in_mouse = true;
			particle.phase = 0.0;
			particle.brightness = 0.9;
		}
	}

	// Progress oscillator + flash
	if (!in_mouse) {
		particle.phase += FIREFLY_PHASE_PER_MS * dt;// + random(gl_FragCoord.xy * 0.001) * dt;
		if (particle.phase > 1.0 && particle.refractory == 0.0) {
			particle.phase = mod(particle.phase, 1.0);
			particle.brightness = 1.0;
			particle.refractory = 0.0;
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
	vec2 wander_point = particle_world_coords + normalize(particle.velocity) * 100.0;
	float theta = particle.wander_theta + atan(particle.velocity.y, particle.velocity.x);
	wander_point += vec2(
		cos(theta) * PARTICLE_WANDER_RADIUS,
		sin(theta) * PARTICLE_WANDER_RADIUS
	);

	particle.wander_theta += random(gl_FragCoord.xy + time + 0.00782) * (PARTICLE_WANDER_DISPLACEMENT*2.0) - PARTICLE_WANDER_DISPLACEMENT;

	desired_velocity += normalize(wander_point - particle_world_coords) * particle.agility * (sin(time * 10.) * 0.25);
	n_desired_forces += 1.0;

	// Attract to brightness
	desired_velocity += brightest_force * smoothstep(0.0, 5.0, max_brightness) * 2. * particle.agility;
	n_desired_forces += 1.0;

	// Align with neighbors (already normalized by total_weight)
	if (alignment_force != vec2(0.0, 0.0)) {
		desired_velocity += normalize(alignment_force) * PARTICLE_ALIGNMENT_STRENGTH;
		n_desired_forces += 1.0;
	}

	// Separate from neighbors (already normalized by total_separation_weight)
	if (separation_force != vec2(0.0, 0.0)) {
		desired_velocity += separation_force * PARTICLE_SEPARATION_STRENGTH;
		n_desired_forces += 1.0;
	}
	
	// Add cohesion force (new)
	if (cohesion_force != vec2(0.0, 0.0)) {
		// Use a lower multiplier (0.5) to balance against separation
		desired_velocity += normalize(cohesion_force) * PARTICLE_COHESION_STRENGTH;
		n_desired_forces += 1.0;
	}
	
	// Add a very subtle pull towards the center of the world
	vec2 center_pull = -particle_world_coords; // Direction to center (0,0)
	float distance_from_center = length(center_pull);
	if (distance_from_center > 0.1) { // Only apply when not too close to center
		center_pull = normalize(center_pull) * PARTICLE_ATTRACT_TO_CENTER_STRENGTH * distance_from_center; // Stronger pull when farther away
		desired_velocity += center_pull;
		n_desired_forces += 1.0;
	}

	// --- Wall Avoidance Force ---
	// const float AVOID_DISTANCE_THRESHOLD = 20.0; // pixels, start avoiding when this close
	// const float AVOIDANCE_STRENGTH = 0.5;      // Multiplier for the avoidance force
	// const float MIN_AVOID_DIST = 1.0;          // Minimum distance to prevent division by zero or extreme forces

	// vec2 particle_sdf_uv = particle_world_coords / vec2(firefly_deposit_texture_dimensions); // UV for SDF texture
	vec4 wall_data = texture(walls_sdf_texture, world_uv); // xy = gradient, z = dist

	vec2 wall_normal = wall_data.xy; // Normalized direction away from wall
	float dist_to_wall = wall_data.z;
	float wall_threshold = 0.002; // Tune: how close is "touching" a wall
	float steer_start = 0.0035;     // distance at which to start steering away
	float steer_falloff_power = 1.5; // how sharp the falloff is

	if (dist_to_wall < wall_threshold) {
		// COLLISION: project out of wall along gradient
		particle.position += wall_normal * (4.0 + wall_threshold);
	
		// Optionally: clip velocity so it doesn't push back into the wall
		float into_wall = dot(particle.velocity, wall_normal);
		if (into_wall < 0.0) {
			particle.velocity -= wall_normal * into_wall;
		}
	} else if (dist_to_wall < steer_start) {
		// SOFT AVOIDANCE: apply curved steering falloff
		float steer_strength = pow(1.0 - dist_to_wall / steer_start, steer_falloff_power);
	
		// Use actual velocity for approach angle, not desired velocity
		vec2 approach_dir = normalize(particle.velocity);
		
		// Calculate tangents
		vec2 tangent = vec2(-wall_normal.y, wall_normal.x); // Clockwise tangent
		
		// Find the correct tangent direction by projecting approach velocity
		// If projection is negative, flip the tangent
		float tangent_projection = dot(approach_dir, tangent);
		if (tangent_projection < 0.0) {
			tangent = -tangent; // Use opposite tangent if needed
		}
		
		// Calculate projection of approach onto wall normal (shows how much we're moving toward the wall)
		float normal_component = dot(approach_dir, wall_normal);
		
		// Create a balanced directional force:
		// 1. Keep momentum along the wall with the correct tangent
		// 2. Add substantial repulsion based on proximity to wall and how directly we're approaching
		vec2 wall_repulsion = wall_normal * (steer_strength * 2.0 + abs(normal_component) * 0.5);
		vec2 redirected = normalize(tangent + wall_repulsion);
		
		// Apply final steered velocity while maintaining speed
		float wall_nudge_strength = random(gl_FragCoord.xy + time + 0.00952) * 1.5;

		// Apply final steered velocity with stronger wall avoidance
		desired_velocity += redirected + (wall_normal * wall_nudge_strength) * (1.0 + steer_strength);
		n_desired_forces += 1.0;
	}
	// --- End Wall Avoidance ---


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
`,ot=l`
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
`,$t=l`#version 300 es

precision highp float;

${Z}
${ot}
${pe}
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
`,Yt=l`#version 300 es

precision highp float;

${Z}
${me}
${xe}

in vec4 a_position;
in vec2 a_texcoord;
out vec4 v_color;
out vec2 v_texcoord;
out float random_offset;
out mat2 rotation;

uniform float interpolation;
uniform mat4 projection;
uniform mat4 view;

${ve}

${pe}
${ot}

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

	c = hsv2rgb(vec3(0.75 + ((animate_brightness) / 8.), smoothstep(0.95, 0.0, pow(animate_brightness, 3.0)), 0.3 + (random(vec2(id) + 0.00782) * 0.3) + (animate_brightness / 2.)));
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
`,qt=l`#version 300 es

precision highp float;

in vec4 deposit_data;
in vec4 deposit2_data;
layout(location = 0) out vec4 deposit_data_texture;
layout(location = 1) out vec4 deposit2_data_texture;

void main() {
  deposit_data_texture += deposit_data;
  deposit2_data_texture += deposit2_data;
}
`,kt=l`#version 300 es

precision highp float;

${Z}
${me}
${xe}

in vec4 a_dummy;
out vec4 deposit_data;
out vec4 deposit2_data;

uniform mat4 projection;
uniform mat4 view;

${ve}
${rt}

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
`,jt=l`#version 300 es

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
`,Kt=l`#version 300 es

precision highp float;

uniform vec2 resolution;
uniform sampler2D input_texture;
layout(location = 0) out vec4 color;

void main() {
	vec2 uv = vec2((gl_FragCoord.x-0.5) / (resolution.x-1.0), gl_FragCoord.y / -resolution.y);
	color = vec4(texture(input_texture, uv).rgb, 1.0);
}
`,Jt=l`#version 300 es

precision highp float;

${Z}
${et}

uniform sampler2D walls_read_texture;
layout(location = 0) out vec4 walls_write_texture;

float sdfLineSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

void main() {
	vec2 aspect = vec2(viewport_aspect_ratio, 1.0);
  vec2 position = (gl_FragCoord.xy / viewport_resolution * aspect * 2.0) - aspect;
  vec2 uv = vec2(
		(position.x + viewport_aspect_ratio) / (viewport_aspect_ratio*2.0), // (1.777) / 3.55 = 0.5
		(position.y + 1.0) / 2.0 // 1. / 2. = 0.5
	);

	bool in_mouse = false;
	if (pointer_down) {
		vec2 pointer_xy = (pointer_state.xy / viewport_resolution * aspect * 2.0) - aspect;
    vec2 pointer_last_xy = (pointer_state.zw / viewport_resolution * aspect * 2.0) - aspect;

    // Distance to the pointer-drawn line segment
    float thickness = 0.007;
    float dist = sdfLineSegment(position, pointer_last_xy, pointer_xy) - thickness;

		if (dist <= 0.0) {
			in_mouse = true;
		}
	}

	if (in_mouse) {
		walls_write_texture = vec4(vec3(0.5), 1.0);
	} else {
    walls_write_texture = texture(walls_read_texture, uv);
  }
}
`,Zt=l`#version 300 es
precision highp float;

uniform vec2 resolution;
uniform sampler2D wall_texture;  // input: binary wall mask

layout(location = 0) out vec4 write_texture;

bool isWall(vec2 uv) {
  return texture(wall_texture, uv).rgb != vec3(0.0);
}

void main() {
  vec2 frag_uv = gl_FragCoord.xy / resolution;

  bool center = isWall(frag_uv);
  bool edge = false;
	int shell_radius = 2;

  if (center) {
    float px = 1.0 / resolution.x; // assumes square pixels
    // Check 8-neighbors
    for (int y = -shell_radius; y <= shell_radius; ++y) {
      for (int x = -shell_radius; x <= shell_radius; ++x) {
        if (x == 0 && y == 0) continue;
        vec2 offset = vec2(float(x), float(y)) * px;
        if (!isWall(frag_uv + offset)) {
          edge = true;
        }
      }
    }
  }

  // Only emit seed if this is a wall *and* on the edge
  if (center && edge) {
    write_texture = vec4(frag_uv, 1.0, 1.0); // store UV, discard rest
  } else {
    write_texture = vec4(0.0, 0.0, center ? 1.0 : 0.0, 1.0);
  }
}
`,Qt=l`#version 300 es

precision highp float;

uniform int step_size;
uniform vec2 resolution;
uniform sampler2D read_texture;
layout(location = 0) out vec4 write_texture;

vec4 readTexture(vec2 p) {
	vec2 uv = p / resolution;
	return texture(read_texture, uv);
}

void main() {
  vec2 center = gl_FragCoord.xy;
	vec2 data = readTexture(center).zw;

	float best_dist = 999999.0;
	vec2 best_coord = vec2(0.0);
	for (int y = -1; y <= 1; ++y) {
		for (int x = -1; x <= 1; ++x) {
			vec2 fc = center + vec2(x,y) * float(step_size);
			vec2 ntc = readTexture(fc).xy;
			float d = length(ntc - (center/resolution));
			if ((ntc.x != 0.0) && (ntc.y != 0.0) && (d < best_dist)) {
					best_dist = d;
					best_coord = ntc;
			}
		}
	}

	write_texture = vec4(best_coord, data);
}
`,er=l`#version 300 es
precision highp float;

uniform vec2 resolution;
uniform sampler2D read_texture;

const float falloff_radius = 0.05;

layout(location = 0) out vec4 write_texture;

vec4 readTexture(vec2 p) {
	vec2 uv = p / resolution;
	return texture(read_texture, uv);
}

void main() {
	vec2 frag_uv = gl_FragCoord.xy / resolution;
	vec4 data = readTexture(gl_FragCoord.xy);
	// write_texture = data;

	vec2 seed_coords = data.xy;
	bool is_wall = data.z == 1.0;
	if (seed_coords == vec2(0.0) && !is_wall) {
		write_texture = vec4(0.0);
	} else {
		vec2 dir = is_wall ? seed_coords - frag_uv : frag_uv - seed_coords;
		float dist = length(dir);
		vec2 grad = dist > 0.0 ? normalize(dir) : vec2(0.0);
	
		// Store normalized gradient and raw SDF distance
		write_texture = vec4(grad, is_wall ? 0.0 : dist, 1.0);
	}
}
`,tr=l`#version 300 es
precision highp float;

uniform vec2 resolution;
uniform sampler2D read_texture;

layout(location = 0) out vec4 write_texture;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution;
  float px = 1.0 / resolution.x; // assumes square pixels

  vec2 sum = vec2(0.0);
  float count = 0.0;

  // 3×3 box blur of gradient vectors only
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      vec2 offset = vec2(float(x), float(y)) * px;
      vec4 read_sample = texture(read_texture, uv + offset);
      vec2 grad = read_sample.xy;

      sum += grad;
      count += 1.0;
    }
  }

  vec2 avg = sum / count;
  vec2 norm_grad = normalize(avg);

  // Use the original distance value (z)
  float dist = texture(read_texture, uv).z;

  write_texture = vec4(norm_grad, dist, 1.0);
}
`;function rr(t){const c=i([0,0]),e=i();function f(){e.value&&Lt(t,e.value,t.TRIANGLE_FAN,0,3)}const g=i(),A=i(),T=i(),p=(()=>{let{program:m,createUniform:a,setResolution:s}=S(t,Zt);return u(c,s),a("1i","read_texture")(0),function(I){t.useProgram(m),t.viewport(0,0,c.value[0],c.value[1]),n(t,t.TEXTURE0,I),w(t,T.value,t.COLOR_ATTACHMENT0,g.value),t.drawBuffers([t.COLOR_ATTACHMENT0]),f()}})(),v=(()=>{let{program:m,createUniform:a,setResolution:s}=S(t,Qt);u(c,s);const U=a("1i","step_size");return a("1i","read_texture")(0),function(d){t.useProgram(m),t.viewport(0,0,c.value[0],c.value[1]),U(d),n(t,t.TEXTURE0,g.value),w(t,T.value,t.COLOR_ATTACHMENT0,A.value),t.drawBuffers([t.COLOR_ATTACHMENT0]),f()}})(),b=(()=>{let{program:m,createUniform:a,setResolution:s}=S(t,er);return u(c,s),a("1i","read_texture")(0),function(){t.useProgram(m),t.viewport(0,0,c.value[0],c.value[1]),n(t,t.TEXTURE0,g.value),w(t,T.value,t.COLOR_ATTACHMENT0,A.value),t.drawBuffers([t.COLOR_ATTACHMENT0]),f()}})(),R=(()=>{let{program:m,createUniform:a,setResolution:s}=S(t,tr);return u(c,s),a("1i","read_texture")(0),function(I){t.useProgram(m),t.viewport(0,0,c.value[0],c.value[1]),n(t,t.TEXTURE0,g.value),w(t,T.value,t.COLOR_ATTACHMENT0,I),t.drawBuffers([t.COLOR_ATTACHMENT0]),f()}})();return{reset(){e.value=Ue(t,new Float32Array([-1,3,3,-1,-1,-1]))},destroy(){Ce(t),Qe(t),Pe(t)},createSDFTexture(m,a,s,U){T.value=ae(t),c.value=[s,U],g.value=h(t,c.value[0],c.value[1],t.RGBA32F,t.RGBA,t.FLOAT),A.value=h(t,c.value[0],c.value[1],t.RGBA32F,t.RGBA,t.FLOAT),p(m);for(let F=1;F<=10;F++)v(Math.floor(Math.pow(2,F))),P(g,A);b(),P(g,A),R(a),je(t,g.value),je(t,A.value),t.deleteFramebuffer(T.value)}}}function or(t,{n_fireflies:c=3e4}={}){const e=t.getContext("webgl2",{premultipliedAlpha:!1});if(!e)throw new Error("Unable to get WebGL context.");if(!e.getExtension("EXT_color_buffer_float"))throw new Error("EXT_color_buffer_float not supported!");if(!e.getExtension("EXT_float_blend"))throw new Error("EXT_float_blend not supported!");const f=window.devicePixelRatio||1,g=i(0),A=i(.001+Math.random()*.01),T=i(0),p=i(1),v=i(1),b=N(()=>[p.value,v.value]),R=N(()=>p.value/v.value);function m(){const o=t.getBoundingClientRect();p.value=t.width=Math.max(1,o.width*f),v.value=t.height=Math.max(1,o.height*f)}new ResizeObserver(m).observe(t),m();function s(o){u(T,o("1f","dt")),u(g,o("1f","time")),u(A,o("1f","seed")),u(b,o("2fv","viewport_resolution")),u(R,o("1f","viewport_aspect_ratio"))}function U(o){u(b,o("2fv","world_resolution")),u(R,o("1f","world_aspect_ratio"))}function F(){e.drawArrays(e.TRIANGLE_FAN,0,3)}const I=Fe({type:0}),d=Fe({down:!1,x:0,y:0,last_x:0,last_y:0});function Le(o){u(N(()=>d.down),o("1i","pointer_down")),u(N(()=>[d.x*f,d.y*f,d.last_x*f,d.last_y*f]),o("4fv","pointer_state")),u(N(()=>[I.type,0,0,0]),o("4fv","pointer_tool"))}function Se(o){if(!d.down){o.preventDefault();const r=t.getBoundingClientRect(),x=o.pageX-r.x,E=o.pageY-r.y;d.x=x,d.y=E,d.last_x=x,d.last_y=E,d.down=!0,document.addEventListener("pointermove",ye),document.addEventListener("pointerup",ge)}}function ye(o){if(d.down){const r=t.getBoundingClientRect(),x=o.pageX-r.x,E=o.pageY-r.y;d.x=x,d.y=E}}function ge(){d.down&&(d.down=!1,document.removeEventListener("pointermove",ye),document.removeEventListener("pointerup",ge))}const Oe=i(.05),De=i(.45),Ne=i(.01),Ie=i(10),Be=i(40),Ge=i(.25),it=6,at=new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),Me=i(),st=new Float32Array([0,1,1,1,0,0,0,0,1,1,1,0]),lt=Ue(e,st),se=i(c),_=N(()=>St(se.value)||[2,2]);function he(o){u(_,o("2iv","particle_data_dimensions"))}const W=i(),Q=i(),$=i(),ee=i(),Y=i(),te=i(),q=i(),re=i(),k=i(),oe=i(),L=i(),ze=i(6),B=N(()=>b.value.map(o=>Math.ceil(o/ze.value)));function He(o){u(ze,o("1f","firefly_deposit_bin_divisor")),u(B,o("2iv","firefly_deposit_texture_dimensions"))}const le=i(),ne=i(),we=i(),ie=i(),Te=i(),Ve=i();i(),i(),i();const Ae=i(),Re=i(),Xe=i(),Ee=Fe([0,0,0,0]),ce=document.getElementById("nav-base");function We(){if(ce){const{x:o,y:r,width:x,height:E}=ce.getBoundingClientRect();Ee.length=0,Ee.push(o*f,r*f,x*f,E*f)}}const nt=new ResizeObserver(We);ce&&nt.observe(ce),We();const $e=rr(e),ct=(()=>{const{program:o,createUniform:r}=S(e,Kt),x=r("2fv","resolution");return r("1i","input_texture")(0),function(G,O,M){e.useProgram(o),e.bindFramebuffer(e.FRAMEBUFFER,null),e.viewport(0,0,O,M),e.enable(e.BLEND),x([O,M]),n(e,e.TEXTURE0,G),F(),e.disable(e.BLEND)}})(),_t=(()=>{let{program:o,createUniform:r}=S(e,zt);return s(r),U(r),function(){e.useProgram(o),e.viewport(0,0,_.value[0],_.value[1]),w(e,L.value,e.COLOR_ATTACHMENT0,Q.value),w(e,L.value,e.COLOR_ATTACHMENT1,ee.value),w(e,L.value,e.COLOR_ATTACHMENT2,te.value),w(e,L.value,e.COLOR_ATTACHMENT3,re.value),w(e,L.value,e.COLOR_ATTACHMENT4,oe.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3,e.COLOR_ATTACHMENT4]),F()}})(),dt=(()=>{let{program:o,createUniform:r}=S(e,Wt);s(r),U(r),he(r);const x=r("1i","particle_physics1_read_texture"),E=r("1i","particle_physics2_read_texture"),j=r("1i","particle_physics3_read_texture"),G=r("1i","particle_body_read_texture"),O=r("1i","particle_firefly_read_texture");x(0),E(1),j(2),G(3),O(4),He(r);const M=r("1i","firefly_deposit_texture"),_e=r("1i","firefly_deposit2_texture");return M(5),_e(6),r("1i","world_level_texture")(7),r("1i","walls_sdf_texture")(8),Le(r),u(Oe,r("1f","PARTICLE_DISTANCE_PER_MS")),u(De,r("1f","PARTICLE_ALIGNMENT_STRENGTH")),u(Ne,r("1f","PARTICLE_COHESION_STRENGTH")),u(Ie,r("1f","PARTICLE_SEPARATION_STRENGTH")),u(Be,r("1f","PARTICLE_WANDER_RADIUS")),u(Ge,r("1f","PARTICLE_WANDER_DISPLACEMENT")),function(){e.useProgram(o),e.viewport(0,0,_.value[0],_.value[1]),n(e,e.TEXTURE0,W.value),n(e,e.TEXTURE1,$.value),n(e,e.TEXTURE2,Y.value),n(e,e.TEXTURE3,q.value),n(e,e.TEXTURE4,k.value),n(e,e.TEXTURE5,le.value),n(e,e.TEXTURE6,ne.value),n(e,e.TEXTURE7,Re.value),n(e,e.TEXTURE8,Ae.value),w(e,L.value,e.COLOR_ATTACHMENT0,Q.value),w(e,L.value,e.COLOR_ATTACHMENT1,ee.value),w(e,L.value,e.COLOR_ATTACHMENT2,te.value),w(e,L.value,e.COLOR_ATTACHMENT3,re.value),w(e,L.value,e.COLOR_ATTACHMENT4,oe.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1,e.COLOR_ATTACHMENT2,e.COLOR_ATTACHMENT3,e.COLOR_ATTACHMENT4]),F()}})(),ut=(()=>{let{program:o,createUniform:r}=Ke(e,kt,qt);s(r),U(r),he(r),He(r);const x=r("1i","particle_physics1_read_texture"),E=r("1i","particle_physics2_read_texture"),j=r("1i","particle_physics3_read_texture"),G=r("1i","particle_body_read_texture"),O=r("1i","particle_firefly_read_texture");x(0),E(1),j(2),G(3),O(4),r("1i","firefly_deposit_texture")(5),r("1i","firefly_deposit2_texture")(6);const de=r("Matrix4fv","projection");return r("Matrix4fv","view")(!1,Je(fe())),function(){e.useProgram(o),e.viewport(0,0,B.value[0],B.value[1]),e.enable(e.BLEND),e.blendEquationSeparate(e.FUNC_ADD,e.FUNC_ADD),e.blendFuncSeparate(e.ONE,e.ONE,e.ONE,e.ONE),de(!1,Ze(fe(),-R.value,R.value,-1,1,-1,1)),n(e,e.TEXTURE0,W.value),n(e,e.TEXTURE1,$.value),n(e,e.TEXTURE2,Y.value),n(e,e.TEXTURE3,q.value),n(e,e.TEXTURE4,k.value),n(e,e.TEXTURE5,le.value),n(e,e.TEXTURE6,ne.value),w(e,we.value,e.COLOR_ATTACHMENT0,le.value),w(e,we.value,e.COLOR_ATTACHMENT1,ne.value),e.drawBuffers([e.COLOR_ATTACHMENT0,e.COLOR_ATTACHMENT1]),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),e.drawArrays(e.POINTS,0,se.value),e.disable(e.BLEND)}})(),ft=(()=>{let{program:o,createUniform:r,createAttribute:x}=Ke(e,Yt,$t);s(r),U(r),he(r);const E=r("1i","particle_physics1_read_texture"),j=r("1i","particle_physics2_read_texture"),G=r("1i","particle_physics3_read_texture"),O=r("1i","particle_body_read_texture"),M=r("1i","particle_firefly_read_texture");E(0),j(1),G(2),O(3),M(4);const _e=r("1f","interpolation"),de=r("Matrix4fv","projection"),be=r("Matrix4fv","view"),ue=x("a_position"),qe=x("a_texcoord");return e.bindBuffer(e.ARRAY_BUFFER,lt),e.enableVertexAttribArray(qe),e.vertexAttribPointer(qe,2,e.FLOAT,!1,0,0),be(!1,Je(fe())),function(ht=0){e.useProgram(o),e.bindFramebuffer(e.FRAMEBUFFER,null),e.viewport(0,0,p.value,v.value),_e(ht),de(!1,Ze(fe(),-R.value,R.value,-1,1,-1,1)),n(e,e.TEXTURE0,W.value),n(e,e.TEXTURE1,$.value),n(e,e.TEXTURE2,Y.value),n(e,e.TEXTURE3,q.value),n(e,e.TEXTURE4,k.value),e.bindBuffer(e.ARRAY_BUFFER,Me.value),e.enableVertexAttribArray(ue),e.vertexAttribPointer(ue,2,e.FLOAT,!1,0,0),e.drawArraysInstanced(e.TRIANGLES,0,it,se.value)}})(),pt=(()=>{let{program:o,createUniform:r}=S(e,Jt);return s(r),Le(r),r("1i","walls_read_texture")(0),function(){e.useProgram(o),e.viewport(0,0,p.value,v.value),n(e,e.TEXTURE0,ie.value),w(e,Ve.value,e.COLOR_ATTACHMENT0,Te.value),e.drawBuffers([e.COLOR_ATTACHMENT0]),F()}})(),vt=(()=>{let{program:o,createUniform:r}=S(e,jt);s(r),U(r);const x=r("4fv","nav_base_dimensions");return function(){e.useProgram(o),e.viewport(0,0,p.value,v.value),x(Ee),w(e,Xe.value,e.COLOR_ATTACHMENT0,Re.value),e.drawBuffers([e.COLOR_ATTACHMENT0]),F()}})();function Ye(){Ce(e),Pe(e),W.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),Q.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),$.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),ee.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),Y.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),te.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),q.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),re.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),k.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),oe.value=h(e,_.value[0],_.value[1],e.RGBA32F,e.RGBA,e.FLOAT),L.value=ae(e),Me.value=Ue(e,at),le.value=h(e,B.value[0],B.value[1],e.RGBA32F,e.RGBA,e.FLOAT),ne.value=h(e,B.value[0],B.value[1],e.RGBA32F,e.RGBA,e.FLOAT),we.value=ae(e),ie.value=h(e,p.value,v.value,e.RGBA32F,e.RGBA,e.FLOAT),Te.value=h(e,p.value,v.value,e.RGBA32F,e.RGBA,e.FLOAT),Ve.value=ae(e),Ae.value=h(e,p.value,v.value,e.RGBA32F,e.RGBA,e.FLOAT),Re.value=h(e,p.value,v.value,e.RGBA32F,e.RGBA,e.FLOAT),Xe.value=ae(e),_t(),vt(),P(W,Q),P($,ee),P(Y,te),P(q,re),P(k,oe)}function mt(){m(),t.addEventListener("pointerdown",Se),$e.reset(),Ye()}function xt(){t.removeEventListener("pointerdown",Se),document.removeEventListener("pointermove",ye),document.removeEventListener("pointerup",ge),Ce(e),Qe(e),Pe(e);let o=e.getExtension("WEBGL_lose_context");o&&o.loseContext()}function yt(o){T.value=o,g.value+=o,I.type==0&&d.down&&(pt(),P(ie,Te),$e.createSDFTexture(ie.value,Ae.value,p.value,v.value)),ut(),dt(),P(W,Q),P($,ee),P(Y,te),P(q,re),P(k,oe),d.last_x=d.x,d.last_y=d.y}function gt(o){e.bindFramebuffer(e.FRAMEBUFFER,null),e.clearColor(0,0,0,0),e.clear(e.COLOR_BUFFER_BIT),ct(ie.value,p.value,v.value),ft(o)}return{init:mt,reset:Ye,update:yt,render:gt,destroy:xt,n_particles:se,distance_per_ms:Oe,alignment_strength:De,cohesion_strength:Ne,separation_strength:Ie,wander_radius:Be,wander_displacement:Ge}}const ir={class:"relative w-full h-full"},ar={class:"flex absolute bottom-0 inset-x-0 pointer-events-none p-1"},sr={class:"flex-1 max-w-15 flex flex-col gap-1/2"},lr={class:"flex-1 flex flex-col"},nr={class:"flex-1 flex flex-col"},cr={class:"flex-1 flex flex-col"},_r={class:"flex-1 flex flex-col"},dr={class:"flex-1 flex flex-col"},ur={class:"flex-1 flex flex-col"},fr={class:"flex gap-1/2"},pr={class:"flex-1 flex flex-col"},vr={layout:"fullscreen"},Tr=Et({...vr,__name:"fireflies-obs",setup(t){const c=i(),e=i(2e5),f=N(()=>e.value.toLocaleString()),{setFireflies:g,distance_per_ms:A,alignment_strength:T,cohesion_strength:p,separation_strength:v,wander_radius:b,wander_displacement:R}=Dt(c,{n_fireflies:e.value});return(m,a)=>(Ft(),bt("div",ir,[y("div",ar,[ke(Ot,{class:"text-neutral-200 h-2 w-3 pointer-events-auto hover:text-white cursor-pointer",onClick:a[0]||(a[0]=s=>m.$router.push("/"))}),a[10]||(a[10]=y("div",{class:"flex-1"},null,-1)),y("div",sr,[y("div",lr,[z(y("input",{type:"range",min:"0",max:"1",step:"0.001","onUpdate:modelValue":a[1]||(a[1]=s=>K(A)?A.value=s:null),class:"w-full pointer-events-auto"},null,512),[[H,C(A),void 0,{number:!0}]]),D(" "+V(C(A))+" Distance Per MS ",1)]),y("div",nr,[z(y("input",{type:"range",min:"0",max:"1",step:"0.001","onUpdate:modelValue":a[2]||(a[2]=s=>K(T)?T.value=s:null),class:"w-full pointer-events-auto"},null,512),[[H,C(T),void 0,{number:!0}]]),D(" "+V(C(T))+" Alignment ",1)]),y("div",cr,[z(y("input",{type:"range",min:"0",max:"0.05",step:"0.001","onUpdate:modelValue":a[3]||(a[3]=s=>K(p)?p.value=s:null),class:"w-full pointer-events-auto"},null,512),[[H,C(p),void 0,{number:!0}]]),D(" "+V(C(p))+" Cohesion ",1)]),y("div",_r,[z(y("input",{type:"range",min:"0",max:"10",step:"0.001","onUpdate:modelValue":a[4]||(a[4]=s=>K(v)?v.value=s:null),class:"w-full pointer-events-auto"},null,512),[[H,C(v),void 0,{number:!0}]]),D(" "+V(C(v))+" Separation ",1)]),y("div",dr,[z(y("input",{type:"range",min:"0",max:"360",step:"1","onUpdate:modelValue":a[5]||(a[5]=s=>K(b)?b.value=s:null),class:"w-full pointer-events-auto"},null,512),[[H,C(b),void 0,{number:!0}]]),D(" "+V(C(b))+" Wander Radius ",1)]),y("div",ur,[z(y("input",{type:"range",min:"0",max:"1",step:"0.001","onUpdate:modelValue":a[6]||(a[6]=s=>K(R)?R.value=s:null),class:"w-full pointer-events-auto"},null,512),[[H,C(R),void 0,{number:!0}]]),D(" "+V(C(R))+" Wander Displacement ",1)]),y("div",fr,[y("div",pr,[z(y("input",{type:"range",min:"1000",max:"4000000",step:"10000","onUpdate:modelValue":a[7]||(a[7]=s=>e.value=s),class:"w-full pointer-events-auto"},null,512),[[H,e.value,void 0,{number:!0}]]),D(" "+V(f.value)+" Fireflies ",1)]),ke(Pt,{size:"xs",onClick:a[8]||(a[8]=s=>C(g)(e.value)),class:"px-1 pointer-events-auto"},{default:Ct(()=>a[9]||(a[9]=[D("Set")])),_:1})])])]),y("canvas",{ref_key:"canvas",ref:c,class:"w-full h-full","touch-action":"none"},null,512)]))}});export{Tr as default};
