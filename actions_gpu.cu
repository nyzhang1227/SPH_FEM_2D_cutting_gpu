// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "actions_gpu.h"
#include "plasticity.cuh"
#include "tool_gpu.cuh"


#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

extern int global_step;

static bool m_plastic = false;
static bool m_thermal = false;			//consider thermal conduction in workpiece
static bool m_fric_heat_gen = false;	//consider that friction produces heat

__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static joco_constants johnson_cook;
__constant__ static trml_constants thermals_wp;
__constant__ static trml_constants thermals_tool;
__constant__ static tool_constants tool_consts;
__constant__ static wear_constants wear_consts;



__device__ float_t find_max(float_t a, float_t b)
{
	if (a >= b) return a;
	return b;
}

__device__ float_t find_min(float_t a, float_t b)
{
	if (a >= b) return b;
	return a;
}

__device__ __forceinline__ float_t stress_angle(float_t sxx, float_t sxy, float_t syy, float_t eps) {
	float_t numer = 2.*sxy;
	float_t denom = sxx - syy + eps;
	return 0.5*atan2f(numer,denom);
}

__device__ int on_segment_mesh(float2_t left, float2_t right, float2_t point) {
	// ray towards right
	if (point.y <= fmaxf(left.y, right.y) && point.y >= fminf(left.y, right.y)) {	    
		if (left.x == right.x && point.x < left.x) return 1;  // vertical segment
		if (abs(left.y - right.y) != 0){
		    float_t x_on_seg = right.x + (left.x - right.x) / (left.y - right.y)  * (point.y - right.y);
		    if (x_on_seg > point.x && x_on_seg != left.x) return 1;
		} else {
			if (point.x <= fmax(left.x, right.x)) return 1; // horizontal segment	
		}
	}
	return 0;
}

__device__ vec2_t segment_closest_point_mesh(line_gpu l, vec2_t xq) {
	if (l.vertical) {
		return vec2_t(l.b, xq.y);
	}

	float_t b = l.b;
	float_t a = l.a;

	float_t bb = -1;
	float_t cc = b;
	float_t aa = a;

	float_t px = (bb * (bb * xq.x - aa * xq.y) - aa * cc) / (aa * aa + bb * bb);
	float_t py = (aa * (-bb * xq.x + aa * xq.y) - bb * cc) / (aa * aa + bb * bb);

	return vec2_t(px, py);
}

__device__ int inside_judgement(float2_t qp, const float2_t* left, const float2_t* right, const int num_seg) {
	// ray casting method

	int intersections = 0;
	for (int i = 0; i < num_seg; i++) {
		intersections += on_segment_mesh(left[i], right[i], qp);
	}

	if (intersections % 2 == 0) return 0;
	return 1;
}

__device__ int find_shortest_distance(float2_t qp, const float2_t* left, const float2_t* right, const float2_t* n, const int num_seg) {
	// wrong
	int seg = -1;
	float_t depth = FLT_MAX;


	for (int i = 0; i < num_seg; i++) {

		float_t d = 0.;
		//float_t length = sqrtf((left[i].y - right[i].y) * (left[i].y - right[i].y) + (left[i].x - right[i].x) * (left[i].x - right[i].x));
		float_t ymax = find_max(left[i].y, right[i].y);
		float_t ymin = find_min(left[i].y, right[i].y);
		float_t xmax = find_max(left[i].x, right[i].x);
		float_t xmin = find_min(left[i].x, right[i].x);

			

		// vertical segment
		
		if (left[i].x == right[i].x ){
			if (qp.y <= ymax && qp.y >= ymin) {
				//depth = left[i].x - qp.x;
				seg = i;
				break;
			}
		}
		
		d = (qp.x - left[i].x) * n[i].x + (qp.y - left[i].y) * n[i].y;

		if (d > 0. && d < depth)
		{
				float_t intersec_y = qp.y - d * n[i].y;

				if (intersec_y <= ymax && intersec_y >= ymin )
				{
					depth = d;
					seg = i;
				}
		}
	}

	return seg;
}

__device__ int find_shortest_distance_v1(float2_t qp, float_t& dN, vec2_t& n_re, const float2_t* left, const float2_t* right, const float2_t* n, const int num_seg) {
	int seg = -1;
	float_t depth = FLT_MAX;
	float_t limit_value = 2.5e-4; // 1.4e-4 for Ti64


	for (int i = 0; i < num_seg; i++) {

		float_t d = 0.;
		float_t ymax = find_max(left[i].y, right[i].y);
		float_t ymin = find_min(left[i].y, right[i].y);
		float_t xmax = find_max(left[i].x, right[i].x);
		float_t xmin = find_min(left[i].x, right[i].x);


        if (qp.x > xmin - 0.0008 && qp.x < xmax + 0.0008 && qp.y > ymin - 0.0008 && qp.y < ymax + 0.0008) 
		{
			d = (qp.x - left[i].x) * n[i].x + (qp.y - left[i].y) * n[i].y;

		    if (d >= 0. && d < depth)
		    {
			    float_t intersec_x = qp.x - d * n[i].x;
			    float_t intersec_y = qp.y - d * n[i].y;

			    if (intersec_y <= ymax && intersec_y >= ymin && intersec_x <= xmax && intersec_x >= xmin)
			    {
			    	depth = d;
				    seg = i;
			    }
		    }
		
		}
	}
    // for the concave geometry, find the closest segment point
	if (seg != -1 && depth <= limit_value) {

		n_re = vec2_t(-n[seg].x, -n[seg].y);
		dN = -depth;
	} else {
		depth = FLT_MAX;
		float_t d_trial = 0.;

		for (int i = 0; i < num_seg; i++) {
			d_trial = sqrtf((qp.x - right[i].x) * (qp.x - right[i].x) + (qp.y - right[i].y) * (qp.y - right[i].y));
			if (d_trial < depth) {
				depth = d_trial;
				seg = i;
			}	
		}
		if (seg == -1) {
            printf("Could not find the cloest segment for particle at location x:%f, y:%f!\n", qp.x, qp.y);
            return seg;
		}
		
		float2_t n_sec;
		n_sec.x = n[seg].x;
		n_sec.y = n[seg].y;

		
		// push the particle from the middle line, finding two adjacent segments is not easy. They are not numbered sequentially
		for (int j = 0; j < num_seg; j++) 
		{
		  if (left[j].y == right[seg].y && left[j].x == right[seg].x) {
			n_sec.x = (n[j].x + n[seg].x) / 2.;
			n_sec.y = (n[j].y + n[seg].y) / 2.;
			break;
		  }
		}
		
		float_t n_length = sqrtf(n_sec.x * n_sec.x + n_sec.y * n_sec.y);
        
		n_re = vec2_t(-n_sec.x / n_length, -n_sec.y / n_length);

		if (depth > limit_value) {
			printf("Strange depth value step 2, seg is %d, depth is %f at x:%f y:%f, seg right is x: %f y:%f, seg left is x: %f y:%f \n", seg, depth, qp.x, qp.y, right[seg].x, right[seg].y, left[seg].x, left[seg].y);

		}
		dN = -1.0*depth;
	}

	return seg;
}

__global__ void do_material_eos(const float_t *__restrict__ rho, float_t *__restrict__ p, const float_t *__restrict__ in_tool, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float_t rho0 = physics.rho0;
	float_t c0   = sqrtf(physics.K/rho0);
	float_t rhoi = rho[pidx];
	p[pidx] = c0*c0*(rhoi - rho0);

}

__global__ void do_corrector_artificial_stress(const float_t *__restrict__ rho, const float_t *__restrict__ p, const float4_t *__restrict__ S, const float_t *__restrict__ in_tool,
		float4_t *__restrict__ R, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float_t eps  = correctors.stresseps;

	float_t rhoi = rho[pidx];
	float_t pi   = p[pidx];
	float4_t Si  = S[pidx];

	float_t sxx  = Si.x;
	float_t sxy  = Si.y;
	float_t syy  = Si.z;

	sxx -= pi;
	syy -= pi;

	float_t rhoi21 = 1./(rhoi*rhoi);

	float_t theta = stress_angle(sxx,sxy,syy,0.);

	float_t cos_theta = cosf(theta);
	float_t sin_theta = sinf(theta);

	float_t cos_theta2 = cos_theta*cos_theta;
	float_t sin_theta2 = sin_theta*sin_theta;

	float_t rot_sxx = cos_theta2*sxx + 2.0*cos_theta*sin_theta*sxy + sin_theta2*syy;
	float_t rot_syy = sin_theta2*sxx - 2.0*cos_theta*sin_theta*sxy + cos_theta2*syy;

	float_t rot_rxx = 0.;
	float_t rot_ryy = 0.;

	if (rot_sxx > 0) rot_rxx = -eps*rot_sxx*rhoi21;
	if (rot_syy > 0) rot_ryy = -eps*rot_syy*rhoi21;

	float4_t Ri = make_float4_t(cos_theta2*rot_rxx + sin_theta2*rot_ryy,
			cos_theta*sin_theta*(rot_rxx - rot_ryy),
			sin_theta2*rot_rxx + cos_theta2*rot_ryy,
			0.);

	R[pidx] = Ri;
}

__global__ void do_material_stress_rate_jaumann(const float4_t *__restrict__ v_der, const float4_t *__restrict__ Stress, const float_t *in_tool,
		float4_t *S_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx] == 1.) return;

	float_t G = physics.G;

	float4_t vi_der = v_der[pidx];
	float4_t Si     = Stress[pidx];

	float_t vx_x = vi_der.x;
	float_t vx_y = vi_der.y;
	float_t vy_x = vi_der.z;
	float_t vy_y = vi_der.w;

	float_t Sxx = Si.x;
	float_t Sxy = Si.y;
	float_t Syy = Si.z;
	float_t Szz = Si.w;

	mat3x3_t epsdot = mat3x3_t(vx_x, 0.5*(vx_y + vy_x), 0., 0.5*(vx_y + vy_x), vy_y, 0., 0., 0., 0.);
	mat3x3_t omega  = mat3x3_t(0.  , 0.5*(vy_x - vx_y), 0., 0.5*(vx_y - vy_x), 0., 0., 0., 0., 0.);
	mat3x3_t S      = mat3x3_t(Sxx, Sxy, 0., Sxy, Syy, 0., 0., 0., Szz);
	mat3x3_t I      = mat3x3_t(1.);

	float_t trace_epsdot = epsdot[0][0] + epsdot[1][1] + epsdot[2][2];

	mat3x3_t Si_t = float_t(2.)*G*(epsdot - float_t(1./3.)*trace_epsdot*I) + omega*S + S*glm::transpose(omega);	//Belytschko (3.7.9)

	S_t[pidx].x = Si_t[0][0];
	S_t[pidx].y = Si_t[0][1];
	S_t[pidx].z = Si_t[1][1];
	S_t[pidx].w = Si_t[2][2];
}

__global__ void do_material_fric_heat_gen(const float2_t * __restrict__ vel, const float2_t * __restrict__ f_fric, const float2_t * __restrict__ n, const float_t *in_tool,
		float_t *__restrict__ T, float2_t vel_tool, float_t dt, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	const float_t eta = thermals_wp.eta;

	//compute F_fric_mag;
	float2_t f_T =  f_fric[pidx];
	float_t  f_fric_mag = sqrtf(f_T.x*f_T.x + f_T.y*f_T.y);

	if (f_fric_mag == 0.) {
		return;
	}

	//compute v_rel
	float2_t normal     = n[pidx];
	float2_t v_particle = vel[pidx];
	float2_t v_diff     = make_float2_t(v_particle.x-vel_tool.x, v_particle.y-vel_tool.y);

	float_t  v_diff_dot_normal = v_diff.x*normal.x + v_diff.y*normal.y;
	float2_t v_relative = make_float2_t(v_diff.x -  v_diff_dot_normal * normal.x, v_diff.y - v_diff_dot_normal * normal.y);

	float_t  v_rel_mag  = sqrtf(v_relative.x*v_relative.x + v_relative.y*v_relative.y);

	T[pidx] += eta*dt*f_fric_mag*v_rel_mag/(thermals_wp.cp*physics.mass);
}

__global__ void do_material_fric_heat_gen_FEM_v21(particle_gpu particles, segment_FEM_gpu segments,  float2_t vel_tool, 
    const float_t* __restrict__ T_nodes, float_t dt, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (particles.blanked[pidx]==1.) return;
	if (particles.eps_pl[pidx] > 20. )  return;  //for preventing hot particles oscillation

	//compute F_fric_mag;
	float2_t f_T = particles.ft[pidx];
	float_t  f_fric_mag = sqrtf(f_T.x * f_T.x + f_T.y * f_T.y);


	int seg = particles.on_seg[pidx];
	if (seg == -1) {
		return;
	}

	const float_t partition_wp = tool_consts.beta;

	//compute v_rel
	float2_t normal = particles.n[pidx];
	float2_t v_particle = particles.vel[pidx];
	float2_t v_diff = make_float2_t(v_particle.x - vel_tool.x, v_particle.y - vel_tool.y);

	float_t  v_diff_dot_normal = v_diff.x * normal.x + v_diff.y * normal.y;
	float2_t v_relative = make_float2_t(v_diff.x - v_diff_dot_normal * normal.x, v_diff.y - v_diff_dot_normal * normal.y);

	float_t  v_rel_mag = sqrtf(v_relative.x * v_relative.x + v_relative.y * v_relative.y);
	float_t value = tool_consts.eta * f_fric_mag * v_rel_mag;
	
	if (isnan(v_rel_mag)) 
	{	
		printf("nan velocity, v_particle is %e, %e,  v_diff is %e, %e, v_diff_dot_normal is %e \n", v_particle.x, v_particle.y, v_diff.x, v_diff.y, v_diff_dot_normal);
	    particles.blanked[pidx]=1.;
		return;
	}
	if (v_rel_mag > 0.03) printf("large velocity on friction, v_rel_mag is %f, x is %f, y is %f, normal_x: %f, normal_y: %f, v_particle is %e, %e,  v_diff is %e, %e, v_diff_dot_normal is %e \n", 
	    v_rel_mag, particles.pos[pidx].x,particles.pos[pidx].y , normal.x, normal.y,v_particle.x, v_particle.y, v_diff.x, v_diff.y, v_diff_dot_normal);
	
	float2_t f_C = particles.fc[pidx];
	float_t  f_normal = sqrtf(f_C.x * f_C.x + f_C.y * f_C.y);
	float_t rho_ = particles.rho[pidx];
    float_t length = sqrtf(tool_consts.slave_mass / rho_);
	float_t pressure_trial = f_normal / length;

	if (v_rel_mag > 0.03) return;
	float_t T_buffer = particles.T[pidx] + value * (1 - partition_wp) * dt  / thermals_wp.cp / physics.mass;
	
	if (T_buffer > 1550.){ 
        particles.T[pidx] = 1550.;
		return;
	} else {
		particles.T[pidx] = T_buffer;
		atomicAdd(&segments.fric_heat[seg], value * partition_wp);
	}
    
#ifdef WEAR_NODE_SHIFT
	int left_no = segments.marks[seg].z;
	int right_no = segments.marks[seg].w;
	float_t T_seg = 0.5 * (T_nodes[left_no] + T_nodes[right_no]);

	float_t vel_tool_norm = sqrt(vel_tool.x * vel_tool.x + vel_tool.y * vel_tool.y);
	
    if ( v_rel_mag > 0.008 ){ 
		return;
	} 	else if ( v_rel_mag > vel_tool_norm * 8.0 && abs (normal.y) > 0.5){   // around the flank face        //particles.blanked[pidx] = 1.;
        return;
	} 
	
	atomicAdd(&segments.physical_para[seg].x, 1.);
	atomicAdd(&segments.physical_para[seg].y, pressure_trial);
	atomicAdd(&segments.physical_para[seg].z, v_rel_mag);
	atomicAdd(&segments.physical_para[seg].w, T_seg);
	atomicAdd(&segments.sliding_force[seg], f_fric_mag / length);
#endif
}


__global__ void do_compute_contact_forces_FEM_tool_v3_s1(int N, particle_gpu particles, segment_FEM_gpu segments, vec2_t tl_ref, float_t ref_dist, vec2_t vel_t, float_t dt, int seg_num)
{
#ifdef USE_FEM_TOOL
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (particles.blanked[idx]==1.) return;

	float2_t pos = particles.pos[idx];
	float2_t vel = particles.vel[idx];

	vec2_t xs(pos.x, pos.y);
	vec2_t vs(vel.x, vel.y);

	int inside = 0;

	if (glm::distance(xs, tl_ref) < 1.05 * ref_dist) {
		inside = inside_judgement(pos, segments.left, segments.right, seg_num);
	}

	if (inside == 0) {
		particles.fc[idx].x = 0.;
		particles.fc[idx].y = 0.;
		particles.ft[idx].x = 0.;
		particles.ft[idx].y = 0.;
		particles.n[idx].x = 0.;
		particles.n[idx].y = 0.;

		particles.on_seg[idx] = -1;

		return;
	}

	int seg_no = -1;
	float_t gN ;
	vec2_t n_;
	// find the matching segment

	seg_no = find_shortest_distance_v1(pos, gN, n_, segments.left, segments.right, segments.n, seg_num); 
	__syncthreads();
	particles.on_seg[idx] = seg_no;
   

	if (seg_no == -1) {
		particles.fc[idx].x = 0.;
		particles.fc[idx].y = 0.;
		particles.ft[idx].x = 0.;
		particles.ft[idx].y = 0.;
		particles.n[idx].x = 0.;
		particles.n[idx].y = 0.;
#ifdef USE_FEM_TOOL
		particles.on_seg[idx] = -1;
#endif
		//printf("particle %d didn't find the cloest segement, x: %f, y: %f, inside number is %d \n", idx, particles.pos[idx].x, particles.pos[idx].y, inside );
		return;
	}

	if (gN <-0.00025){  // need to reset when different particle sizes are used
		particles.blanked[idx] = 1.;
		particles.fc[idx].x = 0.;
		particles.fc[idx].y = 0.;
		particles.ft[idx].x = 0.;
		particles.ft[idx].y = 0.;
		particles.n[idx].x = 0.;
		particles.n[idx].y = 0.;
#ifdef USE_FEM_TOOL
		particles.on_seg[idx] = -1;
#endif
        printf("Blank particle with penetration depth %f, seg %d , seg_n_x:%f, seg_n_y:%f, particle location x:%f y:%f, particle speed x:%f y:%f , temperature %fK\n", -gN, seg_no, n_.x, n_.y, pos.x, pos.y, vel_t.x, vel_t.y, particles.T[idx]);
		return;
	}

	float_t dt2 = dt * dt;

	vec2_t  fN = -tool_consts.slave_mass * gN * n_ / dt2 * tool_consts.contact_alpha;	//nianfei 2009
	vec2_t  fT(0., 0.);

	vec2_t vm = vec2_t(vel_t.x, vel_t.y);
	//vec2_t v = vs - vm;
	float2_t v = make_float2_t(vs.x - vel_t.x, vs.y - vel_t.y);
	//vec2_t vr = v - v * n_;	//relative velocity
	vec2_t vr;
	float_t vel_dot_nor = v.x * n_.x + v.y * n_.y;
	vr.x = v.x - n_.x * vel_dot_nor;
	vr.y = v.y - n_.y * vel_dot_nor;
    float_t v_rel_mag = sqrtf(vr.x * vr.x + vr.y * vr.y);

	//---- lsdyna theory manual ----
	glm::dvec2 fricold(particles.ft[idx].x, particles.ft[idx].y);

	glm::dvec2 kdeltae = tool_consts.contact_alpha * tool_consts.slave_mass * vr / dt;
	//glm::dvec2 kdeltae =  tool_consts.slave_mass * vr / dt;
	float_t cof = tool_consts.mu;
#ifdef VELOCITY_FRI
	//float_t cutting_speed = sqrtf(vel_t.x * vel_t.x + vel_t.y * vel_t.y);
	cof = tool_consts.mu * pow(v_rel_mag *6*1e5, -tool_consts.exp_co_vel); // unit of velocity: m/min
#endif

#ifdef TEMP_FRI
	float_t theta = (particles.T[idx] - tool_consts.T0) / (tool_consts.T_melt - tool_consts.T0);
	cof = tool_consts.mu *(1 - pow(theta, tool_consts.exp_co_temp));
#endif
#ifdef PRES_FRI
    float_t length = sqrt(tool_consts.slave_mass / particles.rho[idx]);
	float_t pressure_calculated = sqrtf(fN.x * fN.x + fN.y * fN.y) / length * 100.;
    cof = tool_consts.mu * pow(pressure_calculated, -tool_consts.exp_co_pres); // unit of pressure: GPa
	cof = fmin(tool_consts.up_limit, cof);
	cof = fmax(tool_consts.lo_limit, cof);
#endif

	float_t fy = cof * glm::length(fN);	//coulomb friction
	glm::dvec2 fstar = fricold - kdeltae;



#ifdef SHEAR_LIMIT_FRI // 2022.07.18 updated
    float_t eps_pl     = particles.eps_pl[idx];
	float_t eps_pl_dot = particles.eps_pl_dot[idx];
	float_t T          = particles.T[idx];

    float_t k_shear = sigma_yield(johnson_cook, eps_pl, eps_pl_dot, T);
	k_shear = k_shear / 1.73205081;
	
	float_t len = sqrt(tool_consts.slave_mass / particles.rho[idx]);
    /*
	// Shear stress limit, for the plastic contact, this calculation way should also be correct
	float4_t S = particles.S[idx];
	float_t S_xx = S.x;
	float_t S_yy = S.z;
	float_t S_zz = S.w;
	float_t S_xy = S.y;
	float_t k_shear = sqrt(1.5 * (S_xx * S_xx + S_yy * S_yy + S_zz * S_zz + 2. * S_xy * S_xy)) / 1.73205081;
	*/
    if (k_shear > 0.008) printf("shear flow stress is %d \n", k_shear);

	float_t m_factor = tool_consts.shear_limit;
	float_t F_shear = m_factor * len * k_shear;
    
	if (F_shear < fy) {
		fy = F_shear;
	}
	
#endif

	if (glm::length(fstar) > fy) {
		fT = fy * fstar / glm::length(fstar);
	}
	else {
		fT = fstar;
	}

    //float_t f_normal = sqrtf(fN.x * fN.x + fN.y * fN.y);

	particles.fc[idx].x = fN.x;
	particles.fc[idx].y = fN.y;
	particles.ft[idx].x = fT.x;
	particles.ft[idx].y = fT.y;

	particles.n[idx].x = n_.x;
	particles.n[idx].y = n_.y;
#endif
}

__global__ void do_compute_contact_forces_FEM_tool_v3_s2(int N, particle_gpu particles, segment_FEM_gpu segments,
	const float_t* __restrict__ T_nodes, float_t dt)
{
#ifdef USE_FEM_TOOL
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (particles.blanked[idx]==1.) return;
	//if (particles.eps_pl[idx] > 12. )  return;

	int seg_no = particles.on_seg[idx];
	if (seg_no == -1) return;

	int left_no = segments.marks[seg_no].z;
	int right_no = segments.marks[seg_no].w;

	float_t T_ = particles.T[idx];
	float_t rho_ = particles.rho[idx];
	float_t T_seg = 0.5 * (T_nodes[left_no] + T_nodes[right_no]);
	float_t delta_T = T_seg - T_;
	float_t length = sqrtf(tool_consts.slave_mass / rho_);
	__syncthreads();

	atomicAdd(&segments.heat_exchange[seg_no].x, T_ * length);
	atomicAdd(&segments.heat_exchange[seg_no].y, length);
	float_t delta_T_part = delta_T * tool_consts.h * length / thermals_wp.cp / tool_consts.slave_mass * dt;
	if (delta_T_part < 0){
        particles.T[idx] += fmax(delta_T_part, delta_T);
	} else{
        particles.T[idx] += fmin(delta_T_part, delta_T);
	}

#endif
}

__global__ void do_heat_convection_to_particle(int N, float_t* T, float_t* T_t, float_t dt) {
	 int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	T[idx] += dt * T_t[idx];

}

__global__ void do_contmech_continuity(const float_t *__restrict__ rho, const float4_t *__restrict__ v_der, const float_t *in_tool,
		float_t *__restrict__ rho_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	double rhoi  = rho[pidx];
	float4_t vi_der = v_der[pidx];

	float_t vx_x = vi_der.x;
	float_t vy_y = vi_der.w;

	rho_t[pidx] = -rhoi*(vx_x + vy_y);
}

__global__ void do_contmech_momentum(const float4_t *__restrict__ S_der, const float2_t *__restrict__ fc, const float2_t *__restrict__ ft, const float_t *in_tool,
		float2_t *__restrict__ vel_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float_t mass = physics.mass;

	float4_t Si_der = S_der[pidx];
	float2_t fci    = fc[pidx];
	float2_t fti    = ft[pidx];
	float2_t veli_t = vel_t[pidx];

	float_t Sxx_x = Si_der.x;
	float_t Sxy_y = Si_der.y;
	float_t Sxy_x = Si_der.z;
	float_t Syy_y = Si_der.w;

	float_t fcx   = fci.x;
	float_t fcy   = fci.y;

	float_t ftx   = fti.x;
	float_t fty   = fti.y;

	//redundant mult and div by rho elimnated in der stress
	veli_t.x += Sxx_x + Sxy_y + fcx / mass + ftx / mass;
	veli_t.y += Sxy_x + Syy_y + fcy / mass + fty / mass;

	vel_t[pidx] = veli_t;

}

__global__ void do_contmech_advection(const float2_t *__restrict__ vel, const float_t *in_tool,
		float2_t *__restrict__ pos_t, int N) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N) return;
	if (in_tool[pidx]) return;

	float2_t veli   = vel[pidx];
	float2_t posi_t = pos_t[pidx];

	float2_t posi_t_new;
	posi_t_new.x = posi_t.x + veli.x; // XSPH + v
	posi_t_new.y = posi_t.y + veli.y;

	pos_t[pidx] = posi_t_new;
}

__global__ void do_plasticity_johnson_cook(particle_gpu particles, float_t dt) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.tool_particle[pidx]) return;
	if (particles.blanked[pidx]==1.) return; 

	float_t mu = physics.G;

	float4_t S = particles.S[pidx];
	float_t Strialxx = S.x;
	float_t Strialyy = S.z;
	float_t Strialzz = S.w;
	float_t Strialxy = S.y;

	float_t norm_Strial = sqrtf(Strialxx*Strialxx + Strialyy*Strialyy + Strialzz*Strialzz + 2. * Strialxy*Strialxy);

	float_t p = particles.p[pidx];

	float_t cxx = Strialxx - p;
	float_t cyy = Strialyy - p;
	float_t czz = Strialzz - p;
	float_t cxy = Strialxy;

	float_t eps_pl     = particles.eps_pl[pidx];
	float_t eps_pl_dot = particles.eps_pl_dot[pidx];
	float_t T          = particles.T[pidx];

	float_t svm = sqrtf((cxx*cxx + cyy*cyy + czz*czz) - cxx * cyy - cxx * czz - cyy * czz + 3.0 * cxy * cxy);
	float_t sigma_Y = sigma_yield(johnson_cook, eps_pl, eps_pl_dot, T);

	// elastic case
	if (svm < sigma_Y) {
		particles.eps_pl_dot[pidx] = 0.;
		return;
	}

	float_t delta_lambda = solve_secant(johnson_cook, fmax(eps_pl_dot*dt*sqrt(2./3.), 1e-8), 1e-6,
			norm_Strial, eps_pl, T, dt, physics.G);

	float_t eps_pl_new = eps_pl + sqrtf(2.0/3.0) * fmaxf(delta_lambda,0.);
	float_t eps_pl_dot_new = sqrtf(2.0/3.0) *  fmaxf(delta_lambda,0.) / dt;

	particles.eps_pl[pidx] = eps_pl_new;
	particles.eps_pl_dot[pidx] = eps_pl_dot_new;

	float4_t S_new;
	S_new.x = Strialxx - Strialxx/norm_Strial*delta_lambda*2.*mu;
	S_new.z = Strialyy - Strialyy/norm_Strial*delta_lambda*2.*mu;
	S_new.w = Strialzz - Strialzz/norm_Strial*delta_lambda*2.*mu;
	S_new.y = Strialxy - Strialxy/norm_Strial*delta_lambda*2.*mu;

	particles.S[pidx] = S_new;

	//plastic work to heat
	if (thermals_wp.tq != 0.) {
		float_t delta_eps_pl = eps_pl_new - eps_pl;
		float_t sigma_Y = sigma_yield(johnson_cook, eps_pl_new, eps_pl_dot_new, T);
		float_t rho = particles.rho[pidx];
		particles.T[pidx] += thermals_wp.tq/(thermals_wp.cp*rho)*delta_eps_pl*sigma_Y;
	}

	if (particles.T[pidx] > johnson_cook.Tmelt) {	// Maximaltemperatur auf Schmelztemperatur limitieren; HK, Do, 13.02.2020
		particles.T[pidx] = johnson_cook.Tmelt;
		particles.blanked[pidx] = 1.;
	}
}

__global__ void do_boundary_conditions_thermal(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	if (particles.fixed[pidx] == 1.) {
		particles.T[pidx] = thermals_wp.T_init;
	}
}

__global__ void do_boundary_conditions(particle_gpu particles) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;
	if (particles.tool_particle[pidx]) return;
	

	if (particles.fixed[pidx]) {
		particles.vel[pidx].x = 0.;
		particles.vel[pidx].y = 0.;
		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.pos_t[pidx].x = 0.;
		particles.pos_t[pidx].y = 0.;
		particles.vel_t[pidx].x = 0.;
		particles.vel_t[pidx].y = 0.;
	}
}

__device__ __forceinline__ bool isnaninf(float_t val) {
	return isnan(val) || isinf(val);
}

__global__ void do_invalidate(particle_gpu particles, int global_step) {
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N) return;

	if (particles.blanked[pidx] == 1.) {
		return;
	}

	bool invalid = false;

	invalid = invalid || isnaninf(particles.pos_t[pidx].x);
	invalid = invalid || isnaninf(particles.pos_t[pidx].y);

	invalid = invalid || isnaninf(particles.vel_t[pidx].x);
	invalid = invalid || isnaninf(particles.vel_t[pidx].y);

	invalid = invalid || isnaninf(particles.S_t[pidx].x);
	invalid = invalid || isnaninf(particles.S_t[pidx].y);
	invalid = invalid || isnaninf(particles.S_t[pidx].z);
	invalid = invalid || isnaninf(particles.S_t[pidx].w);

	invalid = invalid || isnaninf(particles.rho_t[pidx]);
	invalid = invalid || isnaninf(particles.T_t[pidx]);

#ifdef USE_FEM_TOOL
	invalid = invalid || isnaninf(particles.on_seg[pidx]);
#endif // USE_FEM_TOOL


	if (invalid) {
		particles.blanked[pidx] = 1.;
		printf("invalidated particle %d due to nan at %f %f, %f %f, %f %f %f %f %f %f %f %f %d\n",
				pidx, particles.pos[pidx].x, particles.pos[pidx].y,
				particles.pos_t[pidx].x, particles.pos_t[pidx].y,
				particles.vel_t[pidx].x, particles.vel_t[pidx].x,
				particles.S_t[pidx].x, particles.S_t[pidx].y, particles.S_t[pidx].z, particles.S_t[pidx].w, particles.rho_t[pidx], particles.T_t[pidx], particles.on_seg[pidx]);
	}
}

//---------------------------------------------------------------------

// float2 + struct
struct add_float2 {
    __device__ float2_t operator()(const float2_t& a, const float2_t& b) const {
        float2_t r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        return r;
    }
 };

void material_eos(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_material_eos<<<dG,dB>>>(particles->rho, particles->p, particles->tool_particle, particles->N);
}

void corrector_artificial_stress(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_corrector_artificial_stress<<<dG,dB>>>(particles->rho, particles->p, particles->S, particles->tool_particle, particles->R, particles->N);
}

void material_stress_rate_jaumann(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_material_stress_rate_jaumann<<<dG,dB>>>(particles->v_der, particles->S, particles->tool_particle, particles->S_t, particles->N);
}

void contmech_continuity(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_contmech_continuity<<<dG,dB>>>(particles->rho, particles->v_der, particles->tool_particle, particles->rho_t, particles->N);
}

void contmech_momentum(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_contmech_momentum<<<dG,dB>>>(particles->S_der, particles->fc, particles->ft, particles->tool_particle, particles->vel_t, particles->N);
}

void contmech_advection(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_contmech_advection<<<dG,dB>>>(particles->vel, particles->tool_particle, particles->pos_t, particles->N);
}

void plasticity_johnson_cook(particle_gpu *particles) {
	if (!m_plastic) return;
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_plasticity_johnson_cook<<<dG,dB>>>(*particles, global_dt);
}

void perform_boundary_conditions_thermal(particle_gpu *particles) {
	if (!m_thermal) return;
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_boundary_conditions_thermal<<<dG,dB>>>(*particles);
}

void perform_boundary_conditions(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_boundary_conditions<<<dG,dB>>>(*particles);
}

void debug_check_valid(particle_gpu *particles) {
	thrust::device_ptr<float2_t> t_pos(particles->pos);
	float2_t ini;
	ini.x = 0.;
	ini.y = 0.;
	ini = thrust::reduce(t_pos, t_pos + particles->N, ini, add_float2());

	if (isnan(ini.x) || isnan(ini.y)) {
		printf("nan found!\n");
		exit(-1);
	}
}

void debug_invalidate(particle_gpu *particles) {
	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_invalidate<<<dG,dB>>>(*particles, global_step);
}

void actions_setup_physical_constants(phys_constants physics_h) {
	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_corrector_constants(corr_constants correctors_h) {
	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_johnson_cook_constants(joco_constants johnson_cook_h) {
	cudaMemcpyToSymbol(johnson_cook, &johnson_cook_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
	m_plastic = true;
}

void actions_setup_thermal_constants_wp(trml_constants thermal_h) {
	cudaMemcpyToSymbol(thermals_wp, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	if (thermal_h.tq != 0.) {
		printf("considering generation of heat due to plastic work\n");
	}

	if (thermal_h.eta != 0.) {
		printf("considering that friction generates heat\n");
		m_fric_heat_gen = true;
	}
}

void actions_setup_thermal_constants_tool(trml_constants thermal_h) {
	cudaMemcpyToSymbol(thermals_tool, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);

	if (thermal_h.alpha != 0.) {
		m_thermal = true;
	}
}

void actions_setup_tool_constants(tool_constants tc_h)
{
	cudaMemcpyToSymbol(tool_consts, &tc_h, sizeof(tool_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_wear_constants(wear_constants wear)
{
	cudaMemcpyToSymbol(wear_consts, &wear, sizeof(wear_constants), 0, cudaMemcpyHostToDevice);
}

void material_fric_heat_gen(particle_gpu *particles, vec2_t vel) {
	if (!m_fric_heat_gen) {
		return;
	}

	dim3 dG((particles->N + BLOCK_SIZE-1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_material_fric_heat_gen<<<dG,dB>>>(particles->vel, particles->ft, particles->n,
			particles->tool_particle, particles->T, make_float2_t(vel.x, vel.y), global_dt, particles->N);
}

void material_fric_heat_gen_FEM_v2(particle_gpu* particles, segment_FEM_gpu* segments, vec2_t vel, tool_FEM* m_tool) {
	cudaEvent_t fric_FEM;
	cudaEventCreate(&fric_FEM);
	dim3 dG((particles->N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
#ifdef USE_FEM_TOOL
#ifdef USE_MATERIAL_STRESS
    do_material_fric_heat_gen_FEM_v3 << <dG, dB >> > (*particles, *segments, make_float2_t(vel.x, vel.y), m_tool->FEM_solver->m_T, global_dt, particles->N);
#else
	do_material_fric_heat_gen_FEM_v21 << <dG, dB >> > (*particles, *segments, make_float2_t(vel.x, vel.y), m_tool->FEM_solver->m_T, global_dt, particles->N);
#endif
#endif
	cudaEventSynchronize(fric_FEM);
	check_cuda_error("friction_heat_generation_FEM\n");
}

void compute_contact_forces_FEM_tool_v3(particle_gpu* particles, segment_FEM_gpu* segments, tool_FEM* m_tool)
{
	int seg_num = m_tool->cutting_segment_size;
	vec2_t vel = m_tool->get_vel();

	dim3 dG((particles->N + BLOCK_SIZE / 2 - 1) / BLOCK_SIZE * 2);
	dim3 dB(BLOCK_SIZE / 2);
	cudaEvent_t contact_FEM;
	cudaEventCreate(&contact_FEM);
	do_compute_contact_forces_FEM_tool_v3_s1 << <dG, dB >> > (particles->N, *particles, *segments, m_tool->tl_point_ref, m_tool->ref_dist, vel, global_dt, seg_num);
	cudaEventSynchronize(contact_FEM);
	check_cuda_error("compute_contact_forces_FEM_tool_step1\n");

	do_compute_contact_forces_FEM_tool_v3_s2 << <dG, dB >> > (particles->N, *particles, *segments, m_tool->FEM_solver->m_T, global_dt);
	cudaEventSynchronize(contact_FEM);
	check_cuda_error("compute_contact_forces_FEM_tool_step2\n");
}

void heat_convection_to_particle(particle_gpu* particles, float_t dt)
{
	dim3 dG((particles->N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dB(BLOCK_SIZE);
	do_heat_convection_to_particle << <dG, dB >> > (particles->N, particles->T, particles->T_t, dt);
	
}

void wear_adjustment(tool_FEM* m_tool)
{
	//m_tool->mapping_data_copy();    // not necessary here
	m_tool->FEM_solver->mesh_retreat_wear(m_tool->mesh_GPU);
	m_tool->nodal_shift_wear(global_dt);
	//m_tool->nodal_value_interpolation();
	m_tool->FEM_solver->mesh_resume_wear(m_tool->mesh_GPU);
	m_tool->segment_wear_update();
	m_tool->correct_flank_contact();
	m_tool->FEM_solver->inverse_mass_matrix();
	// i need to interpolate the temperature value as well
}

void remesh_low_level(tool_FEM* m_tool, int freq)
{
	m_tool->mapping_data_copy();
	m_tool->mesh_shift(freq);
	m_tool->FEM_solver->reset_FEM_matrix(m_tool->mesh_GPU);
	m_tool->FEM_solver->stiffness_mass_matrix_construction(m_tool->mesh_GPU);
	m_tool->FEM_solver->dirichlet_bdc(m_tool->mesh_GPU);
	m_tool->FEM_solver->inverse_mass_matrix();
	m_tool->nodal_value_interpolation();
}

void remesh_gmsh(tool_FEM* m_tool, int freq)
{
	m_tool->mapping_data_copy();
	m_tool->remesh_read_segments();
	m_tool->read_remesh_gmsh(); 

}