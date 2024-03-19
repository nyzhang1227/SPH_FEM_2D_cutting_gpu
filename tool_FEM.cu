// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "tool_FEM.h"
#include "tool_gpu.cuh"
#include "constants_structs.h"

#include <cuda_runtime_api.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

__device__ float_t flank_retreat;
__device__ float_t flank_contact;
__device__ wear_constants wear_para;


static vec2_t solve_quad_FEM(float_t a, float_t b, float_t c) {
	float_t x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
	float_t x2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);

	return vec2_t(x1, x2);
}

static float_t myatan2_FEM(float_t y, float_t x) {
	float_t t = atan2(y, x);
	if (t > 0.) {
		return t;
	}
	else {
		return t + 2 * M_PI;
	}
}

__device__ float_t distance_segment(float2_t left, float2_t right) {
	return sqrtf((left.x - right.x) * (left.x - right.x) + (left.y - right.y) * (left.y - right.y));
}

__device__ float_t cross(float2_t a, float2_t b) {
	return a.x * b.y - a.y * b.x;
}

__device__ float2_t nodal_shift(float2_t n, float_t dist)
{
	float2_t point_sh;
	point_sh.x = n.x * dist;
	point_sh.y = n.y * dist;
	return point_sh;
}

__device__ float_t value_interpolation_in_layer_element(int node_no, int N, float2_t x_, const int4* node_num, const float2_t* pos, const float_t* T) {

	// barycentric coordinate methods
	// https://math.stackexchange.com/questions/51326/determining-if-an-arbitrary-point-lies-inside-a-triangle-defined-by-three-points
	float_t T_new;
	int nA, nB, nC;
	float2_t xA, xB, xC, xP, xBC;
	float_t d, wA, wB, wC, w, TA, TB, TC;
	for (int i = 0; i < N; i++) {
		nA = node_num[i].x;
		nB = node_num[i].y;
		nC = node_num[i].z;
		//if (nA == node_no || nB == node_no || nC == node_no) {
			xA = pos[nA];
			xB.x = pos[nB].x - xA.x;
			xB.y = pos[nB].y - xA.y;
			xC.x = pos[nC].x - xA.x;
			xC.y = pos[nC].y - xA.y;
			xP.x = x_.x - xA.x;
			xP.y = x_.y - xA.y;
			xBC.x = xB.x - xC.x;
			xBC.y = xB.y - xC.y;
			d = cross(xB, xC);  // This is not the cross product!


			if (abs(d) >1e-13) {
				wA = cross(xB, xC) + cross(xP, xBC);
				wB = cross(xP, xC);   // (xP, xCA)
				wC = -cross(xP, xB);  // (xP, xAB)
				//w = wA + wB + wC;
				if (wA >= 0 && wB >= 0 && wC >= 0 && wA <= 1 && wB <= 1 && wC <=1) { // inside the element
				    
					TA = T[nA];
					TB = T[nB];
					TC = T[nC];
					if (node_no == 533) {
				    //printf("xA: %f, %f, xB: %f, %f, xC: %f, %f, xP: %f, %f \n", xA.x, xA.y, pos[nB].x, pos[nB].y, pos[nC].x, pos[nC].y, x_.x, x_.y);
			        }
					T_new = (TA * wA + TB * wB + TC * wC) / d;
					//printf("node: %d, element:%d, %d, %d, xA: %f, %f, xB: %f, %f, xC: %f, %f, xP: %f, %f, T_old: %f, T_new:%f \n", node_no, xA.x, xA.y, nA, nB, nC, pos[nB].x, pos[nB].y, pos[nC].x, pos[nC].y, x_.x, x_.y, T[node_no], T_new);
					return T_new;
				}
			}

		//}

	}
	// if the interpolation failed
	//printf("Interpolation of node %d failed. \n", node_no);
	return T[node_no];
}

__global__ void do_tool_update_FEM_gpu(int N, float2_t* position, vec2_t vel, float_t dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	position[idx].x += vel.x * dt;
	position[idx].y += vel.y * dt + flank_retreat;

}

__global__ void do_segments_update(int N, float2_t* left, float2_t* right, float2_t* heat_exchange, float_t* fric_heat, vec2_t vel, float_t dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	float_t seg_l_x = left[idx].x;
	float_t seg_l_y = left[idx].y;
	float_t seg_r_x = right[idx].x;
	float_t seg_r_y = right[idx].y;
	left[idx].x = seg_l_x + vel.x * dt;
	left[idx].y = seg_l_y + vel.y * dt + flank_retreat;
	right[idx].x = seg_r_x + vel.x * dt;
	right[idx].y = seg_r_y + vel.y * dt + flank_retreat;

	heat_exchange[idx].x = 0;
	heat_exchange[idx].y = 0;
	
	fric_heat[idx] = 0.;
	//fric_heat[idx].y = 0.;

}

__global__ void do_reset_vector(int N, float_t* vector)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	vector[idx] = 0.;

}

__global__ void do_reset_vector2(int N, float2_t* vector)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	vector[idx].x = 0.;
	vector[idx].y = 0.;

}

__global__ void do_reset_vector4(int N, float4_t* vector)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	vector[idx].x = 0.;
	vector[idx].y = 0.;
	vector[idx].z = 0.;
	vector[idx].w = 0.;

}

__global__ void do_nodal_shift_wear_01(int N, float_t dt, segment_FEM_gpu segments_FEM, mesh_GPU_m mesh_GPU) 
{
#ifdef WEAR_NODE_SHIFT
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	float4_t wear = segments_FEM.physical_para[idx];
	if (wear.x == 0.) return;
	float_t wear_value;
	//float_t length = distance_segment(segments_FEM.left[idx], segments_FEM.right[idx]);
	float_t pressure = wear.y / wear.x;
	float_t vel = wear.z / wear.x;
	float_t temperature = wear.w / wear.x;

#ifdef USE_ABRASIVE_WEAR
    wear_value = dt * wear_para.A * vel;
#endif
#ifdef USE_SUN_WEAR
    float_t p_ui = pressure * 1e5;  // MPa
	float_t T_cel = temperature - 273.15;  // °C
	float_t wear_rate_ui = p_ui * (1.57 * log(T_cel) - 1.27 * p_ui / 1000. - 9.849) *(exp(wear_para.C * T_cel - 7.5) * pow(p_ui, -wear_para.D));
    wear_value = dt * wear_rate_ui * 1e-10;
#endif
#ifdef USE_USUI_WEAR  
	wear_value = dt * wear_para.C * pressure * vel * exp(-wear_para.D / temperature);
#endif
#ifdef USE_MALA_WEAR  // wrong, in this model wear volume is used
	wear_value = dt * wear_para.C * vel * exp(-wear_para.D / (temperature-273.15));
#endif
#ifdef  USE_DIFFUSIVE_WEAR
    wear_value = dt * wear_para.C * exp(-wear_para.D / temperature);
#endif
#ifdef USE_ZANGER_WEAR  
    wear_value = dt * (wear_para.A * vel * exp(-wear_para.B / temperature) + wear_para.C * pressure * exp(-wear_para.D / temperature));
#endif
    //if (wear_value > 1e-9) {printf("large wear: %f, temp: %f, pressure: %f, v_rel: %f \n", wear_value, temperature,pressure, vel);}
	//printf("large wear: %f, temp: %f, pressure: %f, v_rel: %f \n", wear_value, temperature,pressure, vel);
	segments_FEM.wear_nodes[idx] = wear_value / dt;    // wear rate for the segments
	float2_t delta_node = nodal_shift(segments_FEM.n[idx], wear_value);
	int left = segments_FEM.marks[idx].z;
	int right = segments_FEM.marks[idx].w;
    int4 node_mark = mesh_GPU.node_num[segments_FEM.marks[idx].x];
	int third;
	if (node_mark.x != left && node_mark.x != right) third = node_mark.x;
	if (node_mark.y != left && node_mark.y != right) third = node_mark.y;
	if (node_mark.z != left && node_mark.z != right) third = node_mark.z;

	atomicAdd(&mesh_GPU.position[left].x, delta_node.x);
	atomicAdd(&mesh_GPU.position[left].y, delta_node.y);
	atomicAdd(&mesh_GPU.position[right].x, delta_node.x);
	atomicAdd(&mesh_GPU.position[right].y, delta_node.y);
	
	atomicAdd(&mesh_GPU.position[third].x, delta_node.x * 0.5);
	atomicAdd(&mesh_GPU.position[third].y, delta_node.y * 0.5);
	
#endif
}

__global__ void do_nodal_shift_wear_02(int N, float_t dt, segment_FEM_gpu segments_FEM, mesh_GPU_m mesh_GPU)
{
	// without shift the internal node
#ifdef WEAR_NODE_SHIFT

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	float4_t wear = segments_FEM.physical_para[idx];
	if (wear.x == 0. ) return;

	float_t wear_value;
	//float_t length = distance_segment(segments_FEM.left[idx], segments_FEM.right[idx]);
	float_t pressure = wear.y / wear.x;
	float_t vel = wear.z / wear.x;
    float_t temperature = wear.w / wear.x;
#ifdef RESTRICT_WEAR_TEMP
    vel = fmin(vel, 0.008);
    pressure = fmin(pressure, 0.1);
	temperature = fmin(temperature, 1500.);
#endif


#ifdef USE_ABRASIVE_WEAR
    wear_value = dt * wear_para.A * vel;
#endif
#ifdef USE_SUN_WEAR
    float_t p_ui = pressure * 1e5;  // MPa
	float_t T_cel = temperature - 273.15;  // °C
	float_t wear_rate_ui = p_ui * (1.57 * log(T_cel) - 1.27 * p_ui / 1000. - 9.849) *(exp(wear_para.C * T_cel - 7.5) * pow(p_ui, -wear_para.D));
    wear_value = dt * wear_rate_ui * 1e-10;
#endif
#ifdef USE_USUI_WEAR  
	wear_value = dt * wear_para.C * pressure * vel * exp(-wear_para.D / temperature);
#endif
#ifdef USE_MALA_WEAR  // wrong, in this model wear volume is used
	wear_value = dt * wear_para.C * vel * exp(-wear_para.D / (temperature-273.15));
#endif
#ifdef  USE_DIFFUSIVE_WEAR
    wear_value = dt * wear_para.C * exp(-wear_para.D / temperature);
#endif
#ifdef USE_ZANGER_WEAR  
    wear_value = dt * (wear_para.A * vel * exp(-wear_para.B / temperature) + wear_para.C * pressure * exp(-wear_para.D / temperature));
#endif
#ifdef USE_WHOLE_WEAR  
	wear_value = dt * (wear_para.A * pressure * vel * exp(-wear_para.B / temperature) +  wear_para.C * exp(-wear_para.D / temperature));
#endif


	segments_FEM.wear_nodes[idx] = wear_value / dt;    // wear rate for the segments
	float2_t delta_node = nodal_shift(segments_FEM.n[idx], wear_value);
	int left = segments_FEM.marks[idx].z;
	int right = segments_FEM.marks[idx].w;
	/*
	if (wear_value > 3e-6) {
		printf("large wear: %f, temp: %f, pressure: %f, v_rel: %f at location x %f, y %f\n", wear_value, temperature,pressure, vel, mesh_GPU.position[left].x, mesh_GPU.position[left].y);
		return;
	}
	*/

	atomicAdd(&mesh_GPU.position[left].x, delta_node.x);
	atomicAdd(&mesh_GPU.position[left].y, delta_node.y);
	atomicAdd(&mesh_GPU.position[right].x, delta_node.x);
	atomicAdd(&mesh_GPU.position[right].y, delta_node.y);

#endif
}

__global__ void do_nodal_shift_wear_smooth_temp(int N, float_t dt, segment_FEM_gpu segments_FEM, mesh_GPU_m mesh_GPU, float_t* T)
{
	// without shift the internal node
#ifdef WEAR_NODE_SHIFT

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	float4_t wear = segments_FEM.physical_para[idx];
	if (wear.x == 0. ) return;
    
	int left = segments_FEM.marks[idx].z;
	int right = segments_FEM.marks[idx].w;

	//float_t length = distance_segment(segments_FEM.left[idx], segments_FEM.right[idx]);
	float_t pressure = wear.y / wear.x;
	float_t vel = wear.z / wear.x;
	float_t temperature_left = T[left];
	float_t temperature_right = T[right];
        
	if (vel > 0.008) {
		printf("Large velocity %f m/s \n", vel*10000);
		return;
	}

#ifdef RESTRICT_WEAR_TEMP
	
	temperature_left = fmin(temperature_left, 1350.);
	temperature_right = fmin(temperature_right, 1350.);

    /*
    if (pressure > 0.035) 
	{
		//printf("Large pressure %f GPa \n", pressure*100);
		pressure = 0.035;
	}
	*/
	//if (vel > 0.0009) printf("Large velocity %f m/s \n", vel*10000);
    vel = fmin(vel, 0.008);
    pressure = fmin(pressure, 0.04);
#endif

	float_t wear_value_left;
	float_t wear_value_right;

#ifdef USE_ABRASIVE_WEAR
	wear_value_left = dt * wear_para.A * vel;
	wear_value_right = dt * wear_para.A * vel;
#endif
#ifdef USE_ABRASIVE_NEW
    float_t pressure_scale = fmin(pressure / wear_para.B, 1.0);
	wear_value_left = dt * wear_para.A * vel * pressure_scale;
	wear_value_right = dt * wear_para.A * vel * pressure_scale;
#endif
#ifdef USE_SUN_WEAR
	float_t p_ui = pressure * 1e5;  // MPa
	float_t T_cel = temperature - 273.15;  // °C
	float_t wear_rate_ui = p_ui * (1.57 * log(T_cel) - 1.27 * p_ui / 1000. - 9.849) * (exp(wear_para.C * T_cel - 7.5) * pow(p_ui, -wear_para.D));
	wear_value_left = dt * wear_rate_ui * 1e-10;
	wear_value_right = dt * wear_rate_ui * 1e-10;
#endif
#ifdef USE_USUI_WEAR  
	wear_value_left = dt * wear_para.C * pressure * vel * exp(-wear_para.D / temperature_left);
	wear_value_right = dt * wear_para.C * pressure * vel * exp(-wear_para.D / temperature_right);
#endif
#ifdef USE_USUI_NEW
    float_t pressure_term = fmin(1.0, pressure / wear_para.B);  
	wear_value_left = dt * wear_para.C * pressure_term * vel * exp(-wear_para.D / temperature_left);
	wear_value_right = dt * wear_para.C * pressure_term * vel * exp(-wear_para.D / temperature_right);
#endif
#ifdef USE_MALA_WEAR  // wrong, in this model wear volume is used
	wear_value_left = dt * wear_para.C * vel * exp(-wear_para.D / (temperature_left - 273.15));
	wear_value_right = dt * wear_para.C * vel * exp(-wear_para.D / (temperature_right - 273.15));
#endif
#ifdef  USE_DIFFUSIVE_WEAR
	wear_value_left = dt * wear_para.C * exp(-wear_para.D / temperature_left);
	wear_value_right = dt * wear_para.C * exp(-wear_para.D / temperature_right);
#endif
#ifdef USE_ZANGER_WEAR  
	wear_value_left = dt * (wear_para.A * vel * exp(-wear_para.B / temperature_left) + wear_para.C * pressure * exp(-wear_para.D / temperature_left));
	wear_value_right = dt * (wear_para.A * vel * exp(-wear_para.B / temperature_right) + wear_para.C * pressure * exp(-wear_para.D / temperature_right));
#endif
#ifdef USE_RECH_WEAR  
    float_t pv = pressure * vel;
	pv = fmin(8e-6, pv);
	wear_value_left = dt * wear_para.A *  exp(wear_para.B *pv);
	wear_value_right = dt * wear_para.A *  exp(wear_para.B *pv);
#endif
#ifdef USE_WHOLE_WEAR  
	wear_value_left = dt * (wear_para.A * pressure * vel * exp(-wear_para.B / temperature_left) +  wear_para.C * exp(-wear_para.D / temperature_left));
	wear_value_right = dt * (wear_para.A * pressure * vel * exp(-wear_para.B / temperature_right) + wear_para.C * exp(-wear_para.D / temperature_right));
#endif
#ifdef USE_ABRASIVE_NEW_USUI
    float_t pressure_scale = fmin(pressure / wear_para.B, 1.0);
    wear_value_left = dt * (wear_para.A * vel * pressure_scale +  wear_para.C * pressure * vel * exp(-wear_para.D / temperature_left));
	wear_value_right = dt * (wear_para.A * vel * pressure_scale +  wear_para.C * pressure * vel * exp(-wear_para.D / temperature_right));
#endif
#ifdef USE_ABRASIVE_USUI
    wear_value_left = dt * (wear_para.A * vel  +  wear_para.C * pressure * vel * exp(-wear_para.D / temperature_left));
	wear_value_right = dt * (wear_para.A * vel +  wear_para.C * pressure * vel * exp(-wear_para.D / temperature_right));
#endif
#ifdef USE_USUI_RAKE_FLANK
    if (abs(segments_FEM.n[idx].x) > 0.85) {
        wear_value_left = dt * wear_para.C * pressure * vel * exp(-wear_para.D / temperature_left);
	    wear_value_right = dt * wear_para.C * pressure * vel * exp(-wear_para.D / temperature_right);
	} else {
		wear_value_left = dt * wear_para.A * pressure * vel * exp(-wear_para.B / temperature_left);
	    wear_value_right = dt * wear_para.A * pressure * vel * exp(-wear_para.B / temperature_right);
	}
#endif

    
	
    if (pressure > 0.025 && (wear_value_left / dt > 1.5e-5 || wear_value_right / dt > 1.5e-5 )) {
		//printf("Too large wear rate %.4e um/s detected! Pressure is %f GPa, velocity is %f m/s, temperature is %f, %f, location at %f, %f \n", fmax(wear_value_left, wear_value_right) / dt * 1e10, pressure * 100, vel*10000, temperature_left, temperature_right, mesh_GPU.position[left].x, mesh_GPU.position[left].y);
        //return; // 871071 this one is also commented
	}
/*
	if (wear_value_left / dt > 3.0e-4 || wear_value_right / dt > 3.0e-4 ) {
		printf("Too large wear rate %.4e um/s detected! Pressure is %f GPa, velocity is %f m/s, temperature is %f, %f, location at %f, %f \n", fmax(wear_value_left, wear_value_right) / dt * 1e10, pressure * 100, vel*10000, temperature_left, temperature_right, mesh_GPU.position[left].x, mesh_GPU.position[left].y);
        return;
	}
*/
	segments_FEM.wear_nodes[idx] = (wear_value_left + wear_value_right) / dt;    // wear rate for the segments
	float2_t delta_node_left = nodal_shift(segments_FEM.n[idx], wear_value_left);
	float2_t delta_node_right = nodal_shift(segments_FEM.n[idx], wear_value_right);

	atomicAdd(&mesh_GPU.position[left].x, delta_node_left.x);
	atomicAdd(&mesh_GPU.position[left].y, delta_node_left.y);
	atomicAdd(&mesh_GPU.position[right].x, delta_node_right.x);
	atomicAdd(&mesh_GPU.position[right].y, delta_node_right.y);

#endif
}

__global__ void do_nodal_value_interpolation_old(int N, int ele_size, mesh_GPU_m mesh_GPU, const float_t* m_T_mapping, float_t* m_T) 
{
#ifdef WEAR_NODE_SHIFT
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	    float2_t x_ = mesh_GPU.position[idx];
    	float_t T_map = value_interpolation_in_layer_element(idx, ele_size, x_, mesh_GPU.node_num_mapping,  mesh_GPU.position_mapping, m_T_mapping);
    	m_T[idx] = T_map;
#endif
}

__global__ void do_nodal_value_interpolation(int N, int m, const mesh_GPU_m mesh_GPU, const float_t* m_T_mapping, float_t* T_n, float2_t* position_to_map) 
{
#ifdef WEAR_NODE_SHIFT
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	// barycentric coordinate methods
	// https://math.stackexchange.com/questions/51326/determining-if-an-arbitrary-point-lies-inside-a-triangle-defined-by-three-points
	float_t T_new;
	float2_t x_ = position_to_map[idx];
	int nA, nB, nC;
	float2_t xAC, xBA, xCB, xPA, xPB, xPC, xA, xB, xC;
	float_t d, wA, wB, wC, TA, TB, TC;
	for (int i = 0; i < m; i++) {
		nA = mesh_GPU.node_num_mapping[i].x;
		nB = mesh_GPU.node_num_mapping[i].y;
		nC = mesh_GPU.node_num_mapping[i].z;
		xA = mesh_GPU.position_mapping[nA];
		xB = mesh_GPU.position_mapping[nB];
		xC = mesh_GPU.position_mapping[nC];
		if (distance_segment(x_, xA)<1e-10)  {
			T_n[idx] = m_T_mapping[nA];
			return;
		}
		if (distance_segment(x_, xB)<1e-10)  {
			T_n[idx] = m_T_mapping[nB];
			return;
		}
		if (distance_segment(x_, xC)<1e-10)  {
			T_n[idx] = m_T_mapping[nC];
			return;
		}
			xAC.x = xA.x - xC.x;
			xAC.y = xA.y - xC.y;
			xBA.x = xB.x - xA.x;
			xBA.y = xB.y - xA.y;
			xCB.x = xC.x - xB.x;
			xCB.y = xC.y - xB.y;
			xPC.x = x_.x - xC.x;
			xPC.y = x_.y - xC.y;
			xPA.x = x_.x - xA.x;
			xPA.y = x_.y - xA.y;
			xPB.x = x_.x - xB.x;
			xPB.y = x_.y - xB.y;
			d = cross(xBA, xAC);  // This is not the cross product!
            
			if (abs(d) >0) {
				wC = cross(xBA, xPA) / d;
				wA = cross(xCB, xPB) / d;   // (xP, xCA)
				wB = cross(xAC, xPC) / d;  // (xP, xAB)
				if ((wA >= -1e-8 && wB >= -1e-8 && wC >= -1e-8 ) || (wA <= 1e-8 && wB <= 1e-8 && wC <= 1e-8 ) ) { // inside the element
					TA = m_T_mapping[nA];
					TB = m_T_mapping[nB];
					TC = m_T_mapping[nC];
					
					T_new = abs(TA * wA  + TB * wB + TC * wC );
					float_t T_buffer_l = fmin(TA, TB);
					T_buffer_l = fmin(T_buffer_l, TC);
					if (T_new < T_buffer_l){
					   T_new = T_buffer_l;
					}
					float_t T_buffer_h = fmax(TA, TB);
					T_buffer_h = fmax(T_buffer_h, TC);
					if (T_new > T_buffer_h){
					  T_new = T_buffer_h;
					}
					T_n[idx] = T_new;
					return;
				}
			}

	}
#endif
}


__global__ void do_segment_wear_update(int N, segment_FEM_gpu segments, mesh_GPU_m mesh_GPU)
{
#ifdef WEAR_NODE_SHIFT
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	int left_no = segments.marks[idx].z;
	int right_no = segments.marks[idx].w;

	float2_t left_pos = mesh_GPU.position[left_no];
	float2_t right_pos = mesh_GPU.position[right_no];
	
	segments.left[idx].x = left_pos.x;
	segments.left[idx].y = left_pos.y;
	segments.right[idx].x = right_pos.x;
	segments.right[idx].y = right_pos.y;
	
	vec2_t left(left_pos.x, left_pos.y);
	vec2_t right(right_pos.x, right_pos.y);

	vec2_t dist = right - left;
	vec2_t n_(dist.y, -dist.x);

	n_ = glm::normalize(n_);
	segments.n[idx].x = n_.x;
	segments.n[idx].y = n_.y;

#endif
}

__device__ void warp_reduce(volatile float_t* vals, int idx) {
	vals[idx] = fmin(vals[idx], vals[idx + 32]);
	vals[idx] = fmin(vals[idx], vals[idx + 16]);
	vals[idx] = fmin(vals[idx], vals[idx + 8]);
	vals[idx] = fmin(vals[idx], vals[idx + 4]);
	vals[idx] = fmin(vals[idx], vals[idx + 2]);
	vals[idx] = fmin(vals[idx], vals[idx + 1]);
}

__global__ void do_correct_flank_contact(int N, const float2_t* left)
{
	// only for segment number smaller than 512
#ifdef WEAR_NODE_SHIFT
	volatile __shared__ float_t vals[512];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	vals[idx] = FLT_MAX;
	if (idx < N) {
		vals[idx] = left[idx].y;
	}
	
	__syncthreads();
	if (idx < 256) vals[idx] = fmin(vals[idx], vals[idx + 256]);
	__syncthreads();
	if (idx < 128) vals[idx] = fmin(vals[idx], vals[idx + 128]);
	__syncthreads();
	if (idx < 64) vals[idx] = fmin(vals[idx], vals[idx + 64]);
	__syncthreads();
	if (idx < 32) warp_reduce(vals, idx);

	
	__syncthreads();
	if (idx == 0) {
		flank_retreat = flank_contact - vals[idx];
	}
	if (flank_retreat > 1e-7)printf("strange_flank_value\n");

#endif
}

__global__ void do_mesh_shift(int N, int freq, mesh_GPU_m mesh_GPU, vec2_t tl, vec2_t br) {
#ifdef WEAR_NODE_SHIFT
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	int bool_cut = mesh_GPU.cutting_edge_flag[idx];
	int bool_fixed = mesh_GPU.fixed_edge_flag[idx];
	if (bool_cut == 0 && bool_fixed == 0) {
		float_t shift_value = flank_retreat * 1.0 * freq ;
		shift_value = fmin(shift_value, -0.0000065);
		float_t weight_y = (mesh_GPU.position[idx].x - tl.x) / (br.x - tl.x);
		float_t weight_x = fmin(1.0, (mesh_GPU.position[idx].y - tl.y) / (br.y - tl.y));

		float_t center_KT = 0.94;
		if (weight_x <= center_KT)	{
			mesh_GPU.position[idx].x += 1.45 * shift_value * (pow(center_KT, 0.7)-pow(abs(weight_x-center_KT), 0.55))*weight_y*weight_y*weight_y; // 1.85 function for adjusting the location of maximum value
		} else if (weight_x < 1.1 && weight_x > center_KT) {
			mesh_GPU.position[idx].x += 1.45 * shift_value * (pow(center_KT, 0.7)-pow(abs(weight_x-center_KT), 0.55))*weight_y*weight_y; // 1.6
		}		
		if (weight_y < 1.01){
		    mesh_GPU.position[idx].y += - 0.9 * shift_value * fmin ( pow(weight_y , 6), 1.) * pow(weight_x ,6);	// 1.65
		} else if (weight_y >= 0.99){
			mesh_GPU.position[idx].y += - 0.9 * shift_value * fmin ( pow(weight_y , 6), 1.) * weight_x*weight_x* weight_x*weight_x;	
		}
	}

#endif
}

std::vector<vec2_t> tool_FEM::get_points()
{
	return points;
}

tool_FEM::tool_FEM(vec2_t tl, float_t length, float_t height, float_t rake_angle, float_t clearance_angle, float_t r, float_t mu_fric)
	:m_mu(mu_fric) {

	tl_point_ref = tl;
	vec2_t tr(tl.x + length, tl.y);
	vec2_t bl(tl.x, tl.y - height);

	float_t alpha_rake = rake_angle * M_PI / 180.;
	float_t alpha_free = (180 - 90 - clearance_angle) * M_PI / 180.;

	mat2x2_t rot_rake(cos(alpha_rake), -sin(alpha_rake), sin(alpha_rake), cos(alpha_rake));
	mat2x2_t rot_free(cos(alpha_free), -sin(alpha_free), sin(alpha_free), cos(alpha_free));

	vec2_t down(0., -1.);

	vec2_t trc = tr + down * rot_rake;
	vec2_t blc = bl + down * rot_free;

	line l1(tr, trc);
	line l2(bl, blc);

	vec2_t br = l1.intersect(l2);
	br_point = br;

	ref_dist = fmax((glm::distance(tl, br) + r), glm::distance(tl, bl));
	ref_dist = fmax(ref_dist, glm::distance(tl, tr));

	if (r == 0.) {
		construct_segments(std::vector<vec2_t>({ tl, tr, br, bl }));
		m_front = br.x;
		return;
	}

	points = construct_points_and_fillet(tl, tr, br, bl, r);
	construct_segments(points);

	m_front = br.x;
}


tool_FEM::tool_FEM(vec2_t tl, float_t length, float_t height,
	float_t rake_angle, float_t clearance_angle,
	float_t mu_fric) : m_mu(mu_fric) {

	tl_point_ref = tl;
	vec2_t tr(tl.x + length, tl.y);
	vec2_t bl(tl.x, tl.y - height);

	float_t alpha_rake = rake_angle * M_PI / 180.;
	float_t alpha_free = (180 - 90 - clearance_angle) * M_PI / 180.;

	mat2x2_t rot_rake(cos(alpha_rake), -sin(alpha_rake), sin(alpha_rake), cos(alpha_rake));
	mat2x2_t rot_free(cos(alpha_free), -sin(alpha_free), sin(alpha_free), cos(alpha_free));

	vec2_t down(0., -1.);

	vec2_t trc = tr + down * rot_rake;
	vec2_t blc = bl + down * rot_free;

	line l1(tr, trc);
	line l2(bl, blc);

	vec2_t br = l1.intersect(l2);
	br_point = br;

	ref_dist = fmax(glm::distance(tl, br), glm::distance(tl, bl));
	ref_dist = fmax(ref_dist, glm::distance(tl, tr));

	points = std::vector<vec2_t>({ tl, tr, br, bl });
	construct_segments(std::vector<vec2_t>({ tl, tr, br, bl }));

	m_front = br.x;
}

tool_FEM::tool_FEM()
{
}

tool_FEM::~tool_FEM()
{
}

std::vector<vec2_t> tool_FEM::construct_points_and_fillet(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t r)
{
	// construct line halfing the space between l1, l2 => lm
	vec2_t pm = br;
	vec2_t nt = tr - br;
	vec2_t nl = bl - br;

	nt = glm::normalize(nt);
	nl = glm::normalize(nl);

	vec2_t nm = float_t(0.5) * (nt + nl);
	nm = glm::normalize(nm);

	line lm(pm, pm + nm);

	// find center of fillet => p
	line l1(tr, br);
	line l2(bl, br);
	vec2_t p = fit_fillet(r, lm, l1); // fit_fillet(r, lm, l2) would work too

	// find points on l1, l2 that meet the fillet => trc, blc (c = "continued")
	vec2_t trc = l1.closest_point(p);
	vec2_t blc = l2.closest_point(p);

	// construct circle segment
	float_t t1 = myatan2_FEM(p.y - trc.y, p.x - trc.x);
	float_t t2 = myatan2_FEM(p.y - blc.y, p.x - blc.x);
	m_fillet = new circle_segment(r, t1, t2, p);

	return std::vector<vec2_t>({ tl, tr, trc, blc, bl });
}

vec2_t tool_FEM::fit_fillet(float_t r, line lm, line l1) const
{
	if (l1.vertical) {
		line lparallel = line(DBL_MAX, l1.b - r, true);
		return lparallel.intersect(lm);
	}

	float_t A0 = lm.a;
	float_t B0 = lm.b;

	float_t a = l1.a;
	float_t b = l1.b;

	float_t A = a - A0;
	float_t B = b - B0;
	float_t C = r * sqrt(a * a + 1.);

	vec2_t sol = solve_quad_FEM(A * A, 2 * A * B, B * B - C * C);
	float_t xm = fmin(sol.x, sol.y);
	float_t ym = lm.a * xm + lm.b;
	return vec2_t(xm, ym);
}


void tool_FEM::construct_segments(std::vector<vec2_t> list_p)
{
	unsigned int n = list_p.size();
	for (unsigned int i = 0; i < n; i++) {
		unsigned int cur = i;
		unsigned int next = (cur + 1 > n - 1) ? 0 : i + 1;
		m_segments.push_back(segment(list_p[cur], list_p[next]));
	}
}

void tool_FEM::set_up_segments_from_mesh()
{
	node_number = this->mesh_GPU->get_node_number();
	element_number = this->mesh_GPU->get_element_number();
	std::vector<segment> m_segments_d;
	std::vector<segment> m_segments;
	std::vector<int4> mark_h;
	std::vector<int4> mark_h_sorted;
	int N = mesh_GPU->get_cutting_edge_element_number();
	for (int i = 0; i < N; i++) {
		int ele_num = this->mesh_CPU->elements.cutting_boundary_elements[i];
		int3 flag = this->mesh_CPU->elements.cutting_boundary_flags[ele_num];
		int4 node_in_ele = this->mesh_CPU->elements.node_num[ele_num];
		vec2_t left_r;
		vec2_t right_r;
		int4 mark;
		if (flag.x != 0) {
			left_r.x = this->mesh_CPU->nodes.x_init[node_in_ele.x];
			left_r.y = this->mesh_CPU->nodes.y_init[node_in_ele.x];
			right_r.x = this->mesh_CPU->nodes.x_init[node_in_ele.y];
			right_r.y = this->mesh_CPU->nodes.y_init[node_in_ele.y];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.x;
				mark.w = node_in_ele.y;
			}
			else if (segment_buff.left == right_r) {
				mark.z = node_in_ele.y;
				mark.w = node_in_ele.x;
			}
			mark_h.push_back(mark);
		}
		if (flag.y != 0) {
			left_r.x = this->mesh_CPU->nodes.x_init[node_in_ele.y];
			left_r.y = this->mesh_CPU->nodes.y_init[node_in_ele.y];
			right_r.x = this->mesh_CPU->nodes.x_init[node_in_ele.z];
			right_r.y = this->mesh_CPU->nodes.y_init[node_in_ele.z];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.y;
				mark.w = node_in_ele.z;
			}
			else if (segment_buff.left== right_r) {
				mark.z = node_in_ele.z;
				mark.w = node_in_ele.y;
			}
			mark_h.push_back(mark);
		}
		if (flag.z != 0) {
			left_r.x = this->mesh_CPU->nodes.x_init[node_in_ele.z];
			left_r.y = this->mesh_CPU->nodes.y_init[node_in_ele.z];
			right_r.x = this->mesh_CPU->nodes.x_init[node_in_ele.x];
			right_r.y = this->mesh_CPU->nodes.y_init[node_in_ele.x];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.z;
				mark.w = node_in_ele.x;
			}
			else if (segment_buff.left == right_r){
				mark.z = node_in_ele.x;
				mark.w = node_in_ele.z;
			}
			mark_h.push_back(mark);
		}  
	}


	float_t y_min = DBL_MAX;
	float_t x_max = - DBL_MAX;
	float_t y_max = -DBL_MAX;
	float_t x_min = DBL_MAX;
	float_t dist_max = -DBL_MAX;


	for (auto it = m_segments_d.begin(); it != m_segments_d.end(); ++it) {
		y_min = fmin(y_min, it->left.y);
		y_min = fmin(y_min, it->right.y);
		x_max = fmax(x_max, it->left.x);
		x_max = fmax(x_max, it->right.x);
		x_min = fmin(x_min, it->left.x);
		x_min = fmin(x_min, it->right.x);
		y_max = fmax(y_max, it->left.y);
		y_max = fmax(y_max, it->right.y);
		dist_max = fmax((glm::distance(tl_point_ref, it->left)), dist_max);
		dist_max = fmax((glm::distance(tl_point_ref, it->right)), dist_max);
	}
	tl_point_ref.x = x_min;
	tl_point_ref.y = y_max;
	for (auto it = m_segments_d.begin(); it != m_segments_d.end(); ++it) {
		dist_max = fmax((glm::distance(tl_point_ref, it->left)), dist_max);
	}
	lowest_point = y_min;
	right_point = x_max;
	ref_dist = dist_max;

	// Correct the normal direction if pockets are modelled, 2022.11.15 NZ
	// Sort the order of segments, 2023.01.03, NZ
	bool is_left = 0;
	int n_segment = m_segments_d.size();
	int num_buffer = 0;
	for (int i = 0; i <  n_segment; i++){
		if (m_segments_d[i].left.y == y_max) {
			is_left = 1;
			num_buffer = i;
			break;
		}
		if (m_segments_d[i].right.y == y_max) {
			is_left = 0;
			num_buffer = i;
			break;
		}
	}

	if (is_left){
		float_t buffer_location_y = y_max;
		float_t buffer_location_x = m_segments_d[num_buffer].left.x;
		int rep = 0;
		for (int i = 0; i < n_segment; i++){
			//printf("Buffer at Point 1: %f, %f, rep is %d \n", buffer_location_x, buffer_location_y, rep);
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].left.y == buffer_location_y && m_segments_d[j].left.x == buffer_location_x){
				    buffer_location_y = m_segments_d[j].right.y;
				    buffer_location_x = m_segments_d[j].right.x;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c1;
			    }
			}
			for (int j = 0; j < n_segment; j++){	
				if (m_segments_d[j].right.y == buffer_location_y && m_segments_d[j].right.x == buffer_location_x && j != rep){
					m_segments_d[j].right.x = m_segments_d[j].left.x;
					m_segments_d[j].right.y = m_segments_d[j].left.y;
					m_segments_d[j].left.x = buffer_location_x;
					m_segments_d[j].left.y = buffer_location_y;
					m_segments_d[j].n.x = -m_segments_d[j].n.x;
					m_segments_d[j].n.y = -m_segments_d[j].n.y;
					buffer_location_y = m_segments_d[j].right.y;
				    buffer_location_x = m_segments_d[j].right.x;
					int change = mark_h[j].w;
					mark_h[j].w = mark_h[j].z;
					mark_h[j].z = change;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
					goto c1;
			    }
			}
		c1:;  
		}
	} else {
		float_t buffer_location_y = y_max;
		float_t buffer_location_x = m_segments_d[num_buffer].right.x;
		int rep = 0;
		for (int i = 0; i < n_segment; i++){
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].right.y == buffer_location_y && m_segments_d[j].right.x == buffer_location_x){
				    buffer_location_y = m_segments_d[j].left.y;
				    buffer_location_x = m_segments_d[j].left.x;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c2;
			    }
			}
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].left.y == buffer_location_y && m_segments_d[j].left.x == buffer_location_x && j != rep){
				    m_segments_d[j].left = m_segments_d[j].right;
					m_segments_d[j].right.x = buffer_location_x;
					m_segments_d[j].right.y = buffer_location_y;
					m_segments_d[j].n.x = -m_segments_d[j].n.x;
					m_segments_d[j].n.y = -m_segments_d[j].n.y;
					buffer_location_y = m_segments_d[j].left.y;
				    buffer_location_x = m_segments_d[j].left.x;
					int change = mark_h[j].w;
					mark_h[j].w = mark_h[j].z;
					mark_h[j].z = change;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c2;
			    }
			}
		c2:;    
		}

	}
    assert(m_segments.size() == n_segment);
	
	cutting_segment_size = n_segment;
	printf("Number of segments on the tool-workpiece interface: %d\n", n_segment);
	float2_t* left_h = new float2_t[n_segment];
	float2_t* right_h = new float2_t[n_segment];
	float2_t* n_h = new float2_t[n_segment];
	float2_t* heat_exchange_h = new float2_t[n_segment];
	float_t* fric_heat_h = new float_t[n_segment];
#ifdef WEAR_NODE_SHIFT
	float2_t* wear_rate_h = new float2_t[n_segment];
	float_t* wear_nodes_h = new float_t[n_segment];
	float4_t* physical_para_h = new float4_t[n_segment];
	float_t* sliding_force_h = new float_t[n_segment];
#endif
	int4* marks_h = new int4[n_segment];
	float2_t zero_float2;
	zero_float2.x = 0.;
	zero_float2.y = 0.;
	float4_t zero_float4;
	zero_float4.x = 0.;
	zero_float4.y = 0.;
	zero_float4.z = 0.;
	zero_float4.w = 0.;


	for (int i = 0; i < n_segment; i++) {
		left_h[i].x = m_segments[i].left.x;
		left_h[i].y = m_segments[i].left.y;

		right_h[i].x = m_segments[i].right.x;
		right_h[i].y = m_segments[i].right.y;

		n_h[i].x = m_segments[i].n.x;
		n_h[i].y = m_segments[i].n.y;

		marks_h[i] = mark_h_sorted[i];
		heat_exchange_h[i] = zero_float2;
		fric_heat_h[i] = 0.;
#ifdef WEAR_NODE_SHIFT
		wear_rate_h[i] = zero_float2;
		wear_nodes_h[i] = 0.;
		physical_para_h[i] = zero_float4;
		sliding_force_h[i] = 0.;
#endif

	}
#ifdef WEAR_NODE_SHIFT
    this->segments_FEM = new segment_FEM_gpu(left_h, right_h, n_h, heat_exchange_h, marks_h, fric_heat_h, wear_rate_h, wear_nodes_h, physical_para_h, sliding_force_h, n_segment);
	//this->segments_FEM = new segment_FEM_gpu(left_h, right_h, n_h, heat_exchange_h, marks_h, fric_heat_h, wear_rate_h,wear_nodes_h, n_segment);
#else
	this->segments_FEM = new segment_FEM_gpu(left_h, right_h, n_h, heat_exchange_h, marks_h, fric_heat_h, n_segment);
#endif

    // find tl, br
	float_t tl_x = FLT_MAX;
	float_t tl_y = -FLT_MAX;
	float_t br_x = -FLT_MAX;
	float_t br_y = FLT_MAX;

	for (int i = 0; i < node_number; i++) {
		if (this->mesh_CPU->nodes.x_init[i] <= tl_x && this->mesh_CPU->nodes.y_init[i] >= tl_y) {
			tl_x = this->mesh_CPU->nodes.x_init[i];
			tl_y = this->mesh_CPU->nodes.y_init[i];
		}
		if (this->mesh_CPU->nodes.x_init[i] >= br_x && this->mesh_CPU->nodes.y_init[i] <= br_y) {
			br_x = this->mesh_CPU->nodes.x_init[i];
			br_y = this->mesh_CPU->nodes.y_init[i];
		}
	}
	this->br_point = vec2_t(br_x, br_y);
	this->tl_point_ref = vec2_t(tl_x, tl_y);

}

void tool_FEM::update_segments_remesh(mesh_read* mesh_CPU_new, mesh_GPU_m* mesh_GPU_new)
{
	node_number = mesh_GPU_new->get_node_number();
	element_number = mesh_GPU_new->get_element_number();
	std::vector<segment> m_segments_d;
	std::vector<int4> mark_h;
	std::vector<int4> mark_h_sorted;
	int N = mesh_GPU_new->get_cutting_edge_element_number();
	for (int i = 0; i < N; i++) {
		int ele_num = mesh_CPU_new->elements.cutting_boundary_elements[i];
		int3 flag = mesh_CPU_new->elements.cutting_boundary_flags[ele_num];
		int4 node_in_ele = mesh_CPU_new->elements.node_num[ele_num];
		vec2_t left_r;
		vec2_t right_r;
		int4 mark;
		if (flag.x != 0) {
			left_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.x];
			left_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.x];
			right_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.y];
			right_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.y];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
		    m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.x;
				mark.w = node_in_ele.y;
			}
			else if (segment_buff.left == right_r) {
				mark.z = node_in_ele.y;
				mark.w = node_in_ele.x;
			}
			mark_h.push_back(mark);
		}
		if (flag.y != 0) {
			left_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.y];
			left_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.y];
			right_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.z];
			right_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.z];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.y;
				mark.w = node_in_ele.z;
			}
			else if (segment_buff.left== right_r) {
				mark.z = node_in_ele.z;
				mark.w = node_in_ele.y;
			}
			mark_h.push_back(mark);
		}
		if (flag.z != 0) {
			left_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.z];
			left_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.z];
			right_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.x];
			right_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.x];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.z;
				mark.w = node_in_ele.x;
			}
			else if (segment_buff.left == right_r){
				mark.z = node_in_ele.x;
				mark.w = node_in_ele.z;
			}
			mark_h.push_back(mark);
					}  
	}


	float_t y_min = DBL_MAX;
	float_t x_max = - DBL_MAX;
	float_t y_max = -DBL_MAX;
	float_t x_min = DBL_MAX;
	float_t dist_max = -DBL_MAX;


	for (auto it = m_segments_d.begin(); it != m_segments_d.end(); ++it) {
		y_min = fmin(y_min, it->left.y);
		y_min = fmin(y_min, it->right.y);
		x_max = fmax(x_max, it->left.x);
		x_max = fmax(x_max, it->right.x);
		x_min = fmin(x_min, it->left.x);
		x_min = fmin(x_min, it->right.x);
		y_max = fmax(y_max, it->left.y);
		y_max = fmax(y_max, it->right.y);
		dist_max = fmax((glm::distance(tl_point_ref, it->left)), dist_max);
		dist_max = fmax((glm::distance(tl_point_ref, it->right)), dist_max);
	}

	for (auto it = m_segments_d.begin(); it != m_segments_d.end(); ++it) {
		dist_max = fmax((glm::distance(tl_point_ref, it->left)), dist_max);
	}
	lowest_point = y_min;
	right_point = x_max;
	ref_dist = dist_max;


    int n_segment = m_segments_d.size();

	// Correct the normal direction if pockets are modelled, 2022.11.15 NZ
	// Sort the order of segments, 2023.01.03, NZ
	bool is_left = 0;
	
	int num_buffer = 0;
	for (int i = 0; i <  n_segment; i++){
		if (m_segments_d[i].left.y == y_max) {
			is_left = 1;
			num_buffer = i;
			break;
		}
		if (m_segments_d[i].right.y == y_max) {
			is_left = 0;
			num_buffer = i;
			break;
		}
	}

	if (is_left){
		float_t buffer_location_y = y_max;
		float_t buffer_location_x = m_segments_d[num_buffer].left.x;
		int rep = 0;
		for (int i = 0; i < n_segment; i++){
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].left.y == buffer_location_y && m_segments_d[j].left.x == buffer_location_x){
				    buffer_location_y = m_segments_d[j].right.y;
				    buffer_location_x = m_segments_d[j].right.x;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c1;
			    }
			}
			for (int j = 0; j < n_segment; j++){	
				if (m_segments_d[j].right.y == buffer_location_y && m_segments_d[j].right.x == buffer_location_x && j != rep){
					m_segments_d[j].right.x = m_segments_d[j].left.x;
					m_segments_d[j].right.y = m_segments_d[j].left.y;
					m_segments_d[j].left.x = buffer_location_x;
					m_segments_d[j].left.y = buffer_location_y;
					m_segments_d[j].n.x = -m_segments_d[j].n.x;
					m_segments_d[j].n.y = -m_segments_d[j].n.y;
					buffer_location_y = m_segments_d[j].right.y;
				    buffer_location_x = m_segments_d[j].right.x;
					int change = mark_h[j].w;
					mark_h[j].w = mark_h[j].z;
					mark_h[j].z = change;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
					goto c1;
			    }
			}
		c1:;  
		}
	} else {
		float_t buffer_location_y = y_max;
		float_t buffer_location_x = m_segments_d[num_buffer].right.x;
		int rep = 0;
		for (int i = 0; i < n_segment; i++){
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].right.y == buffer_location_y && m_segments_d[j].right.x == buffer_location_x){
				    buffer_location_y = m_segments_d[j].left.y;
				    buffer_location_x = m_segments_d[j].left.x;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c2;
			    }
			}
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].left.y == buffer_location_y && m_segments_d[j].left.x == buffer_location_x && j != rep){
				    m_segments_d[j].left = m_segments_d[j].right;
					m_segments_d[j].right.x = buffer_location_x;
					m_segments_d[j].right.y = buffer_location_y;
					m_segments_d[j].n.x = -m_segments_d[j].n.x;
					m_segments_d[j].n.y = -m_segments_d[j].n.y;
					buffer_location_y = m_segments_d[j].left.y;
				    buffer_location_x = m_segments_d[j].left.x;
					int change = mark_h[j].w;
					mark_h[j].w = mark_h[j].z;
					mark_h[j].z = change;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c2;
			    }
			}
		c2:;    
		}

	}
	
	int4* marks_h = new int4[n_segment];
	for (int i = 0; i < n_segment; i++) {
		marks_h[i] = mark_h_sorted[i];
	}
    cudaMemcpy(this->segments_FEM->marks, marks_h, sizeof(int4) * N, cudaMemcpyHostToDevice);
    
}

void tool_FEM::update_segments_remesh_new(mesh_read* mesh_CPU_new, mesh_GPU_m* mesh_GPU_new)
{
	node_number = mesh_GPU_new->get_node_number();
	element_number = mesh_GPU_new->get_element_number();
	std::vector<segment> m_segments_d;
	std::vector<int4> mark_h;
	std::vector<int4> mark_h_sorted;
	int N = mesh_GPU_new->get_cutting_edge_element_number();
	for (int i = 0; i < N; i++) {
		int ele_num = mesh_CPU_new->elements.cutting_boundary_elements[i];
		int3 flag = mesh_CPU_new->elements.cutting_boundary_flags[ele_num];
		int4 node_in_ele = mesh_CPU_new->elements.node_num[ele_num];
		vec2_t left_r;
		vec2_t right_r;
		int4 mark;
		if (flag.x != 0) {
			left_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.x];
			left_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.x];
			right_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.y];
			right_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.y];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
		    m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.x;
				mark.w = node_in_ele.y;
			}
			else if (segment_buff.left == right_r) {
				mark.z = node_in_ele.y;
				mark.w = node_in_ele.x;
			}
			mark_h.push_back(mark);
			
		}
		if (flag.y != 0) {
			left_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.y];
			left_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.y];
			right_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.z];
			right_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.z];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.y;
				mark.w = node_in_ele.z;
			}
			else if (segment_buff.left== right_r) {
				mark.z = node_in_ele.z;
				mark.w = node_in_ele.y;
			}
			mark_h.push_back(mark);
			
		}
		if (flag.z != 0) {
			left_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.z];
			left_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.z];
			right_r.x = mesh_CPU_new->nodes.x_init[node_in_ele.x];
			right_r.y = mesh_CPU_new->nodes.y_init[node_in_ele.x];
			segment segment_buff = segment(left_r, right_r, tl_point_ref);
			m_segments_d.push_back(segment_buff);
			mark.x = ele_num;
			mark.y = node_in_ele.w;
			if (segment_buff.left == left_r) {
				mark.z = node_in_ele.z;
				mark.w = node_in_ele.x;
			}
			else if (segment_buff.left == right_r){
				mark.z = node_in_ele.x;
				mark.w = node_in_ele.z;
			}
			mark_h.push_back(mark);
			
		}  
	}


	float_t y_min = DBL_MAX;
	float_t x_max = - DBL_MAX;
	float_t y_max = -DBL_MAX;
	float_t x_min = DBL_MAX;
	float_t dist_max = -DBL_MAX;


	for (auto it = m_segments_d.begin(); it != m_segments_d.end(); ++it) {
		y_min = fmin(y_min, it->left.y);
		y_min = fmin(y_min, it->right.y);
		x_max = fmax(x_max, it->left.x);
		x_max = fmax(x_max, it->right.x);
		x_min = fmin(x_min, it->left.x);
		x_min = fmin(x_min, it->right.x);
		y_max = fmax(y_max, it->left.y);
		y_max = fmax(y_max, it->right.y);
		dist_max = fmax((glm::distance(tl_point_ref, it->left)), dist_max);
		dist_max = fmax((glm::distance(tl_point_ref, it->right)), dist_max);
	}

	for (auto it = m_segments_d.begin(); it != m_segments_d.end(); ++it) {
		dist_max = fmax((glm::distance(tl_point_ref, it->left)), dist_max);
	}
	lowest_point = y_min;
	right_point = x_max;
	ref_dist = dist_max;


    int n_segment = m_segments_d.size();
	if (n_segment != segments_FEM->segment_num) printf("Segment number is changed, old is %d, new is %d\n", segments_FEM->segment_num, n_segment);

	// Correct the normal direction if pockets are modelled, 2022.11.15 NZ
	// Sort the order of segments, 2023.01.03, NZ
	bool is_left = 0;
	
	int num_buffer = 0;
	for (int i = 0; i <  n_segment; i++){
		if (m_segments_d[i].left.y == y_max) {
			is_left = 1;
			num_buffer = i;
			break;
		}
		if (m_segments_d[i].right.y == y_max) {
			is_left = 0;
			num_buffer = i;
			break;
		}
	}

	if (is_left){
		float_t buffer_location_y = y_max;
		float_t buffer_location_x = m_segments_d[num_buffer].left.x;
		int rep = 0;
		for (int i = 0; i < n_segment; i++){
			//printf("Buffer at Point 1: %f, %f, rep is %d \n", buffer_location_x, buffer_location_y, rep);
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].left.y == buffer_location_y && m_segments_d[j].left.x == buffer_location_x){
				    buffer_location_y = m_segments_d[j].right.y;
				    buffer_location_x = m_segments_d[j].right.x;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c1;
			    }
			}
			for (int j = 0; j < n_segment; j++){	
				if (m_segments_d[j].right.y == buffer_location_y && m_segments_d[j].right.x == buffer_location_x && j != rep){
					m_segments_d[j].right.x = m_segments_d[j].left.x;
					m_segments_d[j].right.y = m_segments_d[j].left.y;
					m_segments_d[j].left.x = buffer_location_x;
					m_segments_d[j].left.y = buffer_location_y;
					m_segments_d[j].n.x = -m_segments_d[j].n.x;
					m_segments_d[j].n.y = -m_segments_d[j].n.y;
					buffer_location_y = m_segments_d[j].right.y;
				    buffer_location_x = m_segments_d[j].right.x;
					int change = mark_h[j].w;
					mark_h[j].w = mark_h[j].z;
					mark_h[j].z = change;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
					goto c1;
			    }
			}
		c1:;  
		}
	} else {
		float_t buffer_location_y = y_max;
		float_t buffer_location_x = m_segments_d[num_buffer].right.x;
		int rep = 0;
		for (int i = 0; i < n_segment; i++){
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].right.y == buffer_location_y && m_segments_d[j].right.x == buffer_location_x){
				    buffer_location_y = m_segments_d[j].left.y;
				    buffer_location_x = m_segments_d[j].left.x;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c2;
			    }
			}
			for (int j = 0; j < n_segment; j++){
				if (m_segments_d[j].left.y == buffer_location_y && m_segments_d[j].left.x == buffer_location_x && j != rep){
				    m_segments_d[j].left = m_segments_d[j].right;
					m_segments_d[j].right.x = buffer_location_x;
					m_segments_d[j].right.y = buffer_location_y;
					m_segments_d[j].n.x = -m_segments_d[j].n.x;
					m_segments_d[j].n.y = -m_segments_d[j].n.y;
					buffer_location_y = m_segments_d[j].left.y;
				    buffer_location_x = m_segments_d[j].left.x;
					int change = mark_h[j].w;
					mark_h[j].w = mark_h[j].z;
					mark_h[j].z = change;
					rep = j;
					m_segments.push_back(m_segments_d[j]);
					mark_h_sorted.push_back(mark_h[j]);
                    goto c2;
			    }
			}
		c2:;    
		}

	}


	cudaFree(segments_FEM->left);
	cudaFree(segments_FEM->right);
	cudaFree(segments_FEM->n);
	cudaFree(segments_FEM->heat_exchange);
	cudaFree(segments_FEM->marks);
	cudaFree(segments_FEM->fric_heat);
#ifdef WEAR_NODE_SHIFT
	cudaFree(segments_FEM->wear_rate);
	cudaFree(segments_FEM->wear_nodes);
	cudaFree(segments_FEM->physical_para);
	cudaFree(segments_FEM->sliding_force);
#endif


	cutting_segment_size = n_segment;
	printf("Number of segments on the tool-workpiece interface: %d\n", n_segment);
	float2_t* left_h = new float2_t[n_segment];
	float2_t* right_h = new float2_t[n_segment];
	float2_t* n_h = new float2_t[n_segment];
	float2_t* heat_exchange_h = new float2_t[n_segment];
	float_t* fric_heat_h = new float_t[n_segment];
#ifdef WEAR_NODE_SHIFT
	float2_t* wear_rate_h = new float2_t[n_segment];
	float_t* wear_nodes_h = new float_t[n_segment];
	float4_t* physical_para_h = new float4_t[n_segment];
	float_t* sliding_force_h = new float_t[n_segment];
#endif
	int4* marks_h = new int4[n_segment];
	float2_t zero_float2;
	zero_float2.x = 0.;
	zero_float2.y = 0.;
	float4_t zero_float4;
	zero_float4.x = 0.;
	zero_float4.y = 0.;
	zero_float4.z = 0.;
	zero_float4.w = 0.;


	for (int i = 0; i < n_segment; i++) {
		left_h[i].x = m_segments[i].left.x;
		left_h[i].y = m_segments[i].left.y;

		right_h[i].x = m_segments[i].right.x;
		right_h[i].y = m_segments[i].right.y;

		n_h[i].x = m_segments[i].n.x;
		n_h[i].y = m_segments[i].n.y;

		marks_h[i] = mark_h_sorted[i];
		heat_exchange_h[i] = zero_float2;
		fric_heat_h[i] = 0.;
#ifdef WEAR_NODE_SHIFT
		wear_rate_h[i] = zero_float2;
		wear_nodes_h[i] = 0.;
		physical_para_h[i] = zero_float4;
		sliding_force_h[i] = 0.;
#endif

	}

    segments_FEM->segment_num = n_segment;

	cudaMalloc((void**)&segments_FEM->left, sizeof(float2_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->right, sizeof(float2_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->n, sizeof(float2_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->heat_exchange, sizeof(float2_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->marks, sizeof(int4) * n_segment);
	cudaMalloc((void**)&segments_FEM->fric_heat, sizeof(float_t) * n_segment);

#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**)&segments_FEM->wear_rate, sizeof(float2_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->wear_nodes, sizeof(float_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->physical_para, sizeof(float4_t) * n_segment);
	cudaMalloc((void**)&segments_FEM->sliding_force, sizeof(float_t) * n_segment);
#endif


	cudaMemcpy(segments_FEM->left, left_h, sizeof(float2_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(segments_FEM->right, right_h, sizeof(float2_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->n, n_h, sizeof(float2_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->heat_exchange, heat_exchange_h, sizeof(float2_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->marks, marks_h, sizeof(int4) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->fric_heat, fric_heat_h, sizeof(float_t) * n_segment, cudaMemcpyHostToDevice);
#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(this->segments_FEM->wear_rate, wear_rate_h, sizeof(float2_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->wear_nodes, wear_nodes_h, sizeof(float_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->physical_para, physical_para_h, sizeof(float4_t) * n_segment, cudaMemcpyHostToDevice);
	cudaMemcpy(this->segments_FEM->sliding_force, sliding_force_h, sizeof(float_t) * n_segment, cudaMemcpyHostToDevice);
#endif	
}


void tool_FEM::set_up_FEM_tool_gpu(tool_constants tc_h)
{
	this->FEM_solver = new FEM_thermal_2D_GPU(mesh_GPU, tc_h);
	float_t ini_flank_retreat = 0.;
	cudaMemcpyToSymbol(flank_retreat, &ini_flank_retreat, sizeof(float_t), 0, cudaMemcpyHostToDevice);
}

void tool_FEM::apply_heat_boundary_condition()
{
	FEM_solver->apply_robin_bdc(this->segments_FEM, cutting_segment_size);
	//FEM_solver->apply_robin_bdc_v1(this->segments_FEM, cutting_segment_size); // pure frictional heat, no heat transfer from WP
}

void tool_FEM::tool_update_FEM_gpu(float_t dt)
{
	int N = node_number;
	vec2_t vel = this->get_vel();
	tl_point_ref += vel * dt;
	br_point += vel * dt;
	lowest_point += vel.y * dt;
	right_point += vel.x * dt;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	//cudaEvent_t node_update;
	//cudaEventCreate(&node_update);
	do_tool_update_FEM_gpu <<< dG, dB >>> (N, this->mesh_GPU->position, vel, dt);
	//cudaEventSynchronize(node_update);
	//float_t low = lowest_point;
#ifdef WEAR_NODE_SHIFT 
	cudaMemcpyToSymbol(flank_contact, &lowest_point, sizeof(float_t), 0, cudaMemcpyHostToDevice);
#endif
}

void tool_FEM::segments_update(float_t dt)
{
	int N = cutting_segment_size;
	vec2_t vel = this->get_vel();
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	//cudaEvent_t seg_update;
	//cudaEventCreate(&seg_update);
	do_segments_update << < dG, dB >> > (N,  segments_FEM->left, segments_FEM->right, segments_FEM->heat_exchange, segments_FEM->fric_heat, vel, dt);
	//cudaEventSynchronize(seg_update);
}

void tool_FEM::segments_fric_reset()
{
	int N = cutting_segment_size;
	vec2_t vel = this->get_vel();
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);

	do_reset_vector << < dG, dB >> > (N, segments_FEM->fric_heat);

}

void tool_FEM::segments_wear_rate_reset()
{
	int N = cutting_segment_size;
	vec2_t vel = this->get_vel();
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);

#ifdef WEAR_NODE_SHIFT
	do_reset_vector2 << < dG, dB >> > (N, segments_FEM->wear_rate);
	do_reset_vector << < dG, dB >> > (N, segments_FEM->wear_nodes);
	do_reset_vector4 << < dG, dB >> > (N, segments_FEM->physical_para);
	do_reset_vector << < dG, dB >> > (N, segments_FEM->sliding_force);
#endif

}

void tool_FEM::mapping_data_copy()
{
#ifdef WEAR_NODE_SHIFT
	int N = mesh_GPU->get_node_number();
	int m = mesh_GPU->get_element_number();

	cudaMemcpy(FEM_solver->m_T_mapping, FEM_solver->m_T, sizeof(float_t) * (N), cudaMemcpyDeviceToDevice);
	cudaMemcpy(mesh_GPU->position_mapping, mesh_GPU->position, sizeof(float2_t) * (N), cudaMemcpyDeviceToDevice);
	cudaMemcpy(mesh_GPU->node_num_mapping, mesh_GPU->node_num, sizeof(int4) * (m), cudaMemcpyDeviceToDevice);

#endif
}

void tool_FEM::nodal_shift_wear(float_t dt)
{
	int N = cutting_segment_size;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
#ifdef WEAR_NODE_SHIFT
	//cudaEvent_t node_shift;
	//cudaEventCreate(&node_shift);
#ifdef SMOOTH_TEMP_WEAR
	do_nodal_shift_wear_smooth_temp << < dG, dB >> > (N, dt, *segments_FEM, *mesh_GPU, FEM_solver->m_T);
#else
	do_nodal_shift_wear_02 << < dG, dB >> > (N, dt, *segments_FEM, *mesh_GPU);
#endif
	//cudaEventSynchronize(node_shift);
#endif
}

void tool_FEM::nodal_value_interpolation()
{
#ifdef WEAR_NODE_SHIFT	
	int N = node_number;
	int ele_size = element_number;

	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
    do_nodal_value_interpolation_old << < dG, dB >> > (N, ele_size, *mesh_GPU, this->FEM_solver->m_T_mapping, this->FEM_solver->m_T);
	


#endif
}

void tool_FEM::nodal_value_interpolation(mesh_GPU_m* mesh_GPU_new)
{
#ifdef WEAR_NODE_SHIFT	
	int N = mesh_GPU_new->get_node_number();
	int ele_size = this->mesh_GPU->get_element_number();

	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
    //do_nodal_value_interpolation_old << < dG, dB >> > (N, ele_size, this->mesh_GPU, this->FEM_solver->m_T_mapping, this->FEM_solver->m_T);
	do_nodal_value_interpolation << < dG, dB >> > (N, ele_size, *mesh_GPU, this->FEM_solver->m_T_mapping, mesh_GPU_new->T_nodes, mesh_GPU_new->position);

#endif
}

void tool_FEM::segment_wear_update()
{
	int N = cutting_segment_size;
	dim3 dB(512);
	dim3 dG((N + 512 - 1) / 512);
#ifdef WEAR_NODE_SHIFT
	//cudaEvent_t seg_update;
	//cudaEventCreate(&seg_update);
	do_segment_wear_update << < dG, dB >> > (N, *segments_FEM, *mesh_GPU);
	//cudaEventSynchronize(seg_update);
#endif
}

void tool_FEM::correct_flank_contact()
{
	int N = cutting_segment_size;
	dim3 dB(512); // can be adjusted as 512, changes should be made in the kernel as well
	dim3 dG(1);
#ifdef WEAR_NODE_SHIFT
	//cudaEvent_t seg_update;
	//cudaEventCreate(&seg_update);
	do_correct_flank_contact << < dG, dB >> > (N, segments_FEM->left);
	//cudaEventSynchronize(seg_update);
#endif
}

void tool_FEM::mesh_shift(int freq)
{
#ifdef WEAR_NODE_SHIFT
	int N = node_number;
	//int ele_size = element_number;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	do_mesh_shift << < dG, dB >> > (N, freq, *mesh_GPU, tl_point_ref, br_point);

#endif
}

void tool_FEM::remesh_read_segments()
{
#ifdef WEAR_NODE_SHIFT
    int seg_num = cutting_segment_size;

	float2_t* left = new float2_t[seg_num];
	float2_t* right = new float2_t[seg_num];
	int4* mark = new int4[seg_num];

	cudaMemcpy(left, this->segments_FEM->left, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(right, this->segments_FEM->right, sizeof(float2_t) * (seg_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(mark, this->segments_FEM->marks, sizeof(int4) * (seg_num), cudaMemcpyDeviceToHost);
/*
	// it seems that the left and right number are not transferred correctly
	for (int i = 0; i < seg_num; i++){
		printf("%d, %d, %d, %d, %f, %f\n", mark[i].x, mark[i].y, mark[i].z, mark[i].w, left[i].x, left[i].y);
	}
*/
	gmsh::initialize();
	gmsh::option::setNumber("General.Verbosity", 2);
	gmsh::model::add("tool_remesh");

	gmsh::model::geo::addPoint(tl_point_ref.x, tl_point_ref.y, 0, 0.075, 1);
	float_t length = 0.004;
	float_t dist_buffer;
	float_t dist_judge;
	float_t dist_prev = 0.;
	std::vector<int> curve_order_cutting;
	int point_number = 0;
	if (left[0].y > right[0].y) {
		for (int i = 0; i < seg_num; i++){
		  dist_judge = sqrtf((left[i].x - right[i+1].x) * (left[i].x - right[i+1].x) + (left[i].y - right[i+1].y) * (left[i].y - right[i+1].y));
		  dist_buffer = sqrtf((left[i].x - right[i].x) * (left[i].x - right[i].x) + (left[i].y - right[i].y) * (left[i].y - right[i].y));		 
          length = 6.0*dist_buffer;
          gmsh::model::geo::addPoint(left[i].x, left[i].y, 0, length, 2 + point_number);
          curve_order_cutting.push_back(2 + point_number);
		  point_number++; 
		  if ((dist_judge < 1.25 * dist_buffer && i != (seg_num - 1) && dist_prev > 0.7 * dist_buffer) || dist_judge < 0.00013)
		  {
			//printf("Adjusting short boundary edge! i is %d, left %f, %f, Dist_judge is %f, dist_buffer is %f\n", i, left[i].x, left[i].y, dist_judge, dist_buffer);
			i++;
		  }
		  dist_prev = dist_buffer;
		  //printf("%f, %f \n", left[i].x, left[i].y);
		}
        gmsh::model::geo::addPoint(right[seg_num-1].x, right[seg_num-1].y, 0, length, point_number + 2);
		//printf("%f, %f \n", right[seg_num-1].x, right[seg_num-1].y);

	} else {
		for (int i = 0; i < seg_num; i++){
          dist_judge = sqrtf((left[i+1].x - right[i].x) * (left[i+1].x - right[i].x) + (left[i+1].y - right[i].y) * (left[i+1].y - right[i].y));
		  dist_buffer = sqrtf((left[i].x - right[i].x) * (left[i].x - right[i].x) + (left[i].y - right[i].y) * (left[i].y - right[i].y));
          length = 6.0*dist_buffer;
		  gmsh::model::geo::addPoint(right[i].x, right[i].y, 0, length, 2 + point_number);
		  curve_order_cutting.push_back(2 + point_number);
		  point_number++;
		  if ((dist_judge < 1.25 * dist_buffer && i != (seg_num - 1) && dist_prev > 0.7 * dist_buffer) || dist_judge < 0.00013)
		  {
			//printf("Adjusting short boundary edge! i is %d, left %f, %f, Dist_judge is %f, dist_buffer is %f\n", i, left[i].x, left[i].y, dist_judge, dist_buffer);
			i++;
		  }
		  dist_prev = dist_buffer;
		  //printf("%f, %f \n", right[i].x, right[i].y);
		}
        gmsh::model::geo::addPoint(left[seg_num-1].x, left[seg_num-1].y, 0, length, point_number + 2);
        //printf("%f, %f \n", left[seg_num-1].x, left[seg_num-1].y);

	}

    std::vector<int> curve_order;
    //printf("Print curve order: \n");
    for (int i = 0; i < point_number + 1; i++){
        gmsh::model::geo::addLine(i+1, i+2, i+1);
        curve_order.push_back(i+1);
	}
	gmsh::model::geo::addLine(point_number + 2, 1, point_number + 2);
	curve_order.push_back(point_number + 2);
	
	gmsh::model::geo::addCurveLoop(curve_order, 1);

	gmsh::model::geo::addPlaneSurface({1}, 1);

	gmsh::model::geo::synchronize();

	gmsh::model::addPhysicalGroup(1, curve_order_cutting, 67, "CuttingEdge");
    gmsh::model::addPhysicalGroup(1, {1,point_number + 2}, 2, "FixedBoundary");

	gmsh::model::geo::synchronize();

	gmsh::model::mesh::generate();
    gmsh::option::setNumber("Mesh.MshFileVersion", 4);
	gmsh::option::setNumber("Mesh.SaveAll", 1);

	gmsh::write("tool_gmsh_remeshed.inp");
	gmsh::write("tool_gmsh_remeshed.msh");
    gmsh::finalize();
	delete[] left;
	delete[] right;


#endif
}

void tool_FEM::read_remesh_gmsh(){



    mesh_read* mesh_CPU_new = new mesh_read("tool_gmsh_remeshed.inp", 1);
	mesh_CPU_new->write_mesh_file();
	mesh_GPU_m* mesh_GPU_new = new mesh_GPU_m(mesh_CPU_new->nodes, mesh_CPU_new->elements, mesh_CPU_new->color_total, mesh_CPU_new->color_cutting_edge, 0);
	
	
	nodal_value_interpolation(mesh_GPU_new);
	update_segments_remesh_new(mesh_CPU_new, mesh_GPU_new);	
	check_cuda_error("Remesh_read_segments\n");
    this->mesh_CPU = mesh_CPU_new;
	mesh_GPU->reset_mesh_gpu(mesh_GPU_new);
	FEM_solver->reconstruct_FEM_matrix(mesh_GPU_new);
	check_cuda_error("Remesh_set_mesh_gpu\n");
}


void tool_FEM::FEM_tool_setup_wear_constants(wear_constants wear)
{
	cudaMemcpyToSymbol(wear_para, &wear, sizeof(wear_constants), 0, cudaMemcpyHostToDevice);
}