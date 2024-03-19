// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "FEM_thermal_gpu.h"
#include <glm/gtc/matrix_integer.hpp>
#include <glm/matrix.hpp>
#include <cuda_runtime_api.h>


#define GLM_FORCE_CUDA

//physical and other simulation constants in constant device memory
__constant__ static tool_constants tc;


__device__ int find_location(const int* base_array, int start, int end, int value) {
	int itr;
	for (itr = start; itr < end; itr++) {
		if (*(base_array + itr) == value) { break; }
	}
	return itr;
}

__device__ mat3x2_t get_element_coordinates_2D(float2_t a, float2_t b, float2_t c)
{
	return mat3x2_t(a.x, a.y, b.x, b.y, c.x, c.y);  // column major order
}

__device__ mat3x2_t get_element_coordinates_2D(const int4 nom, const float2_t* nodes_position)
{
	return mat3x2_t(nodes_position[nom.x].x, nodes_position[nom.x].y, nodes_position[nom.y].x, nodes_position[nom.y].y, nodes_position[nom.z].x, nodes_position[nom.z].y);
}

__device__ float_t get_element_area_2D(mat3x2_t coordinates)
{
	return 0.5 * std::abs((coordinates[1][0] - coordinates[0][0]) * (coordinates[2][1] - coordinates[1][1]) -
		(coordinates[2][0] - coordinates[1][0]) * (coordinates[1][1] - coordinates[0][1]));
}

__device__ float3_t get_element_edge_length_2D(mat3x2_t coordinates)
{
	float3_t length;
	float3_t square;

	square.x = (coordinates[0][0] - coordinates[1][0]) * (coordinates[0][0] - coordinates[1][0]) + (coordinates[0][1] - coordinates[1][1]) * (coordinates[0][1] - coordinates[1][1]);
	square.y = (coordinates[1][0] - coordinates[2][0]) * (coordinates[1][0] - coordinates[2][0]) + (coordinates[1][1] - coordinates[2][1]) * (coordinates[1][1] - coordinates[2][1]);
	square.z = (coordinates[2][0] - coordinates[0][0]) * (coordinates[2][0] - coordinates[0][0]) + (coordinates[2][1] - coordinates[0][1]) * (coordinates[2][1] - coordinates[0][1]);
	length.x = std::pow(square.x, 0.5);
	length.y = std::pow(square.y, 0.5);
	length.z = std::pow(square.z, 0.5);

	return length;
}

__device__ float_t get_segment_length(float2_t left, float2_t right)
{
	float_t dist;
	dist = (left.x - right.x) * (left.x - right.x) + (left.y - right.y) * (left.y - right.y);
	return sqrtf(dist);
}

__device__ float3_t do_mass_ele_matrix_construction_p1q3(float_t determinant, mat3x3_t pre_comp_ref_shape,
	float_t weights) {

	// Parametric finite element methods for 1st order elements with 3 quadrature points. 
	// Higher order elements can be extended based on this form.
	float3_t ele_mass_mat{ 0., 0., 0. };
	float_t value = weights * tc.cp * tc.rho * determinant;

	// for the completeness, the reference shape function at each quadrature points should be 
	// (100 000 000) for quadrature point 1, (000 010 000) for quadrature point 2, (000 000 001) for quadrature point 3
    // this is wrong! there should be only one quadrature point

	// assembly by quadrature points
	// Integration by quadrature points, simplified step, as the matrix is lumped here  (no barycentric coordinate integration, cumbersome)
	ele_mass_mat.x = value;
	ele_mass_mat.y = value;
	ele_mass_mat.z = value;

	return ele_mass_mat;

}

__device__ mat3x3_t do_stiffness_ele_matrix_construction_p1q3(mat2x2_t jacobian, float_t determinant, mat2x3_t pre_comp_ref_grad,
	float_t weights) {

	// Parametric finite element methods for 1st order elements with 3 quadrature points. 
	// 2024.02.06 it should be p1q2, as in principle it should be 1 quadrature point in the middle
	// Anyway, as all the precomputed quantities are constants, so in the end, the result is the same
	// Higher order elements can be extended based on this form.
	mat3x3_t ele_stiffness_mat(0.);
	mat2x2_t jacobian_transpose_inverse = glm::inverse(glm::transpose(jacobian));
    // same weight and value for three quadrature points, so no loop is implemented here
	float_t value = tc.k * determinant;
	mat3x2_t jac_inv_tra_shape_grad = jacobian_transpose_inverse * glm::transpose(pre_comp_ref_grad);
	ele_stiffness_mat += weights * 3. * value * glm::transpose(jac_inv_tra_shape_grad) * jac_inv_tra_shape_grad;
    // here should be one quadrature point with weight as 1
	return ele_stiffness_mat;

}

__device__ float2_t lumped_edge_mass_element_matrix_p1q3(float_t dist)
{
	// direct calculation, no need to use the parametric method, shape function here actually is [0.5, 0.5]
	float2_t edge_ele_mass_matrix;
	edge_ele_mass_matrix.x = 0.5 * dist;
	edge_ele_mass_matrix.y = 0.5 * dist;
	return edge_ele_mass_matrix;
}

__global__ void do_cpy_vector(const int N, const int* vec, int* vec_dest) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	vec_dest[idx] = vec[idx];
}

__global__ void do_stiffness_mass_matrix_construction_p1q3(int i, const int N, const float2_t* position, const int4* __restrict__ node_num,
	const int* __restrict__ csr_row_ptr, const int* __restrict__ csr_col_ind, const int color_number,
	float_t* m_mass_matrix, float_t* m_stiffness_matrix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N) return;

	int4 nodes_in_element = node_num[idx];
	if (nodes_in_element.w != i) return;

	mat3x2_t ReferCoord(0., 0., 1., 0., 0., 1.);   // not used
	float_t  weights = 1. / 3. * 0.5;               // 1st order, three gauss points have the same weigts. 0.5 is for the reference element area
	mat3x3_t pre_comp_ref_shape(1., 0., 0., 0., 1., 0., 0., 0., 1.);     // (1 - x - y, x, y), for each quadrature points
	mat2x3_t pre_comp_ref_grad(-1., 1., 0., -1., 0., 1.);

	mat3x2_t coordinates = get_element_coordinates_2D(nodes_in_element, position);

	mat2x2_t jacobian;    // The Jacobians are the same for three quadrature points, so here only one jacobian value is defined
	jacobian[0] = coordinates[1] - coordinates[0];
	jacobian[1] = coordinates[2] - coordinates[0];
	//float_t jacobian_1 = jacobian[0][0];
	float_t determinant = abs(jacobian[0][0]*jacobian[1][1] - jacobian[1][0]*jacobian[0][1]);

	// get the material property here!!!!!!!!!!!!!
	// 1st order element, higher order elements can be implemented in the future, but not necessary for the heat conduction
	float3_t mass_ele_mat = do_mass_ele_matrix_construction_p1q3(determinant, pre_comp_ref_shape, weights);
	mat3x3_t stiffness_ele_mat = do_stiffness_ele_matrix_construction_p1q3(jacobian, determinant, pre_comp_ref_grad, weights);

	__syncthreads();

	// assemble to the inverse global mass matrix
	m_mass_matrix[nodes_in_element.x] += mass_ele_mat.x;
	m_mass_matrix[nodes_in_element.y] += mass_ele_mat.y;
	m_mass_matrix[nodes_in_element.z] += mass_ele_mat.z;

	int a = 3;
	int b = 3;
	int c = 3;

#pragma unroll
	for (int j = csr_row_ptr[nodes_in_element.x]; j < csr_row_ptr[nodes_in_element.x + 1]; j++) {
		if (csr_col_ind[j] == (nodes_in_element.x)) {
			m_stiffness_matrix[j] += stiffness_ele_mat[0][0];
			a--;
			continue;
		}
		if (csr_col_ind[j] == (nodes_in_element.y)) {
			m_stiffness_matrix[j] += stiffness_ele_mat[1][0];
			a--;
			continue;
		}
		if (csr_col_ind[j] == (nodes_in_element.z)) {
			m_stiffness_matrix[j] += stiffness_ele_mat[2][0];
			a--;
			continue;
		}
		if (a == 0) { break; }
	}
#pragma unroll
	for (int k = csr_row_ptr[nodes_in_element.y]; k < csr_row_ptr[nodes_in_element.y + 1]; k++) {
		if (csr_col_ind[k] == (nodes_in_element.x)) {
			m_stiffness_matrix[k] += stiffness_ele_mat[0][1];
			b--;
			continue;
		}
		if (csr_col_ind[k] == (nodes_in_element.y)) {
			m_stiffness_matrix[k] += stiffness_ele_mat[1][1];
			b--;
			continue;
		}
		if (csr_col_ind[k] == (nodes_in_element.z)) {
			m_stiffness_matrix[k] += stiffness_ele_mat[2][1];
			b--;
			continue;
		}
		if (b == 0) { break; }
	}
#pragma unroll
	for (int l = csr_row_ptr[nodes_in_element.z]; l < csr_row_ptr[nodes_in_element.z + 1]; l++) {
		if (csr_col_ind[l] == (nodes_in_element.x)) {
			m_stiffness_matrix[l] += stiffness_ele_mat[0][2];
			c--;
			continue;
		}
		if (csr_col_ind[l] == (nodes_in_element.y)) {
			m_stiffness_matrix[l] += stiffness_ele_mat[1][2];
			c--;
			continue;
		}
		if (csr_col_ind[l] == (nodes_in_element.z)) {
			m_stiffness_matrix[l] += stiffness_ele_mat[2][2];
			c--;
			continue;
		}
		if (c == 0) { break; }
	}

	assert(a == b == c == 0);

}

__global__ void do_rhs_vector_construction_p1q3_zero(const int N, float_t* m_rhs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	m_rhs[idx] = 0.0;
}

__global__ void do_dirichlet_bdc_p1q3(int i, const int N, const int color_number, const float2_t* position, const int4* node_num, const int3* fixed_boundary_flags,
	const int* fixed_boundary_elements, const int* csr_row_ptr, const int* csr_col_ind, float_t* m_mass_matrix,
	float_t* m_stiffness_matrix, float_t* phi, float_t* T)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;

	int ele_num = fixed_boundary_elements[idx];
	int4 node_in_ele = node_num[ele_num];
	if (node_in_ele.w != i) return;
	int3 flag = fixed_boundary_flags[ele_num];
	mat3x2_t coordinates = get_element_coordinates_2D(node_in_ele, position);
	float3_t edge_length = get_element_edge_length_2D(coordinates);

	int3 node_flag = { 0, 0, 0 };
	if (flag.x != 0) {
		node_flag.x = 1;
		node_flag.y = 1;
	}
	if (flag.y != 0) {
		node_flag.y = 1;
		node_flag.z = 1;
	}
	if (flag.z != 0) {
		node_flag.z = 1;
		node_flag.x = 1;
	}

	int local_no;
	int local_no_sym;

	if (node_flag.x == 1) {

		// Find the non zeros in the stiffness and mass matrix, and the rhs vector. 
		// The problem is simplified since the lumped mass matrix is used			

		int nz_in_line_start = csr_row_ptr[node_in_ele.x];
		int nz_in_line_end = csr_row_ptr[node_in_ele.x + 1];
		for (int i = nz_in_line_start; i < nz_in_line_end; i++) {
			local_no = csr_col_ind[i];

			atomicAdd(&phi[local_no], -m_stiffness_matrix[i] * tc.T0);
			m_stiffness_matrix[i] = 0.;
			local_no_sym = find_location(csr_col_ind, csr_row_ptr[local_no], csr_row_ptr[local_no + 1], node_in_ele.x);
			m_stiffness_matrix[local_no_sym] = 0.;
			if (local_no == node_in_ele.x) { m_stiffness_matrix[i] = 1.0; }
		}
		m_mass_matrix[node_in_ele.x] = 1.0;
		phi[node_in_ele.x] = tc.T0;
		T[node_in_ele.x] = tc.T0;
	}

	if (node_flag.y == 1) {

		int nz_in_line_start = csr_row_ptr[node_in_ele.y];
		int nz_in_line_end = csr_row_ptr[node_in_ele.y + 1];

		for (int i = nz_in_line_start; i < nz_in_line_end; i++) {
			local_no = csr_col_ind[i];

			atomicAdd(&phi[local_no], -m_stiffness_matrix[i] * tc.T0);
			m_stiffness_matrix[i] = 0.;
			local_no_sym = find_location(csr_col_ind, csr_row_ptr[local_no], csr_row_ptr[local_no + 1], node_in_ele.y);
			m_stiffness_matrix[local_no_sym] = 0.;
			if (local_no == node_in_ele.y) { m_stiffness_matrix[i] = 1.0; }
		}
		m_mass_matrix[node_in_ele.y] = 1.0;
		phi[node_in_ele.y] = tc.T0;
		T[node_in_ele.y] = tc.T0;
	}

	if (node_flag.z == 1) {

		int nz_in_line_start = csr_row_ptr[node_in_ele.z];
		int nz_in_line_end = csr_row_ptr[node_in_ele.z + 1];

		for (int i = nz_in_line_start; i < nz_in_line_end; i++) {
			local_no = csr_col_ind[i];

			atomicAdd(&phi[local_no], -m_stiffness_matrix[i] * tc.T0);
			m_stiffness_matrix[i] = 0.;
			local_no_sym = find_location(csr_col_ind, csr_row_ptr[local_no], csr_row_ptr[local_no + 1], node_in_ele.z);
			m_stiffness_matrix[local_no_sym] = 0.;
			if (local_no == node_in_ele.z) { m_stiffness_matrix[i] = 1.0; }
		}
		m_mass_matrix[node_in_ele.z] = 1.0;
		phi[node_in_ele.z] = tc.T0;
		T[node_in_ele.z] = tc.T0;
	}

}

__global__ void do_inverse_mass_matrix_lumped(const int N, float_t* m_mass_matrix_inverse, float_t* m_mass_matrix)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	m_mass_matrix_inverse[idx] = 1. / m_mass_matrix[idx];

}

__global__ void do_copy_vector(int N, float_t* f, float_t* f_temp){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	f_temp[idx] = f[idx];

}

#ifndef SURFACE
__global__ void do_apply_heat_boundary_condition(int i, int N, int4* marks, float2_t* heat_exchange, float_t* fric_heat, float2_t* left, float2_t* right,  int* row_ptr, int* col_ind, float_t* stiffness_matrix, float_t* phi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (marks[idx].y != i) return;          // doesn't match the color
	if (heat_exchange[idx].y == 0) return;  // not in contact with particles


	float_t T_cont_particle = heat_exchange[idx].x / heat_exchange[idx].y;
	float_t fric_eng = fric_heat[idx];
	//if (fric_heat[idx].x != 0) fric_eng = fric_heat[idx].x;
	int l = marks[idx].z;
	int r = marks[idx].w;
	float_t dist = get_segment_length(left[idx], right[idx]);

	// adjust the rhs vector
	atomicAdd(&phi[l], 0.5 * tc.h * dist * T_cont_particle + 0.5 * fric_eng);
	atomicAdd(&phi[r], 0.5 * tc.h * dist * T_cont_particle + 0.5 * fric_eng);

	// adjust the stiffness matrix. here we use the lumped mass element matrix for the convection term
	float2_t edge_mass_matrix = lumped_edge_mass_element_matrix_p1q3(dist);

	int nz_in_line_start = row_ptr[l];
	int nz_in_line_end = row_ptr[l + 1];
	int local_no;
	for (int k = nz_in_line_start; k < nz_in_line_end; k++) {
		local_no = col_ind[k];
		if (local_no == l) { 
			stiffness_matrix[k] += tc.h * edge_mass_matrix.x; 
			break;
		}
	}
	nz_in_line_start = row_ptr[r];
	nz_in_line_end = row_ptr[r + 1];
	for (int k = nz_in_line_start; k < nz_in_line_end; k++) {
		local_no = col_ind[k];
		if (local_no == r) { 
			stiffness_matrix[k] += tc.h * edge_mass_matrix.y; 
			break;
		}
	}

}
#endif

#ifdef SURFACE
__global__ void do_apply_heat_boundary_condition(int i, int N, int4* marks, float2_t* heat_exchange, float_t* fric_heat, float2_t* left, float2_t* right,  int* row_ptr, int* col_ind, float_t* stiffness_matrix, float_t* phi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (marks[idx].y != i) return;          // doesn't match the color

    float_t h_conv;
	float_t T_cont_particle;
	float_t fric_eng;
    if (heat_exchange[idx].y != 0){
      T_cont_particle = heat_exchange[idx].x / heat_exchange[idx].y;
	  h_conv = tc.h;
	  fric_eng = fric_heat[idx];
	} else {
	  T_cont_particle = tc.T0;
	  h_conv = tc.hc;    
	  fric_eng = 0.;
	}

  	
	//if (fric_heat[idx].x != 0) fric_eng = fric_heat[idx].x;
	int l = marks[idx].z;
	int r = marks[idx].w;
	float_t dist = get_segment_length(left[idx], right[idx]);

	// adjust the rhs vector
	atomicAdd(&phi[l], 0.5 * h_conv * dist * T_cont_particle + 0.5 * fric_eng);
	atomicAdd(&phi[r], 0.5 * h_conv * dist * T_cont_particle + 0.5 * fric_eng);

	// adjust the stiffness matrix. here we use the lumped mass element matrix for the convection term
	float2_t edge_mass_matrix = lumped_edge_mass_element_matrix_p1q3(dist);

	int nz_in_line_start = row_ptr[l];
	int nz_in_line_end = row_ptr[l + 1];
	int local_no;
	for (int k = nz_in_line_start; k < nz_in_line_end; k++) {
		local_no = col_ind[k];
		if (local_no == l) { 
			stiffness_matrix[k] += h_conv* edge_mass_matrix.x; 
			break;
		}
	}
	nz_in_line_start = row_ptr[r];
	nz_in_line_end = row_ptr[r + 1];
	for (int k = nz_in_line_start; k < nz_in_line_end; k++) {
		local_no = col_ind[k];
		if (local_no == r) { 
			stiffness_matrix[k] += h_conv * edge_mass_matrix.y; 
			break;
		}
	}

}
#endif

__global__ void do_calculate_temp_rate(const int N, const float_t* m_mass_matrix_inverse, const float_t* m_stiffness_matrix, const int* m_row_ptr, const int* m_col_ind,
	float_t* f, float_t* k, float_t* T) {

	// one thread per row
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	float_t buffer = 0.;

	for (int i = m_row_ptr[idx]; i < m_row_ptr[idx + 1]; i++) {
		buffer += m_stiffness_matrix[i] * T[m_col_ind[i]];
	}

	k[idx] = m_mass_matrix_inverse[idx] * (f[idx] - buffer);


}

__global__ void do_calculate_temp(const int N, float_t* k, float_t* T, float_t* T_old, float_t dt) {

	// one thread per row
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	T[idx] = dt * k[idx] + T_old[idx];


}

__global__ void do_set_thermal_vector(int N, float_t* T, float_t* T_old) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	T[idx] = T_old[idx];
}

__global__ void do_mesh_adjustment_wear(int i, int N, const int2* cutting_boundary_layer, const float2_t* position,
	const int4* __restrict__ node_num,	const int* __restrict__ csr_row_ptr, const int* __restrict__ csr_col_ind, const int color_number,
	float_t* m_mass_matrix, float_t* m_stiffness_matrix, int ope_sign)
{
#ifdef WEAR_NODE_SHIFT
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	if (cutting_boundary_layer[idx].x == 0 && cutting_boundary_layer[idx].y == 0) return;

	int4 nodes_in_element = node_num[idx];
	if (nodes_in_element.w != i) return;
	

	float_t  weights = 1. / 3. * 0.5;                // 1st order, three gauss points have the same weigts
	mat3x3_t pre_comp_ref_shape(1., 0., 0., 0., 1., 0., 0., 0., 1.);     // (1 - x - y, x, y), for each quadrature points
	mat2x3_t pre_comp_ref_grad(-1., 1., 0., -1., 0., 1.);

	mat3x2_t coordinates = get_element_coordinates_2D(nodes_in_element, position);

	mat2x2_t jacobian;    // The Jacobians are the same for three quadrature points, so here only one jacobian value is defined
	jacobian[0] = coordinates[1] - coordinates[0];
	jacobian[1] = coordinates[2] - coordinates[0];
	//float_t jacobian_1 = jacobian[0][0];
	float_t determinant = abs(jacobian[0][0]*jacobian[1][1] - jacobian[1][0]*jacobian[0][1]);

	// get the material property here!!!!!!!!!!!!!
	// 1st order element, higher order elements can be implemented in the future, but not necessary for the heat conduction
	float3_t mass_ele_mat = do_mass_ele_matrix_construction_p1q3(determinant, pre_comp_ref_shape, weights);
	mat3x3_t stiffness_ele_mat = do_stiffness_ele_matrix_construction_p1q3(jacobian, determinant, pre_comp_ref_grad, weights);

	__syncthreads();


	// assemble to the inverse global mass matrix
	m_mass_matrix[nodes_in_element.x] += ope_sign * mass_ele_mat.x;
	m_mass_matrix[nodes_in_element.y] += ope_sign * mass_ele_mat.y;
	m_mass_matrix[nodes_in_element.z] += ope_sign * mass_ele_mat.z;

	int a = 3;
	int b = 3;
	int c = 3;

#pragma unroll
	for (int j = csr_row_ptr[nodes_in_element.x]; j < csr_row_ptr[nodes_in_element.x + 1]; j++) {
		if (csr_col_ind[j] == (nodes_in_element.x)) {
			m_stiffness_matrix[j] += ope_sign * stiffness_ele_mat[0][0];
			a--;
			continue;
		}
		if (csr_col_ind[j] == (nodes_in_element.y)) {
			m_stiffness_matrix[j] += ope_sign * stiffness_ele_mat[1][0];
			a--;
			continue;
		}
		if (csr_col_ind[j] == (nodes_in_element.z)) {
			m_stiffness_matrix[j] += ope_sign * stiffness_ele_mat[2][0];
			a--;
			continue;
		}
		if (a == 0) { break; }
	}
#pragma unroll
	for (int k = csr_row_ptr[nodes_in_element.y]; k < csr_row_ptr[nodes_in_element.y + 1]; k++) {
		if (csr_col_ind[k] == (nodes_in_element.x)) {
			m_stiffness_matrix[k] += ope_sign * stiffness_ele_mat[0][1];
			b--;
			continue;
		}
		if (csr_col_ind[k] == (nodes_in_element.y)) {
			m_stiffness_matrix[k] += ope_sign * stiffness_ele_mat[1][1];
			b--;
			continue;
		}
		if (csr_col_ind[k] == (nodes_in_element.z)) {
			m_stiffness_matrix[k] += ope_sign * stiffness_ele_mat[2][1];
			b--;
			continue;
		}
		if (b == 0) { break; }
	}
#pragma unroll
	for (int l = csr_row_ptr[nodes_in_element.z]; l < csr_row_ptr[nodes_in_element.z + 1]; l++) {
		if (csr_col_ind[l] == (nodes_in_element.x)) {
			m_stiffness_matrix[l] += ope_sign * stiffness_ele_mat[0][2];
			c--;
			continue;
		}
		if (csr_col_ind[l] == (nodes_in_element.y)) {
			m_stiffness_matrix[l] += ope_sign * stiffness_ele_mat[1][2];
			c--;
			continue;
		}
		if (csr_col_ind[l] == (nodes_in_element.z)) {
			m_stiffness_matrix[l] += ope_sign * stiffness_ele_mat[2][2];
			c--;
			continue;
		}
		if (c == 0) { break; }
	}

#endif
}


FEM_thermal_2D_GPU::FEM_thermal_2D_GPU()
{
}

FEM_thermal_2D_GPU::FEM_thermal_2D_GPU(mesh_GPU_m* mesh, tool_constants tc_h)
{
	T_env = tc_h.T0;
	
	node_num = mesh->get_node_number();
	ele_num = mesh->get_element_number();
	nz_num = mesh->get_non_zero_number();
	color_total = mesh->get_color_number();
	color_cutting_edge = mesh->get_cutting_edge_color_number();
	fixed_ele_num = mesh->get_fixed_edge_element_number();
	cutting_ele_num = mesh->get_cutting_edge_element_number();


	cudaMemcpyToSymbol(tc, &tc_h, sizeof(tool_constants), 0, cudaMemcpyHostToDevice);
	initialization_system(mesh);  // In this specific problem, the source term in the rhs is zero.
	stiffness_mass_matrix_construction(mesh);
	dirichlet_bdc(mesh);
	inverse_mass_matrix();

}

void FEM_thermal_2D_GPU::initialization_system(mesh_GPU_m* m_mesh)
{
	// CSR format based storage
	cudaMalloc((void**)&m_row_ptr, sizeof(int) * (node_num + 1));
	cudaMalloc((void**)&m_col_ind, sizeof(int) * nz_num);
	cudaMalloc((void**)&m_stiffness_matrix, sizeof(float_t) * nz_num);
	cudaMalloc((void**)&m_stiffness_matrix_temp, sizeof(float_t) * nz_num);
	cudaMalloc((void**)&m_mass_matrix, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_mass_matrix_inverse, sizeof(float_t) * node_num);    // use lumped mass matrix
	cudaMalloc((void**)&m_rhs, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_f, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_f_temp, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_k, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_T, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_T_old, sizeof(float_t) * node_num);
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**)&m_T_mapping, sizeof(float_t) * node_num);
#endif

	// initialization 
	float_t* node_zeros = new float_t[node_num];
	float_t* nz_zeros = new float_t[nz_num];
	float_t* node_T = new float_t[node_num];
	for (int i = 0; i < node_num; i++) {
		node_zeros[i] = 0.;
		node_T[i] = T_env;
	}
	for (int j = 0; j < nz_num; j++) {
		nz_zeros[j] = 0.;
	}

	cudaMemcpy(m_row_ptr, m_mesh->csr_row_ptr, sizeof(int) * (node_num + 1), cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_col_ind, m_mesh->csr_col_ind, sizeof(int) * (nz_num), cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_stiffness_matrix, nz_zeros, sizeof(float_t) * (nz_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_mass_matrix, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);

	cudaMemcpy(m_f, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_k, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);

#ifdef USE_HOT_TOOL
	cudaMemcpy(m_T, m_mesh->T_nodes, sizeof(float_t) * (node_num), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_T_old, m_mesh->T_nodes, sizeof(float_t) * (node_num), cudaMemcpyDeviceToDevice);
#else
	cudaMemcpy(m_T, node_T, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
    cudaMemcpy(m_T_old, node_T, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
#endif

#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(m_T_mapping, node_T, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
#endif
	delete[] node_zeros;
	delete[] nz_zeros;
	delete[] node_T;

}

void FEM_thermal_2D_GPU::set_zero(mesh_GPU_m* m_mesh){

	// initialization 
	float_t* node_zeros = new float_t[node_num];
	float_t* nz_zeros = new float_t[nz_num];

	for (int i = 0; i < node_num; i++) {
		node_zeros[i] = 0.;
	}
	for (int j = 0; j < nz_num; j++) {
		nz_zeros[j] = 0.;
	}

	cudaMemcpy(m_stiffness_matrix, nz_zeros, sizeof(float_t) * (nz_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_mass_matrix, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);

	cudaMemcpy(m_f, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_k, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);

	delete[] node_zeros;
	delete[] nz_zeros;

		int* csr_row = new int[node_num + 1];
	int* csr_col = new int[nz_num];
	float_t* csr_val = new float_t[nz_num];
	cudaMemcpy(csr_row, m_row_ptr, sizeof(int) * (node_num + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_col, m_col_ind, sizeof(int) * (nz_num), cudaMemcpyDeviceToHost);
	cudaMemcpy(csr_val, m_stiffness_matrix, sizeof(float_t) * (nz_num), cudaMemcpyDeviceToHost);


	for (int i = 0; i < node_num; i++) {
		printf("row index %u: \n", i);
		for (int j = csr_row[i]; j < csr_row[i + 1]; j++) {
			printf("node %u -- %.4e ,", csr_col[j], csr_val[j]);
		}
		printf("\n");
	}


}

void FEM_thermal_2D_GPU::reset_FEM_matrix(mesh_GPU_m* m_mesh)
{
	// initialization 
	float_t* node_zeros = new float_t[node_num];
	float_t* nz_zeros = new float_t[nz_num];
	for (int i = 0; i < node_num; i++) {
		node_zeros[i] = 0.;
	}
	for (int j = 0; j < nz_num; j++) {
		nz_zeros[j] = 0.;
	}

	cudaMemcpy(m_stiffness_matrix, nz_zeros, sizeof(float_t) * (nz_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_mass_matrix, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_f, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);

	delete[] node_zeros;
	delete[] nz_zeros;
}

void FEM_thermal_2D_GPU::reconstruct_FEM_matrix(mesh_GPU_m* mesh)
{
	
	node_num = mesh->get_node_number();
	ele_num = mesh->get_element_number();
	nz_num = mesh->get_non_zero_number();
	color_total = mesh->get_color_number();
	color_cutting_edge = mesh->get_cutting_edge_color_number();
	fixed_ele_num = mesh->get_fixed_edge_element_number();
	cutting_ele_num = mesh->get_cutting_edge_element_number();

	cudaFree(m_row_ptr);
	cudaFree(m_col_ind);
	cudaFree(m_stiffness_matrix);
	cudaFree(m_stiffness_matrix_temp);
	cudaFree(m_mass_matrix);
	cudaFree(m_mass_matrix_inverse);
	cudaFree(m_rhs);
	cudaFree(m_f);
	cudaFree(m_f_temp);
	cudaFree(m_k);
	cudaFree(m_T);
	cudaFree(m_T_old);
#ifdef WEAR_NODE_SHIFT
	cudaFree(m_T_mapping);
#endif
	
	// CSR format based storage
	cudaMalloc((void**)&m_row_ptr, sizeof(int) * (node_num + 1));
	cudaMalloc((void**)&m_col_ind, sizeof(int) * nz_num);
	cudaMalloc((void**)&m_stiffness_matrix, sizeof(float_t) * nz_num);
	cudaMalloc((void**)&m_stiffness_matrix_temp, sizeof(float_t) * nz_num);
	cudaMalloc((void**)&m_mass_matrix, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_mass_matrix_inverse, sizeof(float_t) * node_num);    // use lumped mass matrix
	cudaMalloc((void**)&m_rhs, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_f, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_f_temp, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_k, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_T, sizeof(float_t) * node_num);
	cudaMalloc((void**)&m_T_old, sizeof(float_t) * node_num);
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**)&m_T_mapping, sizeof(float_t) * node_num);
#endif

	// initialization 
	float_t* node_zeros = new float_t[node_num];
	float_t* nz_zeros = new float_t[nz_num];
	for (int i = 0; i < node_num; i++) {
		node_zeros[i] = 0.;
	}
	for (int j = 0; j < nz_num; j++) {
		nz_zeros[j] = 0.;
	}

	cudaMemcpy(m_row_ptr, mesh->csr_row_ptr, sizeof(int) * (node_num + 1), cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_col_ind, mesh->csr_col_ind, sizeof(int) * (nz_num), cudaMemcpyDeviceToDevice);
	cudaMemcpy(m_stiffness_matrix, nz_zeros, sizeof(float_t) * (nz_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_mass_matrix, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);

	cudaMemcpy(m_f, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);
	cudaMemcpy(m_k, node_zeros, sizeof(float_t) * (node_num), cudaMemcpyHostToDevice);


	cudaMemcpy(m_T, mesh->T_nodes, sizeof(float_t) * (node_num), cudaMemcpyDeviceToDevice);
    cudaMemcpy(m_T_old, mesh->T_nodes, sizeof(float_t) * (node_num), cudaMemcpyDeviceToDevice);


	delete[] node_zeros;
	delete[] nz_zeros;

	stiffness_mass_matrix_construction(mesh);
	dirichlet_bdc(mesh);
	inverse_mass_matrix();
}

void FEM_thermal_2D_GPU::stiffness_mass_matrix_construction(mesh_GPU_m* m_mesh)
{
	int N = ele_num;

	// one thread per element

	dim3 dB(FEM_ASSEMB_SIZE);
	dim3 dG((N + FEM_ASSEMB_SIZE - 1) / FEM_ASSEMB_SIZE);
	cudaEvent_t assembly;
	cudaEventCreate(&assembly);
	for (int i = 1; i < color_total + 1; i++) {

		do_stiffness_mass_matrix_construction_p1q3 << <dG, dB >> > (i, N, m_mesh->position, m_mesh->node_num, m_mesh->csr_row_ptr,
			m_mesh->csr_col_ind, color_total, m_mass_matrix, m_stiffness_matrix);
		cudaEventSynchronize(assembly);
	}


}

void FEM_thermal_2D_GPU::rhs_vector_construction_p1q3_zero(mesh_GPU_m* m_mesh)
{
	dim3 dB(FEM_ASSEMB_SIZE);
	dim3 dG((node_num + FEM_ASSEMB_SIZE - 1) / FEM_ASSEMB_SIZE);
	do_rhs_vector_construction_p1q3_zero << <dG, dB >> > (node_num, m_f);
}

void FEM_thermal_2D_GPU::dirichlet_bdc(mesh_GPU_m* m_mesh)
{
	int N = fixed_ele_num;
	dim3 dB(FEM_ASSEMB_SIZE);
	dim3 dG((N + FEM_ASSEMB_SIZE - 1) / FEM_ASSEMB_SIZE);
	cudaEvent_t dirichlet_bdc;
	cudaEventCreate(&dirichlet_bdc);
	for (int i = 1; i < color_total + 1; i++) {
		do_dirichlet_bdc_p1q3 << <dG, dB >> > (i, N, color_total, m_mesh->position, m_mesh->node_num, m_mesh->fixed_boundary_flags,
			m_mesh->fixed_boundary_elements, m_mesh->csr_row_ptr, m_mesh->csr_col_ind, m_mass_matrix, m_stiffness_matrix, m_f, m_T_old);
		cudaEventSynchronize(dirichlet_bdc);
	}
}

void FEM_thermal_2D_GPU::inverse_mass_matrix()
{
	int N = node_num;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	do_inverse_mass_matrix_lumped << <dG, dB >> > (N, m_mass_matrix_inverse, m_mass_matrix);


	
}

void FEM_thermal_2D_GPU::set_temp_stiffness_matrix()
{
	int N = nz_num;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	do_copy_vector << <dG, dB >> > (N, m_stiffness_matrix, m_stiffness_matrix_temp);
}

void FEM_thermal_2D_GPU::set_temp_rhs_vector()
{
	int N = node_num;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	do_copy_vector << <dG, dB >> > (N, m_f, m_f_temp);
}

void FEM_thermal_2D_GPU::apply_robin_bdc(segment_FEM_gpu* segments, int seg_size)
{
	
	dim3 dB(FEM_ASSEMB_SIZE);
	dim3 dG((seg_size + FEM_ASSEMB_SIZE - 1) / FEM_ASSEMB_SIZE);
	//cudaEvent_t heat_bdcs;
	//cudaEventCreate(&heat_bdcs);
	for (int i = 1; i < color_cutting_edge + 1; i++) {
		do_apply_heat_boundary_condition << <dG, dB >> > (i, seg_size, segments->marks, segments->heat_exchange, segments->fric_heat, segments->left, segments->right,
			m_row_ptr, m_col_ind, m_stiffness_matrix_temp, m_f_temp);
		//cudaEventSynchronize(heat_bdcs);
	}
}

void FEM_thermal_2D_GPU::calculate_temp_rate()
{
	int N = node_num;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);

	//cudaEvent_t temp_rate;
	//cudaEventCreate(&temp_rate);
	do_calculate_temp_rate << <dG, dB >> > (N, m_mass_matrix_inverse, m_stiffness_matrix_temp, m_row_ptr, m_col_ind, m_f_temp, m_k, m_T);
	//cudaEventSynchronize(temp_rate);
}

void FEM_thermal_2D_GPU::calculate_temp(float_t dt)
{
	int N = this->node_num;
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	//cudaEvent_t temp_cal;
	//cudaEventCreate(&temp_cal);
	do_calculate_temp << <dG, dB >> > (N, this->m_k, this->m_T, this->m_T_old, dt);
	//cudaEventSynchronize(temp_cal);
}

void FEM_thermal_2D_GPU::reset_FEM_server()
{
	int N = node_num;
	int NZ = nz_num;
	
	dim3 dB(FEM_BLOCK_SIZE);
	dim3 dG((N + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	//cudaEvent_t set_vector;
	//cudaEventCreate(&set_vector);
	do_copy_vector << <dG, dB >> > (N, m_T, m_T_old);
	//cudaEventSynchronize(set_vector);
	do_copy_vector << <dG, dB >> > (N, m_f, m_f_temp);
	//cudaEventSynchronize(set_vector);


	dim3 dG1((NZ + FEM_BLOCK_SIZE - 1) / FEM_BLOCK_SIZE);
	do_copy_vector << <dG1, dB >> > (NZ, m_stiffness_matrix, m_stiffness_matrix_temp);
	//cudaEventSynchronize(set_vector);

}

void FEM_thermal_2D_GPU::mesh_retreat_wear(mesh_GPU_m* m_mesh)
{
	//int N = cutting_ele_num;
#ifdef WEAR_NODE_SHIFT
	int N = ele_num;
	dim3 dB(FEM_ASSEMB_SIZE);
	dim3 dG((N + FEM_ASSEMB_SIZE - 1) / FEM_ASSEMB_SIZE);
	int sign_ope = -1;
	for (int i = 1; i < color_total + 1; i++) {	
		do_mesh_adjustment_wear << < dG, dB >> > (i, N, m_mesh->cutting_boundary_layer, m_mesh->position, m_mesh->node_num, m_mesh->csr_row_ptr,
			m_mesh->csr_col_ind, color_total, m_mass_matrix, m_stiffness_matrix, sign_ope);
	}
#endif
}

void FEM_thermal_2D_GPU::mesh_resume_wear(mesh_GPU_m* m_mesh)
{
	//int N = cutting_ele_num;
#ifdef WEAR_NODE_SHIFT
	int N = ele_num;
	dim3 dB(FEM_ASSEMB_SIZE);
	dim3 dG((N + FEM_ASSEMB_SIZE - 1) / FEM_ASSEMB_SIZE);
	int sign_ope = 1;
	for (int i = 1; i < color_total + 1; i++) {
		do_mesh_adjustment_wear << < dG, dB >> > (i, N, m_mesh->cutting_boundary_layer, m_mesh->position, m_mesh->node_num, m_mesh->csr_row_ptr,
			m_mesh->csr_col_ind, color_total, m_mass_matrix, m_stiffness_matrix, sign_ope);
	}
#endif
}

float_t* FEM_thermal_2D_GPU::T_vector_transfer()
{
	int N = node_num;
	float_t* T_h = new float_t[N];
	cudaEvent_t transfer;
	cudaEventCreate(&transfer);
	cudaMemcpy(T_h, m_T, sizeof(float_t) * (N), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(transfer);
	return T_h;
}
