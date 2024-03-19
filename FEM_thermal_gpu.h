// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// class related initial-boundary value transient thermal problem

#ifndef FEM_THE_GPU
#define FEM_THE_GPU


#include <cmath>
#include <fstream> 
#include <iostream>
#include <vector>
#include <algorithm>    // std::remove_if


#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "device_launch_parameters.h"

#include "mesh_structure_GPU.h"
#include "constants_structs.h"
#include "geometric_primitives.h"
#include "segment_FEM_gpu.h"



class FEM_thermal_2D_GPU {


public:
	int node_num;
	int ele_num;
	int nz_num;
	int color_total;
	int color_cutting_edge;
	int fixed_ele_num;
	int cutting_ele_num;
	float_t T_env = 300.;

	int* m_row_ptr;
	int* m_col_ind;

	float_t* m_stiffness_matrix;
	float_t* m_stiffness_matrix_temp;
	float_t* m_mass_matrix;
	float_t* m_mass_matrix_inverse;
	float_t* m_rhs;
	float_t* m_f;
	float_t* m_f_temp;
	float_t* m_k;
	float_t* m_T;
	float_t* m_T_old;

#ifdef WEAR_NODE_SHIFT
	float_t* m_T_mapping;
#endif

	//tool_constants* tc;
	FEM_thermal_2D_GPU();
	FEM_thermal_2D_GPU(mesh_GPU_m* mesh, tool_constants tc_h);

	virtual ~FEM_thermal_2D_GPU() = default;

	// Allocate memory space on the device
	void initialization_system(mesh_GPU_m* m_mesh);
	void set_zero(mesh_GPU_m* m_mesh);
	void reset_FEM_matrix(mesh_GPU_m* m_mesh);
	void reconstruct_FEM_matrix(mesh_GPU_m* m_mesh_new);
	void stiffness_mass_matrix_construction(mesh_GPU_m* m_mesh);
	void rhs_vector_construction_p1q3_zero(mesh_GPU_m* m_mesh);
	void dirichlet_bdc(mesh_GPU_m* m_mesh);
	void inverse_mass_matrix();
	void set_temp_stiffness_matrix();
	void set_temp_rhs_vector();
	void apply_robin_bdc(segment_FEM_gpu* segments, int seg_num);
	void calculate_temp_rate();
	void calculate_temp(float_t dt);
	void reset_FEM_server();

	void mesh_retreat_wear(mesh_GPU_m* m_mesh);
	void mesh_resume_wear(mesh_GPU_m* m_mesh);

	float_t* T_vector_transfer();

};
#endif // !FEM_THE_GPU