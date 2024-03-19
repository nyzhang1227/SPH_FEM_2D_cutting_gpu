// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// FEM tool
#ifndef TOOL_FEM
#define TOOL_FEM

#include <glm/glm.hpp>
#include <vector>
#include <stdio.h>
#include <assert.h>

#include "particle_gpu.h"
#include "geometric_primitives.h"
#include "mesh_structure_GPU.h"
#include "tool.h"
#include "FEM_thermal_gpu.h"
#include "segment_FEM_gpu.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"

class tool_FEM : public tool {

public:
	//mesh_read mesh;
	int node_number;
	int element_number;
	int cutting_segment_size;

	vec2_t tl_point_ref;
	float_t ref_dist;

	//int lowest_segment;

	mesh_read* mesh_CPU;
	mesh_GPU_m* mesh_GPU;

	FEM_thermal_2D_GPU* FEM_solver;
	segment_FEM_gpu* segments_FEM;

	vec2_t br_point;

	float_t slave_mass;

	std::vector<vec2_t> get_points();
	float_t lowest_point;
	float_t right_point;

	void set_up_segments_from_mesh();
	void update_segments_remesh(mesh_read* mesh_CPU, mesh_GPU_m* mesh_GPU_new);
	void update_segments_remesh_new(mesh_read* mesh_CPU, mesh_GPU_m* mesh_GPU_new);
	void set_up_FEM_tool_gpu(tool_constants tc_h);
	void apply_heat_boundary_condition();
	void tool_update_FEM_gpu(float_t dt);
	void segments_update(float_t dt);
	void segments_fric_reset();
	void segments_wear_rate_reset();
	void mapping_data_copy();
	void nodal_shift_wear(float_t dt);
	void nodal_value_interpolation();
	void nodal_value_interpolation(mesh_GPU_m* mesh_GPU_new);
	void segment_wear_update();
	void correct_flank_contact();

	void mesh_shift(int freq);
	void remesh_read_segments();
	void read_remesh_gmsh(); 

    void FEM_tool_setup_wear_constants(wear_constants wear);


	tool_FEM(vec2_t tl, float_t length, float_t height, float_t rake_angle, float_t clearance_angle, float_t r, float_t mu_fric);
	tool_FEM(vec2_t tl, float_t length, float_t height, float_t rake_angle, float_t clearance_angle, float_t mu_fric);
	tool_FEM();
	~tool_FEM();

private:
	std::vector<vec2_t> points;  // for identifying tool geometries in abaqus

	// friction coefficient
	float_t m_mu = 0.;

	// velocity of tool
	vec2_t m_velocity = vec2_t(0., 0.);

	// front of the tool
	float_t m_front = 0.;

	circle_segment* m_fillet = 0;
	std::vector<segment> m_segments;

	vec2_t fit_fillet(float_t r, line lm, line l1) const;

	// construct segments from list of points
	void construct_segments(std::vector<vec2_t> list_p);

	std::vector<vec2_t> construct_points_and_fillet(vec2_t tl, vec2_t tr, vec2_t br, vec2_t bl, float_t r);
};

#endif // TOOL_FEM