// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// this module contains all kernels which perform a single loop over the particle array
//	-correctors (artificial stress)
//	-continuity, momentum and advection equation
//	-material modelling (equation of state for pressure, hookes law + jaumann rate for the deviatoric part of the stress)
//	-boundary conditions
//  -contact algorithm

#ifndef ACTIONS_GPU_H_
#define ACTIONS_GPU_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include "particle_gpu.h"
#include "constants_structs.h"
#include "types.h"
#include "tool_FEM.h"
#include "mesh_structure_GPU.h"

#include <cuda_profiler_api.h>

extern float_t global_dt;

//adiabatic equation of state for pressure term
void material_eos(particle_gpu *particles);
//artificial stresses according to gray & monaghan 2001
void corrector_artificial_stress(particle_gpu *particles);
//stress rate according to jaumann
void material_stress_rate_jaumann(particle_gpu *particles);
//generate frictional heat
void material_fric_heat_gen(particle_gpu *particles, vec2_t vel);
void material_fric_heat_gen_FEM_v2(particle_gpu* particles, segment_FEM_gpu* segments, vec2_t vel, tool_FEM* m_tool);
void compute_contact_forces_FEM_tool_v3(particle_gpu* particles, segment_FEM_gpu* segments, tool_FEM* m_tool);
//void compute_contact_forces_FEM_tool_v4(particle_gpu* particles, segment_FEM_gpu* segments, tool_FEM* m_tool);
void heat_convection_to_particle(particle_gpu* particles, float_t dt);
void wear_adjustment(tool_FEM* m_tool);
void remesh_low_level(tool_FEM* m_tool, int freq);
void remesh_gmsh(tool_FEM* m_tool, int freq);

//basic equations
void contmech_continuity(particle_gpu *particles);
void contmech_momentum(particle_gpu *particles);
void contmech_advection(particle_gpu *particles);

//plasticity using the johnson cook model
//		implements the radial return best described in the UINTAH user manual
void plasticity_johnson_cook(particle_gpu *particles);

//boundary conditions
void perform_boundary_conditions_thermal(particle_gpu *particles);
void perform_boundary_conditions(particle_gpu *particles);

//set up simulation constants
void actions_setup_johnson_cook_constants(joco_constants joco);
void actions_setup_physical_constants(phys_constants phys);
void actions_setup_corrector_constants(corr_constants corr);
void actions_setup_thermal_constants_wp(trml_constants thrm);
void actions_setup_thermal_constants_tool(trml_constants thrm);
void actions_setup_tool_constants(tool_constants tc);
void actions_setup_wear_constants(wear_constants wear);

//debugging (either report or deactivate particles with NaN entries)
void debug_check_valid(particle_gpu *particles);
void debug_invalidate(particle_gpu *particles);

#endif /* ACTIONS_GPU_H_ */
