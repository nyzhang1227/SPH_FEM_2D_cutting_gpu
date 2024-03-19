// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// this module contains all kernels which perform particles interactions
// particle interactions are made efficient using cell lists obtained from spatial hashing (c.f. grid_green.* or grid_rothlin.*)
// this is the heart of both the mechanical and thermal solver

#ifndef INTERACTIONS_GPU_H_
#define INTERACTIONS_GPU_H_

#include "particle_gpu.h"
#include "grid_gpu_green.h"
#include "constants_structs.h"
#include "tool.h"

#include <stdio.h>

extern int global_step;

//communicate grid information to interaction system
//		ATTN: needs to be called before any of the interaciton methods
void interactions_setup_geometry_constants(grid_base *g);

//performs all interactions needed to compute the spatial derivatives according to monaghan, gray 2001 including
//XSPH, artificial viscosity and artificial stresses.
//		NOTE: if Thermal_Conduction_Brookshaw is defined the laplacian of the thermal field is computed in addition
void interactions_monaghan(particle_gpu *particles, const int *cell_start, const int *cell_end, int num_cell);

//performs Particle Strenght Exchange (PSE) to compute the laplacian of the thermal field
//		NOTE: by only called if Thermal_Conduction_PSE is defined
void interactions_heat_pse(particle_gpu *particles, const int *cell_start, const int *cell_end, int num_cell);

//set up simulation constants
void interactions_setup_physical_constants(phys_constants phys);
void interactions_setup_corrector_constants(corr_constants corr);
void interactions_setup_thermal_constants_workpiece(trml_constants trml);
void interactions_setup_thermal_constants_tool(trml_constants trml, tool *tool);
void interactions_setup_tool_constants(tool_constants tc_h);

#endif /* INTERACTIONS_GPU_H_ */
