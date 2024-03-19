// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// module for the setup of tool

#ifndef TOOL_GPU_H_
#define TOOL_GPU_H_

#include "constants_structs.h"
#include "particle_gpu.h"
#include "tool.h"
#include "types.h"

extern float_t global_dt;

//communcate constants to tool subsystem
void tool_gpu_set_up_tool(tool *tool, float_t alpha, phys_constants phys);

//move tool with specified velocity
void tool_gpu_update_tool(tool *tool, particle_gpu *particles);

//contact and friction force computation
void compute_contact_forces(particle_gpu *particles);

#endif /* TOOL_GPU_H_ */
