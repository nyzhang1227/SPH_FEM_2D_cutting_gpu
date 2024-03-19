// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// output to vtk legacy format
// https://www.google.com/search?q=paraview

#ifndef VTK_WRITER_H_
#define VTK_WRITER_H_

#include "particle_gpu.h"
#include "types.h"
#include "tool.h"
#include "tool_FEM.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include <device_launch_parameters.h>


void vtk_writer_write(const particle_gpu* particles, int step);
void vtk_writer_write(const tool* tool, int step);
void vtk_writer_write_FEM_tool(const tool_FEM* tool_FEM, int step);
void vtk_writer_write_FEM_segments(const tool_FEM* tool_FEM, int step);
void vtk_writer_write_tool_org(const tool_FEM* tool_FEM, int step, float2_t tl_point_ref);
void vtk_writer_write_contact_variables(const tool_FEM* tool_FEM, int step);


#endif /* VTK_WRITER_H_ */
