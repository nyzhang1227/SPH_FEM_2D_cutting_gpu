// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// geometrical primitives on the GPU to establish contact with the tool

#ifndef TOOL_CUH_
#define TOOL_CUH_

#include "types.h"
#include "tool.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct line_gpu {
	float_t a;
	float_t b;
	bool vertical;
};

struct segment_gpu {
	float2_t left;
	float2_t right;
	line_gpu l;
	float2_t n;
};

struct circle_segment_gpu {
	float_t  r;
	float_t  t1;
	float_t  t2;
	float2_t   p;
};

#define TOOL_MAX_SEG 5

#endif /* TOOL_CUH_ */
