// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// interface between SPH and FEM domains 

#ifndef SEG_FEM_GPU
#define SEG_FEM_GPU

#include "types.h"
//#include "tool_FEM.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <glm/glm.hpp>

/*
struct line_gpu_FEM {
	float_t a;
	float_t b;
	bool vertical;
};
*/

class segment_FEM_gpu {
public:
	int segment_num = 0;

	float2_t* left = 0;
	float2_t* right = 0;

	float2_t* n = 0;
	int4* marks = 0; // x represents the element number of the mesh,  y represents the color of element, z and w represent the node number in mesh
	float2_t* heat_exchange = 0;  // x for qT*length, y for length
	float_t* fric_heat = 0;
#ifdef WEAR_NODE_SHIFT
	float2_t* wear_rate;
	float_t* wear_nodes;
	float4_t* physical_para;    // contact particle number, force, relative sliding velocity, temperature
	float_t* sliding_force;
#endif

	void set_segments_memory(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, int N);
	void set_segments_memory(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, float2_t* wear_rate_, float_t* wear_nodes_, float4_t* physical_para_, float_t* sliding_force_, int N);

	segment_FEM_gpu();
	segment_FEM_gpu(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, int N);
	segment_FEM_gpu(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, float2_t* wear_rate_, float_t* wear_nodes_, float4_t* physical_para_, float_t* sliding_force_, int N);
	virtual ~segment_FEM_gpu() = default;
};


#endif /* SEG_FEM_GPU */


