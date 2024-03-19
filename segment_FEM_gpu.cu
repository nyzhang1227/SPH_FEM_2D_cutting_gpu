// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "segment_FEM_gpu.h"

segment_FEM_gpu::segment_FEM_gpu()
{
}

segment_FEM_gpu::segment_FEM_gpu(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, int N)
{
	segment_num = N;
	set_segments_memory(left_, right_, n_, heat_exchange_, marks_, fric_heat_, N);
}

segment_FEM_gpu::segment_FEM_gpu(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, float2_t* wear_rate_, float_t* wear_nodes_, float4_t* physical_para_, float_t* sliding_force_, int N)
{
	segment_num = N;
	set_segments_memory(left_, right_, n_, heat_exchange_, marks_, fric_heat_, wear_rate_, wear_nodes_, physical_para_, sliding_force_, N);
}

void segment_FEM_gpu::set_segments_memory(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, int N)
{
	cudaMalloc((void**)&left, sizeof(float2_t) * N);
	cudaMalloc((void**)&right, sizeof(float2_t) * N);
	cudaMalloc((void**)&n, sizeof(float2_t) * N);
	cudaMalloc((void**)&heat_exchange, sizeof(float2_t) * N);
	cudaMalloc((void**)&marks, sizeof(int4) * N);
	cudaMalloc((void**)&fric_heat, sizeof(float_t) * N);


	cudaMemcpy(left, left_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(right, right_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(n, n_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(heat_exchange, heat_exchange_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(marks, marks_, sizeof(int4) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(fric_heat, fric_heat_, sizeof(float_t) * N, cudaMemcpyHostToDevice);
}

void segment_FEM_gpu::set_segments_memory(float2_t* left_, float2_t* right_, float2_t* n_, float2_t* heat_exchange_, int4* marks_, float_t* fric_heat_, float2_t* wear_rate_, float_t* wear_nodes_, float4_t* physical_para_, float_t* sliding_force_, int N)
{
	cudaMalloc((void**)&left, sizeof(float2_t) * N);
	cudaMalloc((void**)&right, sizeof(float2_t) * N);
	cudaMalloc((void**)&n, sizeof(float2_t) * N);
	cudaMalloc((void**)&heat_exchange, sizeof(float2_t) * N);
	cudaMalloc((void**)&marks, sizeof(int4) * N);
	cudaMalloc((void**)&fric_heat, sizeof(float_t) * N);
#ifdef WEAR_NODE_SHIFT
	cudaMalloc((void**)&wear_rate, sizeof(float2_t) * N);
	cudaMalloc((void**)&wear_nodes, sizeof(float_t) * N);
	cudaMalloc((void**)&physical_para, sizeof(float4_t) * N);
	cudaMalloc((void**)&sliding_force, sizeof(float_t) * N);
#endif



	cudaMemcpy(left, left_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(right, right_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(n, n_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(heat_exchange, heat_exchange_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(marks, marks_, sizeof(int4) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(fric_heat, fric_heat_, sizeof(float_t) * N, cudaMemcpyHostToDevice);
#ifdef WEAR_NODE_SHIFT
	cudaMemcpy(wear_rate, wear_rate_, sizeof(float2_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(wear_nodes, wear_nodes_, sizeof(float_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(physical_para, physical_para_, sizeof(float4_t) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(sliding_force, sliding_force_, sizeof(float_t) * N, cudaMemcpyHostToDevice);
#endif	
}

