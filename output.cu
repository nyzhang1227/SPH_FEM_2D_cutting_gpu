// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "output.h"

// float2 + struct
struct add_float2 {
    __device__ float2_t operator()(const float2_t& a, const float2_t& b) const {
        float2_t r;
        r.x = a.x + b.x;
        r.y = a.y + b.y;
        return r;
    }
 };


float2_t report_force(particle_gpu *particles) {
	thrust::device_ptr<float2_t> t_fc(particles->fc);
	thrust::device_ptr<float2_t> t_ft(particles->ft);
	float2_t ini;
	ini.x = 0.;
	ini.y = 0.;
	ini = thrust::reduce(t_fc, t_fc + particles->N, ini, add_float2());
	return thrust::reduce(t_ft, t_ft + particles->N, ini, add_float2());
}
