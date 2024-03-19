// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// simple function to write forces on tool to disk

#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <thrust/device_vector.h>
#include "particle_gpu.h"

float2_t report_force(particle_gpu *particles);

#endif /* OUTPUT_H_ */
