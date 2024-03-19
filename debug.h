// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

// dump solution to a text file (e.g. for easy MATLAB visualization)

#ifndef DEBUG_H_
#define DEBUG_H_

#include "types.h"
#include "particle_gpu.h"
#include "tool.h"

void dump_state(particle_gpu *particles, tool *tool, int step);

#endif /* DEBUG_H_ */
