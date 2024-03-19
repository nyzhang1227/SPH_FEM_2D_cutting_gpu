// Copyright ETH Zurich, IWF
// SPH-FEM cutting simulations on the basis of mfree_iwf-ul_cut_gpu: 
// https://github.com/iwf-inspire/mfree_iwf-ul_cut_gpu
// Date: 2024.03.10

// You should have received a copy of the GNU General Public License
// along with mfree_iwf-ul_cut_gpu.  If not, see <http://www.gnu.org/licenses/>.

#include "constants_structs.h"

phys_constants make_phys_constants() {
	phys_constants phys;
	memset(&phys, 0, sizeof(phys_constants));
	return phys;
}

trml_constants make_trml_constants() {
	trml_constants trml;
	memset(&trml, 0, sizeof(trml_constants));
	return trml;
}

corr_constants make_corr_constants() {
	corr_constants corr;
	memset(&corr, 0, sizeof(corr_constants));
	corr.wdeltap = 1.;
	return corr;
}

joco_constants make_joco_constants() {
	joco_constants joco;
	memset(&joco, 0, sizeof(joco_constants));
	return joco;
}

geom_constants make_geom_constants() {
	geom_constants geom;
	memset(&geom, 0, sizeof(geom_constants));
	return geom;
}

tool_constants make_tool_constants() {
	tool_constants tc;
	memset(&tc, 0, sizeof(tool_constants));
	return tc;
}

wear_constants make_wear_constants()
{
	wear_constants wear;
	memset(&wear, 0, sizeof(wear_constants));
	return wear;
}