from m5.params import *
from m5.proxy import *
from m5.objects.NDP import NDP
from m5.util.fdthelper import FdtNode, FdtPropertyWords, FdtPropertyStrings

class GemminiDevA(NDP):
	type = 'GemminiDevA'
	cxx_header = "gemmini_dev_a/gemmini_dev_a.hh"
	cxx_class = 'gem5::GemminiDevA'

	def generateDeviceTree(self, state):
		ctrl = self.ndp_ctrl
		node = FdtNode(f"gemmini@{int(ctrl.start):x}")
		node.appendCompatible("gem5,gemmini-dev-a")
		node.append(
			FdtPropertyWords(
				"reg",
				state.addrCells(ctrl.start) + state.sizeCells(ctrl.size()),
			)
		)
		node.append(FdtPropertyStrings("status", ["okay"]))
		yield node
