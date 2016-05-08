//===- GVNHoist.cpp - Hoist scalar and load expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include <functional>
#include <unordered_map>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "gvn-sched"

namespace {

struct SortByDFSIn {
private:
  DenseMap<const BasicBlock *, unsigned> &DFSNumber;

public:
  SortByDFSIn(DenseMap<const BasicBlock *, unsigned> &D) : DFSNumber(D) {}

  bool operator()(const Instruction *A, const Instruction *B) const {
    assert(A != B);
    const BasicBlock *BA = A->getParent();
    const BasicBlock *BB = B->getParent();
    unsigned NA = DFSNumber[BA];
    unsigned NB = DFSNumber[BB];
    if (NA < NB)
      return true;
    if (NA == NB) {
      // Sort them in the order they occur in the same basic block.
      BasicBlock::const_iterator AI(A), BI(B);
      return std::distance(AI, BI) < 0;
    }
    return false;
  }
};

// A multimap from a VN (value number) to all the instructions with that VN.
typedef std::multimap<unsigned, Instruction *> VNtoInsns;

class InsnInfo {
  VNtoInsns VNtoScalars;

public:
  void insert(Instruction *I, GVN::ValueTable &VN) {
    // Scalar instruction.
    unsigned V = VN.lookupOrAdd(I);
    VNtoScalars.insert(std::make_pair(V, I));
  }

  const VNtoInsns &getVNTable() const { return VNtoScalars; }
};

class LoadInfo {
  VNtoInsns VNtoLoads;

public:
  void insert(LoadInst *Load, GVN::ValueTable &VN) {
    if (Load->isSimple()) {
      Value *Ptr = Load->getPointerOperand();
      unsigned V = VN.lookupOrAdd(Ptr);
      VNtoLoads.insert(std::make_pair(V, Load));
    }
  }

  const VNtoInsns &getVNTable() const { return VNtoLoads; }
};

class StoreInfo {
  VNtoInsns VNtoStores;

public:
  void insert(StoreInst *Store, GVN::ValueTable &VN) {
    if (!Store->isSimple())
      return;
    // Hash the store address and the stored value.
    std::string VNS;
    Value *Ptr = Store->getPointerOperand();
    VNS += std::to_string(VN.lookupOrAdd(Ptr));
    VNS += ",";
    Value *Val = Store->getValueOperand();
    VNS += std::to_string(VN.lookupOrAdd(Val));
    VNtoStores.insert(std::make_pair(std::hash<std::string>()(VNS), Store));
  }

  const VNtoInsns &getVNTable() const { return VNtoStores; }
};

class CallInfo {
  VNtoInsns VNtoCallsScalars;
  VNtoInsns VNtoCallsLoads;
  VNtoInsns VNtoCallsStores;

public:
  void insert(CallInst *Call, GVN::ValueTable &VN) {
    if (Call->doesNotReturn() || !Call->doesNotThrow())
      return;

    // A call that doesNotAccessMemory is handled as a Scalar,
    // onlyReadsMemory will be handled as a Load instruction,
    // all other calls will be handled as stores.
    unsigned V = VN.lookupOrAdd(Call);

    if (Call->doesNotAccessMemory())
      VNtoCallsScalars.insert(std::make_pair(V, Call));
    else if (Call->onlyReadsMemory())
      VNtoCallsLoads.insert(std::make_pair(V, Call));
    else
      VNtoCallsStores.insert(std::make_pair(V, Call));
  }

  const VNtoInsns &getScalarVNTable() const { return VNtoCallsScalars; }

  const VNtoInsns &getLoadVNTable() const { return VNtoCallsLoads; }

  const VNtoInsns &getStoreVNTable() const { return VNtoCallsStores; }
};

typedef DenseMap<const BasicBlock *, bool> BBSideEffectsSet;
typedef SmallVector<Instruction *, 4> SmallVecInsn;
typedef SmallVectorImpl<Instruction *> SmallVecImplInsn;

// This pass hoists common computations across branches sharing common
// dominator. The primary goal is to reduce the code size, and in some
// cases reduce critical path (by exposing more ILP).
class GVNHoistLegacyPassImpl {
public:
  GVN::ValueTable VN;
  DominatorTree *DT;
  AliasAnalysis *AA;
  MemoryDependenceResults *MD;
  DenseMap<const BasicBlock *, unsigned> DFSNumber;
  BBSideEffectsSet BBSideEffects;
  MemorySSA *MSSA;
  MemorySSAWalker *MSSAW;
  enum InsKind { Unknown, Scalar, Load, Store };

  GVNHoistLegacyPassImpl(DominatorTree *Dt, AliasAnalysis *Aa,
                         MemoryDependenceResults *Md)
      : DT(Dt), AA(Aa), MD(Md), MSSAW(nullptr) {}

  // Return true when there are exception handling in BB.
  bool hasEH(const BasicBlock *BB) {
    auto It = BBSideEffects.find(BB);
    if (It != BBSideEffects.end())
      return It->second;

    if (BB->isEHPad() || BB->hasAddressTaken()) {
      BBSideEffects[BB] = true;
      return true;
    }

    if (BB->getTerminator()->mayThrow() || !BB->getTerminator()->mayReturn()) {
      BBSideEffects[BB] = true;
      return true;
    }

    BBSideEffects[BB] = false;
    return false;
  }

  // Return true when there are exception handling blocks on the execution path.
  bool hasEH(SmallPtrSetImpl<const BasicBlock *> &Paths) {
    for (const BasicBlock *BB : Paths)
      if (hasEH(BB))
        return true;

    return false;
  }

  // Return true when all paths from A to the end of the function pass through
  // either B or C.
  bool hoistingFromAllPaths(const BasicBlock *A, const BasicBlock *B,
                            const BasicBlock *C) {
    // We fully copy the WL in order to be able to remove items from it.
    SmallPtrSet<const BasicBlock *, 2> WL;
    WL.insert(B);
    WL.insert(C);

    for (auto It = df_begin(A), E = df_end(A); It != E;) {
      // There exists a path from A to the exit of the function if we are still
      // iterating in DF traversal and we removed all instructions from the work
      // list.
      if (WL.empty())
        return false;

      const BasicBlock *BB = *It;
      if (WL.erase(BB)) {
        // Stop DFS traversal when BB is in the work list.
        It.skipChildren();
        continue;
      }

      // Check for end of function, calls that do not return, etc.
      if (!isGuaranteedToTransferExecutionToSuccessor(BB->getTerminator()))
        return false;

      // Increment DFS traversal when not skipping children.
      ++It;
    }

    return true;
  }

  // Each element of a hoisting list contains the basic block where to hoist and
  // a list of instructions to be hoisted.
  typedef std::pair<BasicBlock *, SmallVecInsn> HoistingPointInfo;
  typedef SmallVector<HoistingPointInfo, 4> HoistingPointList;

  // Initialize Paths with all the basic blocks executed in between A and B.
  void gatherAllBlocks(SmallPtrSetImpl<const BasicBlock *> &Paths,
                       const BasicBlock *A, const BasicBlock *B) {
    assert(DT->dominates(A, B) && "Invalid path");

    // We may need to keep B in the Paths set if we have already added it
    // to Paths for another expression.
    bool Keep = Paths.count(B);

    // Record in Paths all basic blocks reachable in depth-first iteration on
    // the inverse CFG from B to A. These blocks are all the blocks that may be
    // executed between the execution of A and B. Hoisting an expression from B
    // into A has to be safe on all execution paths.
    for (auto I = idf_ext_begin(B, Paths), E = idf_ext_end(B, Paths); I != E;) {
      if (*I == A)
        // Stop traversal when reaching A.
        I.skipChildren();
      else
        ++I;
    }

    // Safety check for B will be handled separately.
    if (!Keep)
      Paths.erase(B);

    // Safety check for A will be handled separately.
    Paths.erase(A);
  }

  // Return true when there are users of A in one of the BBs of Paths.
  bool hasMemoryUseOnPaths(MemoryAccess *A,
                           SmallPtrSetImpl<const BasicBlock *> &Paths) {
    Value::user_iterator UI = A->user_begin();
    Value::user_iterator UE = A->user_end();
    const BasicBlock *BBA = A->getBlock();
    for (; UI != UE; ++UI)
      if (MemoryAccess *UM = dyn_cast<MemoryAccess>(*UI))
        for (const BasicBlock *PBB : Paths) {
          if (PBB == BBA) {
            if (MSSA->locallyDominates(UM, A))
              return true;
            continue;
          }
          if (PBB == UM->getBlock())
            return true;
        }
    return false;
  }

  // Return true when all operands of I are available at insertion point
  // HoistPt. When limiting the number of hoisted expressions, one could hoist
  // a load without hoisting its access function. So before hoisting any
  // expression, make sure that all its operands are available at insert point.
  bool allOperandsAvailable(const Instruction *I,
                            const BasicBlock *HoistPt) const {
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      const Value *Op = I->getOperand(i);
      const Instruction *Inst = dyn_cast<Instruction>(Op);
      if (Inst && !DT->dominates(Inst->getParent(), HoistPt))
        return false;
    }

    return true;
  }

  Instruction *firstOfTwo(Instruction *I, Instruction *J) const {
    for (Instruction &I1 : *I->getParent())
      if (&I1 == I || &I1 == J)
        return &I1;
    llvm_unreachable("Both I and J must be from same BB");
  }

  // Replace the use of From with To in Insn.
  void replaceUseWith(Instruction *Insn, Value *From, Value *To) const {
    for (Value::use_iterator UI = From->use_begin(), UE = From->use_end();
         UI != UE;) {
      Use &U = *UI++;
      if (U.getUser() == Insn) {
        U.set(To);
        return;
      }
    }
    llvm_unreachable("should replace exactly once");
  }

  bool makeOperandsAvailable(Instruction *Repl, BasicBlock *HoistPt) const {
    // Check whether the GEP of a ld/st can be synthesized at HoistPt.
    Instruction *Gep = nullptr;
    Instruction *Val = nullptr;
    if (LoadInst *Ld = dyn_cast<LoadInst>(Repl))
      Gep = dyn_cast<Instruction>(Ld->getPointerOperand());
    if (StoreInst *St = dyn_cast<StoreInst>(Repl)) {
      Gep = dyn_cast<Instruction>(St->getPointerOperand());
      Val = dyn_cast<Instruction>(St->getValueOperand());
    }

    if (!Gep || !isa<GetElementPtrInst>(Gep))
      return false;

    // Check whether we can compute the Gep at HoistPt.
    if (!allOperandsAvailable(Gep, HoistPt))
      return false;

    // Also check that the stored value is available.
    if (Val && !allOperandsAvailable(Val, HoistPt))
      return false;

    // Copy the gep before moving the ld/st.
    Instruction *ClonedGep = Gep->clone();
    ClonedGep->insertBefore(HoistPt->getTerminator());
    replaceUseWith(Repl, Gep, ClonedGep);

    // Also copy Val when it is a gep: geps are not hoisted by default.
    if (Val && isa<GetElementPtrInst>(Val)) {
      Instruction *ClonedVal = Val->clone();
      ClonedVal->insertBefore(HoistPt->getTerminator());
      replaceUseWith(Repl, Val, ClonedVal);
    }

    return true;
  }

  unsigned height(const BasicBlock *BB) {
    return 0;
  }

  // We only consider the Distance between the basic blocks because we rely
  // on a local scheduler to place the instruction in the right position.
  unsigned distance(const Instruction *Def, const Instruction *Use) {
    const BasicBlock *DefBB = Def->getParent();
    const BasicBlock *UseBB = Use->getParent();
    return height(UseBB) - height(DefBB);
  }

  unsigned distance(const User *Def, const User *Use) {
    const Instruction *DefI = dyn_cast<Instruction>(Def);
    const Instruction *UseI = dyn_cast<Instruction>(Use);
    assert (UseI && "User must be an instruction");
    if (DefI)
      return distance(DefI, UseI);
    // FIXME: Is returning 0 a good idea? Maybe it should be some invalid value.
    return 0;
  }

  // Profitable when hoisting I reduces the live-range.
  bool profitableToHoist(const Instruction *I) {
    // Distance of each use-operand to their definition max(d1, d2)
    unsigned DistUse = 0;
    std::for_each(I->op_begin(), I->op_end(), [&DistUse, I, this](const Use& U) {
        DistUse = std::max(distance(U.getUser(), I), DistUse); });

    // Distance of def to its first use.
    unsigned DistDef = 0;
    if (I->hasNUsesOrMore(1))
      DistDef = distance(I->use_begin()->getUser(), I);
    return DistUse > DistDef;
  }

  bool profitableToSink(const Instruction *I) {
    return false;
  }

  bool profitableToRematerialize(const Instruction *I) {
    return false;
  }

  // Returns true when I was hoisted.
  bool hoist(Instruction *I) {
    return false;
  }

  // Returns true when I was sunk.
  bool sink(Instruction *I) {
    return false;
  }

  // Returns true when I was rematerialized.
  bool rematerialize(Instruction *I) {
    return false;
  }


  // Hoist all expressions. Returns Number of scalars hoisted
  // and number of non-scalars hoisted.
  bool hoistExpressions(Function &F) {
    for (BasicBlock *BB : depth_first(&F.getEntryBlock())) {
      for (Instruction &I : *BB) {
        if (profitableToHoist(&I))
          hoist(&I);
        else if (profitableToSink(&I))
          sink(&I);
        else if (profitableToRematerialize(&I))
          rematerialize(&I);
      }
    }
    return false;
  }

  bool run(Function &F) {
    VN.setDomTree(DT);
    VN.setAliasAnalysis(AA);
    VN.setMemDep(MD);
    bool Res = false;

    unsigned I = 0;
    for (const BasicBlock *BB : depth_first(&F.getEntryBlock()))
      DFSNumber.insert(std::make_pair(BB, ++I));

    // FIXME: use lazy evaluation of VN to avoid the fix-point computation.
    while (1) {
      MemorySSA M(F);
      MSSA = &M;
      MSSAW = MSSA->buildMemorySSA(AA, DT);

      hoistExpressions(F);
      delete MSSAW;
      return Res;
    }

    return Res;
  }
};

class GVNHoistLegacyPass : public FunctionPass {
public:
  static char ID;

  GVNHoistLegacyPass() : FunctionPass(ID) {
    initializeGVNHoistLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &MD = getAnalysis<MemoryDependenceWrapperPass>().getMemDep();

    GVNHoistLegacyPassImpl G(&DT, &AA, &MD);
    return G.run(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MemoryDependenceWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
} // namespace

PreservedAnalyses GVNHoistPass::run(Function &F,
                                    AnalysisManager<Function> &AM) {
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  MemoryDependenceResults &MD = AM.getResult<MemoryDependenceAnalysis>(F);

  GVNHoistLegacyPassImpl G(&DT, &AA, &MD);
  if (!G.run(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

char GVNHoistLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(GVNHoistLegacyPass, "gvn-hoist",
                      "Early GVN Hoisting of Expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(GVNHoistLegacyPass, "gvn-hoist",
                    "Early GVN Hoisting of Expressions", false, false)

FunctionPass *llvm::createGVNHoistPass() { return new GVNHoistLegacyPass(); }
