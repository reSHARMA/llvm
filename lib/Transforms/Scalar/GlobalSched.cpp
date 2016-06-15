//===- GlobalSched.cpp - Global Scheduling of expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

/*
One of the models could be to hoist to nearby blocks, by having a threshold of
how far we can hoist.


Other could be to hoist in those cases which reduces the effective live-range.
How to compute effective live-range before/after hoisting is illustrated below.
The following idea not only relates to code-hoisting but also to global code
motion (gcm).

 Global code motion. Example:
 bb1: b = ...
 bb2: c = ...
 bb3: a = b + c (to be moved to b5)
 bb4: ... = a
 old-live-range = distance(bb1,bb3) + distance(bb2,bb3) + distance (bb3,bb4)
 new-live-rance = distance(bb1,bb5) + distance(bb2,bb5) + distance (bb5,bb4)
 distance(bbx, bby) = total instruction count in the path from bbx to bby
 
 If the new live-range is less than the old one it will be a good candidate
 for gcm. When both the ranges will be same:
  - It is a simple copy of kind a = b.
  - One of the operands is not a Instruction/Register.
  - In these cases we need to have some heuristic or we can ignore them.
 
 The decision for bb5 can be made by whether hoist/sink is beneficial.
 Loads can be moved early as long as there is load available in each branch
 or there is already a load/store from/to the same underlying object.
 Stores can be moved up if there is a store/load to/from the same object.
 The idea is to establish that the object has memory allocated to it.
 Moving load/store together may help with locality of reference.
 Hoist redundant instructions which are:
  - Already available in the dominator.

  - Are in one of the sibling branches, i.e., the instruction is used at a point
    which shares a common dominator where all the use-operands are
    available. The code motion will hoist the partially-redundant computation in
    the common dominator instead of copying to the other sibling (which is what
    regular PRE does).  After this, the actual PRE will remove the redundant
    computation.

 In some cases it is possible to generate redundancy by restructuring the code
 (Ref. Ras Bodik), but that I'll leave for the next iteration of this patch.

Hoisting also reduces critical path length of execution in out of order machines
(but not in sequential machines), by exposing ILP before the conditional where
the instruction was hoisted.  This feature has already been identified in the
previous patch (http://reviews.llvm.org/D19338).
 
Concerns: Safety of load/store instructions to be checked.  We cannot hoist
loads until all the paths in the parent BBs have the same load/store.  This is
to comply with C semantics.

It seems, code hoisting is beneficial in cases even if the computations are not
redundant.  Say c = f(a, b) is an instruction. Hoisting up will reduce the
liveness of registers a and b, but will only increase the liveness of c. So we
gain 2:1 even when the computation is not redundant.

For loads this is not the case because, load takes only one operand so the
liveness remains the same, additionally hoisting too much loads can have
adversely affect the cache behavior.  On the other hand sinking loads may
improve the cache behavior, because we load as late as possible.  But sinking
computations may increase the live range.

TODO: We might use the concept of pinned instructions from Click's paper for
faster convergence.
*/

#include "llvm/ADT/DepthFirstIterator.h"
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

#define DEBUG_TYPE "global-sched"

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
class GlobalSchedLegacyPassImpl {
public:
  GVN::ValueTable VN;
  DominatorTree *DT;
  AliasAnalysis *AA;
  MemoryDependenceResults *MD;
  DenseMap<const BasicBlock *, unsigned> DFSNumber;
  BBSideEffectsSet BBSideEffects;
  MemorySSA *MSSA;
  MemorySSAWalker *MSSAW;
  DenseMap<const BasicBlock *, unsigned> BBHeightMap;
  enum InsKind { Unknown, Scalar, Load, Store };

  GlobalSchedLegacyPassImpl(DominatorTree *Dt, AliasAnalysis *Aa,
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
  // both B and C.
  bool postDominatedByCombined(const BasicBlock *A, const BasicBlock *B,
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
  bool gatherAllBlocks(const BasicBlock *A, const BasicBlock *B,
                       SmallPtrSetImpl<const BasicBlock *> &Paths) {
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
      else {
        // Bail out early when the path has any pinned instructions/unsafe BBs.
        if (hasEH(*I))
          return false;
        ++I;
      }
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

  // TODO: This has been copied from GVNHoist just in case if it is useful.
  // We might want to delete this if we dont find this useful.
  // Return true when it is safe to hoist an instruction Insn to NewHoistPt and
  // move the insertion point from HoistPt to NewHoistPt.
  bool safeToHoist(const BasicBlock *NewHoistPt, const BasicBlock *HoistPt,
                   const Instruction *Insn, const Instruction *First, InsKind K) {
    if (hasEH(HoistPt))
      return false;

    const BasicBlock *BBInsn = Insn->getParent();
    // When HoistPt already contains an instruction to be hoisted, the
    // expression is needed on all paths.

    // Check that the hoisted expression is needed on all paths: it is unsafe
    // to hoist loads to a place where there may be a path not loading from
    // the same address: for instance there may be a branch on which the
    // address of the load may not be initialized. FIXME: at -Oz we may want
    // to hoist scalars to a place where they are partially needed.
    if (BBInsn != NewHoistPt &&
        !postDominatedByCombined(NewHoistPt, HoistPt, BBInsn))
      return false;

    // Check for unsafe hoistings due to side effects.
    SmallPtrSet<const BasicBlock *, 4> Paths;
    if (!gatherAllBlocks(NewHoistPt, HoistPt, Paths))
      return false;
    if (!gatherAllBlocks(NewHoistPt, BBInsn, Paths))
      return false;

    // Safe to hoist scalars.
    if (K == InsKind::Scalar)
      return true;

    // For loads and stores, we check for dependences on the Memory SSA.
    MemoryAccess *MemdefInsn =
        cast<MemoryUseOrDef>(MSSA->getMemoryAccess(Insn))->getDefiningAccess();
    BasicBlock *BBMemdefInsn = MemdefInsn->getBlock();

    if (DT->properlyDominates(NewHoistPt, BBMemdefInsn))
      // Cannot move Insn past BBMemdefInsn to NewHoistPt.
      return false;

    MemoryAccess *MemdefFirst =
        cast<MemoryUseOrDef>(MSSA->getMemoryAccess(First))->getDefiningAccess();
    BasicBlock *BBMemdefFirst = MemdefFirst->getBlock();

    if (DT->properlyDominates(NewHoistPt, BBMemdefFirst))
      // Cannot move First past BBMemdefFirst to NewHoistPt.
      return false;

    if (K == InsKind::Store) {
      // Check that we do not move a store past loads.
      if (DT->dominates(BBMemdefInsn, NewHoistPt))
        if (hasMemoryUseOnPaths(MemdefInsn, Paths))
          return false;

      if (DT->dominates(BBMemdefFirst, NewHoistPt))
        if (hasMemoryUseOnPaths(MemdefFirst, Paths))
          return false;
    }

    if (DT->properlyDominates(BBMemdefInsn, NewHoistPt) &&
        DT->properlyDominates(BBMemdefFirst, NewHoistPt))
      return true;

    const BasicBlock *BBFirst = First->getParent();
    if (BBInsn == BBFirst)
      return false;

    assert(BBMemdefInsn == NewHoistPt || BBMemdefFirst == NewHoistPt);

    if (BBInsn != NewHoistPt && BBFirst != NewHoistPt)
      return true;

    if (BBInsn == NewHoistPt) {
      if (DT->properlyDominates(BBMemdefFirst, NewHoistPt))
        return true;
      assert(BBInsn == BBMemdefFirst);
      if (MSSA->locallyDominates(MSSA->getMemoryAccess(Insn), MemdefFirst))
        return false;
      return true;
    }

    if (BBFirst == NewHoistPt) {
      if (DT->properlyDominates(BBMemdefInsn, NewHoistPt))
        return true;
      assert(BBFirst == BBMemdefInsn);
      if (MSSA->locallyDominates(MSSA->getMemoryAccess(First), MemdefInsn))
        return false;
      return true;
    }

    // No side effects: it is safe to hoist.
    return true;
  }

  // TODO: This has been copied from GVNHoist just in case if it is useful.
  // We might want to delete this if we dont find this useful.
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

  // Collect the minimum height/depth of each BB from the root node.
  void collectHeight(const BasicBlock *Root) {
    BBHeightMap[Root] = 0;
    // Traverse in reverse post order.
    for (auto I = idf_begin(Root); I != idf_end(Root); ++I) {
      const BasicBlock *BB = *I;
      DenseMap<const BasicBlock*, unsigned>::iterator BBCI = BBHeightMap.find(BB);
      if (BBCI == BBHeightMap.end())
        continue;
      // FIXME: Take the height + 1 from the predecessor which gives minimum height.
      for (unsigned i = 0; i < BB->getTerminator()->getNumSuccessors(); ++i)
        BBHeightMap[BB->getTerminator()->getSuccessor(i)] = BBHeightMap[BB] + 1;
    }
    // There must not be any remaining ones.
    for (auto BBI = Root->getParent()->begin(), E = Root->getParent()->end(); BBI != E; ++BBI)
      assert (BBHeightMap.count(&*BBI));
  }

  unsigned height(const BasicBlock *BB) const {
    assert (BBHeightMap.count(BB) && "BB Height has not been populated.");
    return BBHeightMap.find(BB)->second;
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
    unsigned DistUseFromDef = 0;
    std::for_each(I->op_begin(), I->op_end(), [&DistUseFromDef, I, this](const Use& U) {
        DistUseFromDef = std::max(distance(U.getUser(), I), DistUseFromDef); });

    // Distance of def to its first use.
    unsigned DistDefToUse = 0;
    if (I->hasNUsesOrMore(1))
      DistDefToUse = distance(I->use_begin()->getUser(), I);
    return DistUseFromDef > DistDefToUse;
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
  bool schedule(Function &F) {
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
      MemorySSA M(F, AA, DT);
      MSSA = &M;
      MSSAW = MSSA->getWalker();
      schedule(F);
      return Res;
    }

    return Res;
  }
};

class GlobalSchedLegacyPass : public FunctionPass {
public:
  static char ID;

  GlobalSchedLegacyPass() : FunctionPass(ID) {
    initializeGlobalSchedLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &MD = getAnalysis<MemoryDependenceWrapperPass>().getMemDep();

    GlobalSchedLegacyPassImpl G(&DT, &AA, &MD);
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

PreservedAnalyses GlobalSchedPass::run(Function &F,
                                       AnalysisManager<Function> &AM) {
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  AliasAnalysis &AA = AM.getResult<AAManager>(F);
  MemoryDependenceResults &MD = AM.getResult<MemoryDependenceAnalysis>(F);

  GlobalSchedLegacyPassImpl G(&DT, &AA, &MD);
  if (!G.run(F))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

char GlobalSchedLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(GlobalSchedLegacyPass, "global-sched",
                      "Global Scheduling of Expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(GlobalSchedLegacyPass, "global-sched",
                    "Global Scheduling of Expressions", false, false)

FunctionPass *llvm::createGlobalSchedPass() { return new GlobalSchedLegacyPass(); }
