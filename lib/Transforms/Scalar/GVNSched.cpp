//===- GVNHoist.cpp - Hoist scalar and load expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass hoists expressions from branches to a common dominator. It uses
// GVN (global value numbering) to discover expressions computing the same
// values. The primary goal is to reduce the code size, and in some
// cases reduce critical path (by exposing more ILP).
// Hoisting may affect the performance in some cases. To mitigate that, hoisting
// is disabled in the following cases.
// 1. Scalars across calls.
// 2. geps when corresponding load/store cannot be hoisted.
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

#define DEBUG_TYPE "gvn-hoist"

STATISTIC(NumHoisted, "Number of instructions hoisted");
STATISTIC(NumRemoved, "Number of instructions removed");
STATISTIC(NumLoadsHoisted, "Number of loads hoisted");
STATISTIC(NumLoadsRemoved, "Number of loads removed");
STATISTIC(NumStoresHoisted, "Number of stores hoisted");
STATISTIC(NumStoresRemoved, "Number of stores removed");
STATISTIC(NumCallsHoisted, "Number of calls hoisted");
STATISTIC(NumCallsRemoved, "Number of calls removed");

static cl::opt<int>
    MaxHoistedThreshold("gvn-max-hoisted", cl::Hidden, cl::init(-1),
                        cl::desc("Max number of instructions to hoist "
                                 "(default unlimited = -1)"));
static cl::opt<int> MaxNumberOfBBSInPath(
    "gvn-hoist-max-bbs", cl::Hidden, cl::init(4),
    cl::desc("Max number of basic blocks on the path between "
             "hoisting locations (default = 4, unlimited = -1)"));

static int HoistedCtr = 0;

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

  // Return true when it is safe to hoist an instruction Insn to NewHoistPt and
  // move the insertion point from HoistPt to NewHoistPt.
  bool safeToHoist(const BasicBlock *NewHoistPt, const BasicBlock *HoistPt,
                   const Instruction *Insn, const Instruction *First, InsKind K,
                   int &BBsOnAllPaths) {
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
        !hoistingFromAllPaths(NewHoistPt, HoistPt, BBInsn))
      return false;

    // Check for unsafe hoistings due to side effects.
    SmallPtrSet<const BasicBlock *, 4> Paths;
    gatherAllBlocks(Paths, NewHoistPt, HoistPt);
    gatherAllBlocks(Paths, NewHoistPt, BBInsn);

    // Check whether there are too many blocks on the hoisting path.
    BBsOnAllPaths += Paths.size();
    if (MaxNumberOfBBSInPath != -1 && BBsOnAllPaths >= MaxNumberOfBBSInPath)
      return false;

    if (hasEH(Paths))
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

  // Partition InstructionsToHoist into a set of candidates which can share a
  // common hoisting point. The partitions are collected in HPL. IsScalar is
  // true when the instructions in InstructionsToHoist are scalars. IsLoad is
  // true when the InstructionsToHoist are loads, false when they are stores.
  void partitionCandidates(SmallVecImplInsn &InstructionsToHoist,
                           HoistingPointList &HPL, InsKind K) {
    // No need to sort for two instructions.
    if (InstructionsToHoist.size() > 2) {
      SortByDFSIn Pred(DFSNumber);
      std::sort(InstructionsToHoist.begin(), InstructionsToHoist.end(), Pred);
    }

    // Create a work list of all the BB of the Insns to be hoisted.
    SmallPtrSet<BasicBlock *, 4> WL;
    SmallVecImplInsn::iterator II = InstructionsToHoist.begin();
    SmallVecImplInsn::iterator Start = II;
    BasicBlock *HoistPt = (*II)->getParent();
    WL.insert((*II)->getParent());
    int BBsOnAllPaths = 0;

    for (++II; II != InstructionsToHoist.end(); ++II) {
      Instruction *Insn = *II;
      BasicBlock *BB = Insn->getParent();
      BasicBlock *NewHoistPt = DT->findNearestCommonDominator(HoistPt, BB);
      WL.insert(BB);
      if (safeToHoist(NewHoistPt, HoistPt, Insn, *Start, K, BBsOnAllPaths)) {
        // Extend HoistPt to NewHoistPt.
        HoistPt = NewHoistPt;
        continue;
      }
      // Not safe to hoist: save the previous work list and start over from BB.
      if (std::distance(Start, II) > 1)
        HPL.push_back(std::make_pair(HoistPt, SmallVecInsn(Start, II)));
      else
        WL.clear();

      // We start over to compute HoistPt from BB.
      Start = II;
      HoistPt = BB;
      BBsOnAllPaths = 0;
    }

    // Save the last partition.
    if (std::distance(Start, II) > 1)
      HPL.push_back(std::make_pair(HoistPt, SmallVecInsn(Start, II)));
  }

  // Initialize HPL from Map.
  void computeInsertionPoints(const VNtoInsns &Map, HoistingPointList &HPL,
                              InsKind K) {
    for (VNtoInsns::const_iterator It = Map.begin(); It != Map.end();
         It = Map.upper_bound(It->first)) {
      if (MaxHoistedThreshold != -1 && ++HoistedCtr > MaxHoistedThreshold)
        return;

      unsigned V = It->first;
      if (Map.count(V) < 2)
        continue;

      // Compute the insertion point and the list of expressions to be hoisted.
      auto R = Map.equal_range(V);
      auto First = R.first;
      auto Last = R.second;
      SmallVecInsn InstructionsToHoist;
      for (; First != Last; ++First) {
        Instruction *I = First->second;
        BasicBlock *BB = I->getParent();
        if (!hasEH(BB))
          InstructionsToHoist.push_back(I);
      }

      if (InstructionsToHoist.size())
        partitionCandidates(InstructionsToHoist, HPL, K);
    }
  }

  // Return true when all operands of Instr are available at insertion point
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

  std::pair<unsigned, unsigned> hoist(HoistingPointList &HPL) {
    unsigned NI = 0, NL = 0, NS = 0, NC = 0, NR = 0;
    for (const HoistingPointInfo &HP : HPL) {
      // Find out whether we already have one of the instructions in HoistPt,
      // in which case we do not have to move it.
      BasicBlock *HoistPt = HP.first;
      const SmallVecInsn &InstructionsToHoist = HP.second;
      Instruction *Repl = nullptr;
      for (Instruction *I : InstructionsToHoist)
        if (I->getParent() == HoistPt) {
          // If there are two instructions in HoistPt to be hoisted in place:
          // update Repl to be the first one, such that we can rename the uses
          // of the second based on the first.
          Repl = !Repl ? I : firstOfTwo(Repl, I);
        }

      if (Repl) {
        // Repl is already in HoistPt: it remains in place.
        assert(allOperandsAvailable(Repl, HoistPt) &&
               "instruction depends on operands that are not available");
      } else {
        // When we do not find Repl in HoistPt, select the first in the list
        // and move it to HoistPt.
        Repl = InstructionsToHoist.front();

        // We can move Repl in HoistPt only when all operands are available.
        // The order in which hoistings are done may influence the availability
        // of operands.
        if (!allOperandsAvailable(Repl, HoistPt) &&
            !makeOperandsAvailable(Repl, HoistPt))
          continue;
        Repl->moveBefore(HoistPt->getTerminator());
      }

      if (isa<LoadInst>(Repl))
        ++NL;
      else if (isa<StoreInst>(Repl))
        ++NS;
      else if (isa<CallInst>(Repl))
        ++NC;
      else // Scalar
        ++NI;

      // Remove and rename all other instructions.
      for (Instruction *I : InstructionsToHoist)
        if (I != Repl) {
          ++NR;
          if (isa<LoadInst>(Repl))
            ++NumLoadsRemoved;
          else if (isa<StoreInst>(Repl))
            ++NumStoresRemoved;
          else if (isa<CallInst>(Repl))
            ++NumCallsRemoved;
          I->replaceAllUsesWith(Repl);
          I->eraseFromParent();
        }
    }

    NumHoisted += NL + NS + NC + NI;
    NumRemoved += NR;
    NumLoadsHoisted += NL;
    NumStoresHoisted += NS;
    NumCallsHoisted += NC;
    return {NI, NL + NC + NS};
  }

  // Hoist all expressions. Returns Number of scalars hoisted
  // and number of non-scalars hoisted.
  std::pair<unsigned, unsigned> hoistExpressions(Function &F) {
    InsnInfo II;
    LoadInfo LI;
    StoreInfo SI;
    CallInfo CI;
    for (BasicBlock *BB : depth_first(&F.getEntryBlock())) {
      bool CallWithSideEffect = false;
      for (Instruction &I1 : *BB) {
        if (LoadInst *Load = dyn_cast<LoadInst>(&I1))
          LI.insert(Load, VN);
        else if (StoreInst *Store = dyn_cast<StoreInst>(&I1))
          SI.insert(Store, VN);
        else if (CallInst *Call = dyn_cast<CallInst>(&I1)) {
          if (isa<DbgInfoIntrinsic>(*Call))
            continue;
          if (!Call->onlyReadsMemory())
            CallWithSideEffect = true;
          CI.insert(Call, VN);
        } else if (!CallWithSideEffect && !isa<GetElementPtrInst>(&I1))
          // Do not hoist scalars past calls that may write to memory because
          // that could result in spills later. geps are handled separately.
          // TODO: We can relax this for targets like AArch64 as they have more
          // registers than X86.
          II.insert(&I1, VN);
      }
    }

    HoistingPointList HPL;
    computeInsertionPoints(II.getVNTable(), HPL, InsKind::Scalar);
    computeInsertionPoints(LI.getVNTable(), HPL, InsKind::Load);
    computeInsertionPoints(SI.getVNTable(), HPL, InsKind::Store);
    computeInsertionPoints(CI.getScalarVNTable(), HPL, InsKind::Scalar);
    computeInsertionPoints(CI.getLoadVNTable(), HPL, InsKind::Load);
    computeInsertionPoints(CI.getStoreVNTable(), HPL, InsKind::Store);
    return hoist(HPL);
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

      auto HoistStat = hoistExpressions(F);
      if (HoistStat.first + HoistStat.second == 0) {
        delete MSSAW;
        return Res;
      }
      if (HoistStat.second > 0) {
        // Memory SSA is not updated by our code generator: recompute it.
        delete MSSAW;
        // To address a limitation of the current GVN, we need to rerun the
        // hoisting after we hoisted loads in order to be able to hoist all
        // scalars dependent on the hoisted loads. Same for stores.
        VN.clear();
      }
      Res = true;
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
