#ifndef FurnitLiteRTDelegationAudit_h
#define FurnitLiteRTDelegationAudit_h

#include <stdbool.h>
#include <TensorFlowLiteC/TensorFlowLiteC.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FurnitLiteRTDelegationAudit FurnitLiteRTDelegationAudit;

/// Creates a no-op delegate that inspects the execution plan after the Metal delegate runs.
/// The audit delegate never replaces graph nodes and owns no inference implementation.
FurnitLiteRTDelegationAudit *FurnitLiteRTDelegationAuditCreate(
    TfLiteDelegate *metalDelegate
);

void FurnitLiteRTDelegationAuditDelete(FurnitLiteRTDelegationAudit *audit);

TfLiteDelegate *FurnitLiteRTDelegationAuditGetDelegate(
    FurnitLiteRTDelegationAudit *audit
);

bool FurnitLiteRTDelegationAuditDidPrepare(
    const FurnitLiteRTDelegationAudit *audit
);

int FurnitLiteRTDelegationAuditExecutionPlanNodeCount(
    const FurnitLiteRTDelegationAudit *audit
);

int FurnitLiteRTDelegationAuditMetalPartitionCount(
    const FurnitLiteRTDelegationAudit *audit
);

int FurnitLiteRTDelegationAuditMetalOriginalNodeCount(
    const FurnitLiteRTDelegationAudit *audit
);

int FurnitLiteRTDelegationAuditRemainingCPUNodeCount(
    const FurnitLiteRTDelegationAudit *audit
);

int FurnitLiteRTDelegationAuditOtherDelegateNodeCount(
    const FurnitLiteRTDelegationAudit *audit
);

#ifdef __cplusplus
}
#endif

#endif /* FurnitLiteRTDelegationAudit_h */
