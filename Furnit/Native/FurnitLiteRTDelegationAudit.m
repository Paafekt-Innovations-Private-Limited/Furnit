#import "FurnitLiteRTDelegationAudit.h"

#include <stdlib.h>

struct FurnitLiteRTDelegationAudit {
    TfLiteDelegate delegate;
    TfLiteDelegate *metalDelegate;
    bool didPrepare;
    int executionPlanNodeCount;
    int metalPartitionCount;
    int metalOriginalNodeCount;
    int remainingCPUNodeCount;
    int otherDelegateNodeCount;
};

static TfLiteStatus FurnitLiteRTDelegationAuditPrepare(
    TfLiteContext *context,
    TfLiteDelegate *delegate
) {
    if (context == NULL || delegate == NULL || delegate->data_ == NULL) {
        return kTfLiteError;
    }

    FurnitLiteRTDelegationAudit *audit = delegate->data_;
    audit->didPrepare = false;
    audit->executionPlanNodeCount = 0;
    audit->metalPartitionCount = 0;
    audit->metalOriginalNodeCount = 0;
    audit->remainingCPUNodeCount = 0;
    audit->otherDelegateNodeCount = 0;

    TfLiteIntArray *executionPlan = NULL;
    if (context->GetExecutionPlan == NULL ||
        context->GetExecutionPlan(context, &executionPlan) != kTfLiteOk ||
        executionPlan == NULL) {
        return kTfLiteError;
    }

    audit->executionPlanNodeCount = executionPlan->size;
    for (int index = 0; index < executionPlan->size; index += 1) {
        TfLiteNode *node = NULL;
        TfLiteRegistration *registration = NULL;
        if (context->GetNodeAndRegistration == NULL ||
            context->GetNodeAndRegistration(
                context,
                executionPlan->data[index],
                &node,
                &registration
            ) != kTfLiteOk ||
            node == NULL ||
            registration == NULL) {
            return kTfLiteError;
        }

        if (node->delegate == audit->metalDelegate) {
            audit->metalPartitionCount += 1;
            const TfLiteDelegateParams *parameters = node->builtin_data;
            if (parameters != NULL &&
                parameters->delegate == audit->metalDelegate &&
                parameters->nodes_to_replace != NULL) {
                audit->metalOriginalNodeCount += parameters->nodes_to_replace->size;
            }
        } else if (node->delegate == NULL) {
            audit->remainingCPUNodeCount += 1;
        } else {
            audit->otherDelegateNodeCount += 1;
        }
    }

    audit->didPrepare = true;
    return kTfLiteOk;
}

FurnitLiteRTDelegationAudit *FurnitLiteRTDelegationAuditCreate(
    TfLiteDelegate *metalDelegate
) {
    if (metalDelegate == NULL) {
        return NULL;
    }
    FurnitLiteRTDelegationAudit *audit = calloc(
        1,
        sizeof(FurnitLiteRTDelegationAudit)
    );
    if (audit == NULL) {
        return NULL;
    }
    audit->metalDelegate = metalDelegate;
    audit->delegate = TfLiteDelegateCreate();
    audit->delegate.data_ = audit;
    audit->delegate.Prepare = FurnitLiteRTDelegationAuditPrepare;
    return audit;
}

void FurnitLiteRTDelegationAuditDelete(FurnitLiteRTDelegationAudit *audit) {
    free(audit);
}

TfLiteDelegate *FurnitLiteRTDelegationAuditGetDelegate(
    FurnitLiteRTDelegationAudit *audit
) {
    return audit == NULL ? NULL : &audit->delegate;
}

bool FurnitLiteRTDelegationAuditDidPrepare(
    const FurnitLiteRTDelegationAudit *audit
) {
    return audit != NULL && audit->didPrepare;
}

int FurnitLiteRTDelegationAuditExecutionPlanNodeCount(
    const FurnitLiteRTDelegationAudit *audit
) {
    return audit == NULL ? 0 : audit->executionPlanNodeCount;
}

int FurnitLiteRTDelegationAuditMetalPartitionCount(
    const FurnitLiteRTDelegationAudit *audit
) {
    return audit == NULL ? 0 : audit->metalPartitionCount;
}

int FurnitLiteRTDelegationAuditMetalOriginalNodeCount(
    const FurnitLiteRTDelegationAudit *audit
) {
    return audit == NULL ? 0 : audit->metalOriginalNodeCount;
}

int FurnitLiteRTDelegationAuditRemainingCPUNodeCount(
    const FurnitLiteRTDelegationAudit *audit
) {
    return audit == NULL ? 0 : audit->remainingCPUNodeCount;
}

int FurnitLiteRTDelegationAuditOtherDelegateNodeCount(
    const FurnitLiteRTDelegationAudit *audit
) {
    return audit == NULL ? 0 : audit->otherDelegateNodeCount;
}
