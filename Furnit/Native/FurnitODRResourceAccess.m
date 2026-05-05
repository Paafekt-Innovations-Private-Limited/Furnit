#import "FurnitODRResourceAccess.h"

@implementation FurnitODRResourceAccess

+ (void)beginAccessingResourcesWithRequest:(NSBundleResourceRequest *)request
                                completion:(void (^)(NSError *_Nullable error))completion {
    void (^finish)(NSError *_Nullable) = ^(NSError *_Nullable error) {
        if (completion != nil) {
            completion(error);
        }
    };

    @try {
        [request beginAccessingResourcesWithCompletionHandler:^(NSError *_Nullable error) {
            finish(error);
        }];
    } @catch (NSException *exception) {
        NSDictionary *info = @{
            NSLocalizedDescriptionKey: (id)(exception.reason ?: @"NSBundleResourceRequest raised NSException"),
            @"FurnitNSExceptionName": (id)(exception.name ?: @""),
        };
        NSError *err = [NSError errorWithDomain:NSCocoaErrorDomain code:4099 userInfo:info];
        finish(err);
    }
}

@end
