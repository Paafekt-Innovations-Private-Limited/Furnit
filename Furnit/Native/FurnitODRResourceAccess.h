#import <Foundation/Foundation.h>

@class NSBundleResourceRequest;

NS_ASSUME_NONNULL_BEGIN

/// Wraps ``-[NSBundleResourceRequest beginAccessingResourcesWithCompletionHandler:]`` in `@try/@catch`.
/// Apple may raise ``NSException`` when the streaming-unzip daemon dies (often surfaced as Cocoa 4099);
/// Swift-only `catch` does not trap that and the process terminates.
@interface FurnitODRResourceAccess : NSObject

+ (void)beginAccessingResourcesWithRequest:(NSBundleResourceRequest *)request
                                completion:(void (^)(NSError *_Nullable error))completion
    NS_SWIFT_NAME(beginAccessingResources(_:completion:));

@end

NS_ASSUME_NONNULL_END
