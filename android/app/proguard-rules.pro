# R8 rules for the Paafekt release build.
#
# Keep file names and line numbers so crash stack traces (emailed via
# CrashReportActivity) can be retraced with this release's mapping.txt.
# Archive app/build/outputs/mapping/release/mapping.txt for every upload
# and add it to Play Console for crash/ANR deobfuscation.
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# ONNX Runtime calls back into these classes from JNI; R8 cannot see that.
-keep class ai.onnxruntime.** { *; }

# Filament (used by SceneView) binds Java objects from native code.
-keep class com.google.android.filament.** { *; }
-dontwarn com.google.android.filament.**

# WebView JS bridge: GLBRoomActivity exposes WebAppInterface as "Android".
# Methods invoked from JavaScript must keep their names.
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}
