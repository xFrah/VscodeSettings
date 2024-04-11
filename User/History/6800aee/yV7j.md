Remember to kill the assert in C:\Users\user\FlutterSDK\flutter\packages\flutter\lib\src\animation\animation_controller.dart at line 822.

E/flutter ( 8185): [ERROR:flutter/runtime/dart_vm_initializer.cc(41)] Unhandled Exception: setState() called after dispose(): _WifiCredentialsScreenState#5e4fc(lifecycle state: defunct, not mounted)
E/flutter ( 8185): This error happens if you call setState() on a State object for a widget that no longer appears in the widget tree (e.g., whose parent widget no longer includes the widget in its build). This error can occur when code calls setState() from a timer or an animation callback.
E/flutter ( 8185): The preferred solution is to cancel the timer or stop listening to the animation in the dispose() callback. Another solution is to check the "mounted" property of this object before calling setState() to ensure the object is still in the tree.
E/flutter ( 8185): This error might indicate a memory leak if setState() is being called because another object is retaining a reference to this State object after it has been removed from the tree. To avoid memory leaks, consider breaking the reference to this object during dispose().
