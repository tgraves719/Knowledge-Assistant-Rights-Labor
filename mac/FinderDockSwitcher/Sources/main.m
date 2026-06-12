#import <ApplicationServices/ApplicationServices.h>
#import <Cocoa/Cocoa.h>

@interface FinderWindow : NSObject
@property(nonatomic) NSInteger windowIndex;
@property(nonatomic, copy) NSString *title;
@property(nonatomic, copy) NSString *path;
@end

@implementation FinderWindow
@end

static void LogMessage(NSString *message) {
    NSString *logPath = [@"~/Library/Logs/FinderDockSwitcher.log" stringByExpandingTildeInPath];
    NSString *line = [NSString stringWithFormat:@"%@ %@\n", [NSDate date], message];
    NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:logPath];

    if (handle == nil) {
        [line writeToFile:logPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
        return;
    }

    @try {
        [handle seekToEndOfFile];
        [handle writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
        [handle closeFile];
    } @catch (__unused NSException *exception) {
    }
}

static NSString *RunAppleScript(NSString *source, NSError **error) {
    NSDictionary *errorInfo = nil;
    NSAppleScript *script = [[NSAppleScript alloc] initWithSource:source];
    NSAppleEventDescriptor *result = [script executeAndReturnError:&errorInfo];

    if (errorInfo != nil) {
        NSString *message = errorInfo[NSAppleScriptErrorMessage] ?: @"AppleScript execution failed.";
        LogMessage([NSString stringWithFormat:@"AppleScript error: %@", message]);
        if (error != NULL) {
            *error = [NSError errorWithDomain:@"FinderDockSwitcher"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey: message}];
        }
        return nil;
    }

    LogMessage(@"AppleScript completed successfully.");
    return result.stringValue ?: @"";
}

static NSString *NormalizeWindowPath(NSString *rawPath) {
    if (rawPath.length == 0) {
        return @"";
    }

    NSURL *url = [NSURL URLWithString:rawPath];
    if (url.fileURL && url.path.length > 0) {
        return url.path;
    }

    return rawPath;
}

static NSArray<FinderWindow *> *FetchFinderWindows(NSError **error) {
    NSString *script =
        @"set AppleScript's text item delimiters to linefeed\n"
         "tell application \"System Events\"\n"
         "    if not (exists process \"Finder\") then\n"
         "        return \"\"\n"
         "    end if\n"
         "    tell process \"Finder\"\n"
         "        set windowLines to {}\n"
         "        set indexCounter to 1\n"
         "        repeat with w in windows\n"
         "            try\n"
         "                set windowName to name of w\n"
         "                set windowPath to \"\"\n"
         "                try\n"
         "                    set windowPath to value of attribute \"AXDocument\" of w\n"
         "                end try\n"
         "                copy ((indexCounter as string) & \"||\" & windowName & \"||\" & windowPath) to end of windowLines\n"
         "            on error\n"
         "                copy ((indexCounter as string) & \"||\" & \"Untitled Finder Window\" & \"||\") to end of windowLines\n"
         "            end try\n"
         "            set indexCounter to indexCounter + 1\n"
         "        end repeat\n"
         "        return windowLines as text\n"
         "    end tell\n"
         "end tell\n";

    NSString *output = RunAppleScript(script, error);
    if (output == nil) {
        return nil;
    }

    NSMutableArray<FinderWindow *> *windows = [NSMutableArray array];
    for (NSString *line in [output componentsSeparatedByString:@"\n"]) {
        if (line.length == 0) {
            continue;
        }

        NSRange separator = [line rangeOfString:@"||"];
        if (separator.location == NSNotFound) {
            continue;
        }

        NSArray<NSString *> *parts = [line componentsSeparatedByString:@"||"];
        if (parts.count < 2) {
            continue;
        }

        FinderWindow *window = [[FinderWindow alloc] init];
        window.windowIndex = parts[0].integerValue;
        window.title = parts[1];
        window.path = parts.count >= 3 ? NormalizeWindowPath(parts[2]) : @"";
        [windows addObject:window];
    }

    return windows;
}

static BOOL ActivateFinderWindow(NSInteger windowIndex, NSError **error) {
    NSString *script = [NSString stringWithFormat:
        @"tell application \"Finder\" to activate\n"
         "tell application \"System Events\"\n"
         "    if not (exists process \"Finder\") then\n"
         "        return \"missing\"\n"
         "    end if\n"
         "    tell process \"Finder\"\n"
         "        set frontmost to true\n"
         "        if (count of windows) < %ld then\n"
         "            return \"missing\"\n"
         "        end if\n"
         "        perform action \"AXRaise\" of window %ld\n"
         "        return \"ok\"\n"
         "    end tell\n"
         "end tell\n",
         (long)windowIndex,
         (long)windowIndex];

    NSString *result = RunAppleScript(script, error);
    return result != nil && [result isEqualToString:@"ok"];
}

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic, copy) NSArray<FinderWindow *> *cachedWindows;
@property(nonatomic) BOOL isRefreshing;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    (void)notification;
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    LogMessage(@"Application launched.");
    self.cachedWindows = @[];
    self.isRefreshing = NO;
    [self requestAccessibilityIfNeeded];
    [self refreshFinderWindows];
    [NSTimer scheduledTimerWithTimeInterval:5.0
                                     target:self
                                   selector:@selector(refreshFinderWindows)
                                   userInfo:nil
                                    repeats:YES];
}

- (NSMenu *)applicationDockMenu:(NSApplication *)sender {
    (void)sender;

    NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Finder Windows"];

    if (self.cachedWindows.count == 0 && self.lastError == nil && self.isRefreshing) {
        NSMenuItem *loadingItem = [[NSMenuItem alloc] initWithTitle:@"Loading Finder Windows..."
                                                             action:nil
                                                      keyEquivalent:@""];
        loadingItem.enabled = NO;
        [menu addItem:loadingItem];
    } else if (self.cachedWindows.count == 0 && self.lastError == nil) {
        NSMenuItem *emptyItem = [[NSMenuItem alloc] initWithTitle:@"No Open Finder Windows"
                                                           action:nil
                                                    keyEquivalent:@""];
        emptyItem.enabled = NO;
        [menu addItem:emptyItem];
    } else if (self.lastError != nil) {
        NSMenuItem *errorItem = [[NSMenuItem alloc] initWithTitle:@"Finder Access Failed"
                                                           action:nil
                                                    keyEquivalent:@""];
        errorItem.enabled = NO;
        [menu addItem:errorItem];
    } else {
        for (FinderWindow *window in self.cachedWindows) {
            NSString *title = window.title.length > 0 ? window.title : @"Untitled Finder Window";
            NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title
                                                          action:@selector(activateFinderWindow:)
                                                   keyEquivalent:@""];
            item.target = self;
            item.representedObject = @(window.windowIndex);
            item.toolTip = window.path.length > 0 ? window.path : title;
            [menu addItem:item];

            if (window.path.length > 0) {
                NSMenuItem *pathItem = [[NSMenuItem alloc] initWithTitle:[window.path stringByAbbreviatingWithTildeInPath]
                                                                  action:nil
                                                           keyEquivalent:@""];
                pathItem.enabled = NO;
                [menu addItem:pathItem];
            }
        }
        self.lastError = nil;
    }

    [menu addItem:[NSMenuItem separatorItem]];

    NSString *footer = self.lastError ?: (self.isRefreshing ? @"Refreshing in background..." : @"Right-click again to reload the window list.");
    NSMenuItem *footerItem = [[NSMenuItem alloc] initWithTitle:footer
                                                        action:nil
                                                 keyEquivalent:@""];
    footerItem.enabled = NO;
    [menu addItem:footerItem];

    return menu;
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    (void)sender;
    (void)flag;
    [self activateNewestFinderWindow];
    return NO;
}

- (void)activateFinderWindow:(NSMenuItem *)sender {
    NSNumber *windowIndex = sender.representedObject;
    if (windowIndex == nil) {
        return;
    }

    NSError *error = nil;
    BOOL didActivate = ActivateFinderWindow(windowIndex.integerValue, &error);
    if (!didActivate) {
        self.lastError = error.localizedDescription ?: @"Could not find that Finder window.";
        LogMessage([NSString stringWithFormat:@"Activation failed: %@", self.lastError]);
        NSBeep();
        return;
    }

    self.lastError = nil;
    LogMessage([NSString stringWithFormat:@"Activated Finder window %ld.", (long)windowIndex.integerValue]);
}

- (void)activateNewestFinderWindow {
    NSError *error = nil;
    BOOL didActivate = ActivateFinderWindow(1, &error);
    if (!didActivate) {
        self.lastError = error.localizedDescription ?: @"Could not find the newest Finder window.";
        LogMessage([NSString stringWithFormat:@"Newest-window activation failed: %@", self.lastError]);
        NSBeep();
        return;
    }

    self.lastError = nil;
    LogMessage(@"Activated newest Finder window.");
}

- (void)refreshFinderWindows {
    if (self.isRefreshing) {
        return;
    }

    if (![self hasAccessibilityAccess]) {
        self.cachedWindows = @[];
        self.lastError = @"Enable Accessibility for FinderDockSwitcher in System Settings > Privacy & Security > Accessibility.";
        return;
    }

    self.isRefreshing = YES;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSError *error = nil;
        NSArray<FinderWindow *> *windows = FetchFinderWindows(&error);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.isRefreshing = NO;
            if (error != nil) {
                self.cachedWindows = @[];
                self.lastError = error.localizedDescription;
                LogMessage([NSString stringWithFormat:@"Refresh failed: %@", self.lastError]);
            } else {
                self.cachedWindows = windows ?: @[];
                self.lastError = nil;
                NSMutableArray<NSString *> *titles = [NSMutableArray array];
                for (FinderWindow *window in self.cachedWindows) {
                    NSString *summary = window.path.length > 0
                        ? [NSString stringWithFormat:@"%@ [%@]", window.title ?: @"<untitled>", window.path]
                        : (window.title ?: @"<untitled>");
                    [titles addObject:summary];
                }
                LogMessage([NSString stringWithFormat:@"Refresh succeeded with %lu window(s): %@",
                            (unsigned long)self.cachedWindows.count,
                            [titles componentsJoinedByString:@", "]]);
            }
        });
    });
}

- (BOOL)hasAccessibilityAccess {
    return AXIsProcessTrusted();
}

- (void)requestAccessibilityIfNeeded {
    if ([self hasAccessibilityAccess]) {
        return;
    }

    NSDictionary *options = @{(__bridge NSString *)kAXTrustedCheckOptionPrompt: @YES};
    AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
    self.lastError = @"Accessibility permission is required to read Finder windows.";
    LogMessage(@"Requested Accessibility permission.");
}

@end

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;

    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        AppDelegate *delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }

    return 0;
}
