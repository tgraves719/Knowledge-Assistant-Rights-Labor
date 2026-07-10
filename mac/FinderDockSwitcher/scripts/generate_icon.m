#import <AppKit/AppKit.h>

static NSColor *Color(CGFloat r, CGFloat g, CGFloat b, CGFloat a) {
    return [NSColor colorWithCalibratedRed:r / 255.0
                                     green:g / 255.0
                                      blue:b / 255.0
                                     alpha:a];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Missing output path.\n");
            return 1;
        }

        NSString *outputPath = [NSString stringWithUTF8String:argv[1]];
        const CGFloat size = 1024.0;
        NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:NULL
                          pixelsWide:(NSInteger)size
                          pixelsHigh:(NSInteger)size
                       bitsPerSample:8
                     samplesPerPixel:4
                            hasAlpha:YES
                            isPlanar:NO
                      colorSpaceName:NSCalibratedRGBColorSpace
                         bytesPerRow:0
                        bitsPerPixel:0];
        if (bitmap == nil) {
            fprintf(stderr, "Could not create bitmap.\n");
            return 1;
        }
        NSGraphicsContext *context = [NSGraphicsContext graphicsContextWithBitmapImageRep:bitmap];
        if (context == nil) {
            fprintf(stderr, "Could not create graphics context.\n");
            return 1;
        }
        [NSGraphicsContext saveGraphicsState];
        [NSGraphicsContext setCurrentContext:context];

        NSRect canvas = NSMakeRect(0, 0, size, size);
        NSGradient *background = [[NSGradient alloc] initWithColorsAndLocations:
                                  Color(18, 94, 198, 1.0), 0.0,
                                  Color(56, 155, 255, 1.0), 0.55,
                                  Color(152, 228, 255, 1.0), 1.0,
                                  nil];
        NSBezierPath *roundedBackground = [NSBezierPath bezierPathWithRoundedRect:NSInsetRect(canvas, 72, 72)
                                                                          xRadius:220
                                                                          yRadius:220];
        [background drawInBezierPath:roundedBackground angle:90.0];

        [[Color(255, 255, 255, 0.16) colorWithAlphaComponent:0.22] setFill];
        NSBezierPath *glow = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(160, 580, 700, 280)];
        [glow fill];

        NSShadow *shadow = [[NSShadow alloc] init];
        shadow.shadowBlurRadius = 30;
        shadow.shadowOffset = NSMakeSize(0, -12);
        shadow.shadowColor = [Color(8, 39, 86, 1.0) colorWithAlphaComponent:0.28];
        [shadow set];

        NSBezierPath *rearWindow = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(260, 290, 430, 340)
                                                                   xRadius:52
                                                                   yRadius:52];
        [Color(221, 244, 255, 0.96) setFill];
        [rearWindow fill];

        NSShadow *frontShadow = [[NSShadow alloc] init];
        frontShadow.shadowBlurRadius = 36;
        frontShadow.shadowOffset = NSMakeSize(0, -14);
        frontShadow.shadowColor = [Color(8, 39, 86, 1.0) colorWithAlphaComponent:0.36];
        [frontShadow set];

        NSBezierPath *folderBody = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(188, 214, 620, 432)
                                                                   xRadius:66
                                                                   yRadius:66];
        [Color(255, 213, 79, 1.0) setFill];
        [folderBody fill];

        NSBezierPath *folderTab = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(248, 520, 246, 116)
                                                                  xRadius:42
                                                                  yRadius:42];
        [Color(255, 232, 144, 1.0) setFill];
        [folderTab fill];

        NSBezierPath *frontWindow = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(328, 314, 378, 230)
                                                                    xRadius:38
                                                                    yRadius:38];
        [Color(248, 252, 255, 0.98) setFill];
        [frontWindow fill];

        NSRect headerRect = NSMakeRect(328, 486, 378, 58);
        NSBezierPath *header = [NSBezierPath bezierPathWithRoundedRect:headerRect xRadius:38 yRadius:38];
        [Color(72, 170, 255, 1.0) setFill];
        [header fill];

        [[Color(92, 122, 152, 1.0) colorWithAlphaComponent:0.72] setFill];
        [[NSBezierPath bezierPathWithRoundedRect:NSMakeRect(372, 420, 290, 24) xRadius:12 yRadius:12] fill];
        [[NSBezierPath bezierPathWithRoundedRect:NSMakeRect(372, 380, 210, 24) xRadius:12 yRadius:12] fill];

        NSBezierPath *spark = [NSBezierPath bezierPath];
        [spark moveToPoint:NSMakePoint(770, 382)];
        [spark lineToPoint:NSMakePoint(816, 428)];
        [spark lineToPoint:NSMakePoint(770, 474)];
        [spark lineToPoint:NSMakePoint(724, 428)];
        [spark closePath];
        [Color(255, 255, 255, 0.92) setFill];
        [spark fill];

        [NSGraphicsContext restoreGraphicsState];

        NSData *pngData = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
        if (pngData == nil) {
            fprintf(stderr, "Could not encode PNG.\n");
            return 1;
        }

        NSError *writeError = nil;
        BOOL didWrite = [pngData writeToFile:outputPath options:NSDataWritingAtomic error:&writeError];
        if (!didWrite) {
            fprintf(stderr, "Could not write PNG: %s\n", writeError.localizedDescription.UTF8String);
            return 1;
        }

        return 0;
    }
}
