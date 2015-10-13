
// Volume utilities
// intended for use with data augmentation
// crop is the size of output
// dx,dy are offset wrt incoming volume, of the shift
// fliplr is boolean on whether we also want to flip left<->right
func augment(V: Vol, crop:Int, dx: Int, dy: Int, fliplr: Bool) {
    // note assumes square outputs of size crop x crop
    if(fliplr==null) { var fliplr = false; }
    if(dx==null) { var dx = global.randi(0, V.sx - crop); }
    if(dy==null) { var dy = global.randi(0, V.sy - crop); }
    
    // randomly sample a crop in the input volume
    var W;
    if(crop !== V.sx || dx!==0 || dy!==0) {
        W = Vol(crop, crop, V.depth, 0.0);
        for(var x=0;x<crop;x++) {
            for(var y=0;y<crop;y++) {
                if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) {
                    continue; // oob
                }
                for(var d=0;d<V.depth;d++) {
                    W.set(x,y,d,V.get(x+dx,y+dy,d)); // copy data over
                }
            }
        }
    } else {
        W = V;
    }
    
    if(fliplr) {
        // flip volume horizontally
        var W2 = W.cloneAndZero();
        for(var x=0;x<W.sx;x++) {
            for(var y=0;y<W.sy;y++) {
                for(var d=0;d<W.depth;d++) {
                    W2.set(x,y,d,W.get(W.sx - x - 1,y,d)); // copy data over
                }
            }
        }
        W = W2; //swap
    }
    return W;
}

import UIKit
import CoreGraphics

// img is a DOM element that contains a loaded image
// returns a Vol of size (W, H, 4). 4 is for RGBA
func img_to_vol(img: UIImage, convert_grayscale: Bool = false) -> Vol {
    
    var uiimage = UIImage(contentsOfFile: "/PATH/TO/image.png")
    var image = uiimage.CGImage
    
    
    let width = CGImageGetWidth(image)
    let height = CGImageGetHeight(image)
    let colorspace = CGColorSpaceCreateDeviceRGB()
    let bytesPerRow = (4 * width);
    let bitsPerComponent :UInt = 8
    let pixels = UnsafePointer<UInt8>(malloc(width*height*4))
    
    
    var context = CGBitmapContextCreate(pixels, width, height, bitsPerComponent, bytesPerRow, colorspace,
        CGBitmapInfo());
    
    CGContextDrawImage(context, CGRectMake(0, 0, CGFloat(width), CGFloat(height)), image)
    
    
    for x in 0..width {
        for y in 0..height {
            //Here is your raw pixels
            let offset = 4*((Int(width) * Int(y)) + Int(x))
            let alpha = pixels[offset]
            let red = pixels[offset+1]
            let green = pixels[offset+2]
            let blue = pixels[offset+3]
        }
    }
    ////////////////////////////
    // prepare the input: get pixels and normalize them
    var pv = []
    for(var i=0;i<width*height;i++) {
        pv.push(p[i]/255.0-0.5); // normalize image pixels to [-0.5, 0.5]
    }
    
    var x = Vol(W, H, 4, 0.0); //input volume (image)
    x.w = pv;
    
    if(convert_grayscale) {
        // flatten into depth=1 array
        var x1 = Vol(width, height, 1, 0.0);
        for(var i=0;i<width;i++) {
            for(var j=0;j<height;j++) {
                x1.set(i,j,0,x.get(i,j,0));
            }
        }
        x = x1;
    }
    
    return x;
}


func vol_to_img(){
    public struct PixelData {
        var a: UInt8
        var r: UInt8
        var g: UInt8
        var b: UInt8
    }
    
    var pixels = [PixelData]()
    
    let red = PixelData(a: 255, r: 255, g: 0, b: 0)
    let green = PixelData(a: 255, r: 0, g: 255, b: 0)
    let blue = PixelData(a: 255, r: 0, g: 0, b: 255)
    
    for i in 1...300 {
        pixels.append(red)
    }
    for i in 1...300 {
        pixels.append(green)
    }
    for i in 1...300 {
        pixels.append(blue)
    }
    
    let image = imageFromARGB32Bitmap(pixels, 30, 30)
}
