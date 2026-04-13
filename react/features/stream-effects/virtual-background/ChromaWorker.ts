
/**
 * Worker logic for chroma keying.
 */
const code = `
    let tempBuffer;
    let tempAlpha;
    let lastAlpha;
    let lastData; 
    let lastSmoothingState; 
    let coreShield;

    function rgbToYCbCr(r, g, b) {
        const y = 0.299 * r + 0.587 * g + 0.114 * b;
        const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
        return { y, cb, cr };
    }

    // 3x3 Gaussian kernel for tighter smoothing
    const gauss3 = [
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    ];
    const gaussSum3 = 16;

    onmessage = function(event) {
        const {
            data,
            segData,
            targetRgb,
            baseThreshold,
            softnessRange,
            personThreshold,
            width,
            height,
            segWidth,
            segHeight
        } = event.data;

        const threshold = baseThreshold + softnessRange;
        const pixelCount = width * height;
        const segPixelCount = segWidth * segHeight;

        if (!tempBuffer || tempBuffer.length !== pixelCount) {
            tempBuffer = new Uint8Array(pixelCount);
            tempAlpha = new Uint8Array(pixelCount);
            lastAlpha = new Uint8Array(pixelCount);
            lastSmoothingState = new Uint8Array(pixelCount).fill(1);
            lastData = new Uint8ClampedArray(data.length);
            coreShield = new Uint8Array(segPixelCount);
        }

        // 1. Core Shield Pre-pass (Circular Erosion)
        for (let y = 0; y < segHeight; y++) {
            const yOffset = y * segWidth;
            for (let x = 0; x < segWidth; x++) {
                let minA = 255;
                for (let ky = -2; ky <= 2; ky++) {
                    let ny = y + ky;
                    if (ny < 0 || ny >= segHeight) { minA = 0; break; }
                    const nyOffset = ny * segWidth;
                    for (let kx = -2; kx <= 2; kx++) {
                        if (kx*kx + ky*ky > 6.25) continue;
                        let nx = x + kx;
                        if (nx < 0 || nx >= segWidth) { minA = 0; break; }
                        const a = segData[(nyOffset + nx) * 4 + 3];
                        if (a < minA) minA = a;
                    }
                    if (minA === 0) break;
                }
                coreShield[yOffset + x] = minA;
            }
        }

        const targetYCbCr = rgbToYCbCr(targetRgb.r, targetRgb.g, targetRgb.b);

        // 2. Main Alpha Calculation with Linear Garbage Matte
        for (let y = 0; y < height; y++) {
            const yWidth = y * width;
            const gy = (y * (segHeight - 1)) / (height - 1);
            const gy0 = gy | 0;
            const gy1 = Math.min(gy0 + 1, segHeight - 1);
            const wy = gy - gy0;

            for (let x = 0; x < width; x++) {
                const idx = (yWidth + x) * 4;
                const gx = (x * (segWidth - 1)) / (width - 1);
                const gx0 = gx | 0;
                const gx1 = Math.min(gx0 + 1, segWidth - 1);
                const wx = gx - gx0;

                // Bilinear Core Shield
                const c00 = coreShield[gy0 * segWidth + gx0];
                const c10 = coreShield[gy0 * segWidth + gx1];
                const c01 = coreShield[gy1 * segWidth + gx0];
                const c11 = coreShield[gy1 * segWidth + gx1];
                const shieldVal = (c00*(1-wx)+c10*wx)*(1-wy) + (c01*(1-wx)+c11*wx)*wy;

                if (shieldVal > 240) {
                    tempAlpha[yWidth + x] = 255;
                    continue;
                }

                // Bilinear Garbage Matte with Smooth Falloff (Prevents stairs)
                const s00 = segData[(gy0 * segWidth + gx0) * 4 + 3];
                const s10 = segData[(gy0 * segWidth + gx1) * 4 + 3];
                const s01 = segData[(gy1 * segWidth + gx0) * 4 + 3];
                const s11 = segData[(gy1 * segWidth + gx1) * 4 + 3];
                const segA = (s00*(1-wx)+s10*wx)*(1-wy) + (s01*(1-wx)+s11*wx)*wy;

                let matteAtten = 1.0;
                if (segA < 38) { // wider transition for matte
                    matteAtten = segA / 38.0;
                }
                if (matteAtten === 0) {
                    tempAlpha[yWidth + x] = 0;
                    continue;
                }

                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const pixelYCbCr = rgbToYCbCr(r, g, b);

                const dCb = pixelYCbCr.cb - targetYCbCr.cb;
                const dCr = pixelYCbCr.cr - targetYCbCr.cr;
                let dist = Math.sqrt(dCb * dCb + dCr * dCr);

                let alphaValue = 255;
                if (dist < baseThreshold) {
                    alphaValue = 0;
                } else if (dist < threshold) {
                    const t = (dist - baseThreshold) / softnessRange;
                    alphaValue = (t * t * (3 - 2 * t) * 255) | 0;
                }

                // Luma Protection
                if (pixelYCbCr.y > 180) {
                    let boost = (pixelYCbCr.y - 180) / (255 - 180) * 80;
                    alphaValue = Math.min(255, alphaValue + boost) | 0;
                }

                tempAlpha[yWidth + x] = (alphaValue * matteAtten) | 0;
            }
        }

        // 3. Post-Processing: Tight Erosion & 3x3 Smoothing
        for (let y = 0; y < height; y++) {
            const yWidth = y * width;
            const isHead = y < height * 0.35;
            for (let x = 0; x < width; x++) {
                if (tempAlpha[yWidth + x] === 0) {
                    tempBuffer[yWidth + x] = 0;
                    continue;
                }
                if (isHead) {
                    tempBuffer[yWidth + x] = tempAlpha[yWidth + x];
                } else {
                    let minA = 255;
                    for (let ky = -1; ky <= 1; ky++) {
                        let ny = y + ky;
                        if (ny < 0 || ny >= height) ny = y;
                        for (let kx = -1; kx <= 1; kx++) {
                            if (kx*kx + ky*ky > 1.1) continue;
                            let nx = x + kx;
                            if (nx < 0 || nx >= width) nx = x;
                            const a = tempAlpha[ny * width + nx];
                            if (a < minA) minA = a;
                        }
                    }
                    tempBuffer[yWidth + x] = minA;
                }
            }
        }

        // Selective 3x3 Gaussian Blur (Tighter border radius)
        for (let y = 0; y < height; y++) {
            const yWidth = y * width;
            for (let x = 0; x < width; x++) {
                const centerA = tempBuffer[yWidth + x];
                // Only blur the actual edge to prevent "glow" and internal smudging
                if (centerA > 10 && centerA < 245) {
                    let sum = 0;
                    let wIdx = 0;
                    for (let ky = -1; ky <= 1; ky++) {
                        let ny = y + ky;
                        if (ny < 0) ny = 0; else if (ny >= height) ny = height - 1;
                        for (let kx = -1; kx <= 1; kx++) {
                            let nx = x + kx;
                            if (nx < 0) nx = 0; else if (nx >= width) nx = width - 1;
                            sum += tempBuffer[ny * width + nx] * gauss3[wIdx++];
                        }
                    }
                    tempAlpha[yWidth + x] = (sum / gaussSum3) | 0;
                } else {
                    tempAlpha[yWidth + x] = centerA;
                }
            }
        }

        // Final Alpha Pass: Precision Contrast & Smoothing
        for (let i = 0; i < pixelCount; i++) {
            let a = tempAlpha[i] / 255.0;
            // Sharper Crunch (1.8x) to remove glow/smudge
            a = (a - 0.5) * 1.8 + 0.5;
            if (a < 0) a = 0; if (a > 1) a = 1;
            // Sub-pixel smoothing
            a = 3 * a * a - 2 * a * a * a;
            
            let currentAlpha = (a * 255.0) | 0;
            const prevA = lastAlpha[i];
            const diff = Math.abs(currentAlpha - prevA) / 255.0;
            
            let isSmoothing = lastSmoothingState[i];
            if (diff > 0.12) isSmoothing = 0;
            else if (diff < 0.04) isSmoothing = 1;

            if (isSmoothing === 1) {
                currentAlpha = (currentAlpha * 0.65 + prevA * 0.35) | 0;
            }
            data[i * 4 + 3] = currentAlpha;
            lastAlpha[i] = currentAlpha;
            lastSmoothingState[i] = isSmoothing;
        }

        // 4. White-Shifted Polish
        for (let i = 0; i < data.length; i += 4) {
            const alpha = data[i + 3];
            if (alpha === 0) continue;

            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];
            
            const maxG = (r * 0.7 + b * 0.3) | 0;
            if (g > maxG) {
                const prevG = lastData[i + 1];
                const isSmoothing = lastSmoothingState[i / 4];
                g = isSmoothing ? (maxG * 0.7 + prevG * 0.3) | 0 : maxG;
            }

            // Tight Light Wrap toward White
            if (alpha < 235) {
                const wrapStrength = (1.0 - alpha / 255.0); 
                r = r + (255 - r) * wrapStrength * 0.5; 
                g = g + (255 - g) * wrapStrength * 0.5;
                b = b + (255 - b) * wrapStrength * 0.5;
            }

            data[i] = r | 0;
            data[i + 1] = g | 0;
            data[i + 2] = b | 0;
            lastData[i+1] = g;
        }

        self.postMessage({ data }, [data.buffer]);
    };
`;

// @ts-ignore
export const chromaWorkerScript = URL.createObjectURL(new Blob([ code ], { type: 'application/javascript' }));
