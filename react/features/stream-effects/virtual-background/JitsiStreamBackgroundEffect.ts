import { VIRTUAL_BACKGROUND_TYPE } from '../../virtual-background/constants';

import { chromaWorkerScript } from './ChromaWorker';
import {
    CLEAR_TIMEOUT,
    SET_TIMEOUT,
    TIMEOUT_TICK,
    timerWorkerScript
} from './TimerWorker';

export interface IBackgroundEffectOptions {
    height: number;
    virtualBackground: {
        backgroundType?: string;
        blurValue?: number;
        chromaColor?: string;
        chromaEnabled?: boolean;
        virtualSource?: string;
    };
    width: number;
}

/**
 * Represents a modified MediaStream that adds effects to video background.
 * <tt>JitsiStreamBackgroundEffect</tt> does the processing of the original
 * video stream.
 */
export default class JitsiStreamBackgroundEffect {
    _model: any;
    _options: IBackgroundEffectOptions;
    _stream: any;
    _segmentationPixelCount: number;
    _inputVideoElement: HTMLVideoElement;
    _maskFrameTimerWorker: Worker;
    _chromaWorker: Worker;
    _chromaWorkerInProgress = false;
    _onChromaWorkerDone: ((data: Uint8ClampedArray) => void) | null;
    _outputCanvasElement: HTMLCanvasElement;
    _outputCanvasCtx: CanvasRenderingContext2D | null;
    _compositionCanvas: HTMLCanvasElement;
    _compositionCtx: CanvasRenderingContext2D | null;
    _segmentationMaskCtx: CanvasRenderingContext2D | null;
    _segmentationMask: ImageData;
    _segmentationMaskCanvas: HTMLCanvasElement;
    _virtualImage: HTMLImageElement;
    _virtualVideo: HTMLVideoElement;

    /**
     * Represents a modified video MediaStream track.
     *
     * @class
     * @param {Object} model - Meet model.
     * @param {Object} options - Segmentation dimensions.
     */
    constructor(model: Object, options: IBackgroundEffectOptions) {
        this._options = options;

        const { backgroundType, virtualSource, chromaEnabled } = this._options.virtualBackground;
        const needsImage = backgroundType === VIRTUAL_BACKGROUND_TYPE.IMAGE
            || (chromaEnabled && backgroundType === VIRTUAL_BACKGROUND_TYPE.IMAGE && virtualSource);

        if (needsImage) {
            this._virtualImage = document.createElement('img');
            this._virtualImage.crossOrigin = 'anonymous';
            this._virtualImage.src = virtualSource ?? '';
        }
        this._model = model;
        this._segmentationPixelCount = this._options.width * this._options.height;

        // Bind event handler so it is only bound once for every instance.
        this._onMaskFrameTimer = this._onMaskFrameTimer.bind(this);

        // Workaround for FF issue https://bugzilla.mozilla.org/show_bug.cgi?id=1388974
        this._outputCanvasElement = document.createElement('canvas');
        this._outputCanvasElement.getContext('2d');
        this._inputVideoElement = document.createElement('video');
        this._onChromaWorkerDone = null;
    }

    /**
     * EventHandler onmessage for the maskFrameTimerWorker WebWorker.
     *
     * @private
     * @param {EventHandler} response - The onmessage EventHandler parameter.
     * @returns {void}
     */
    _onMaskFrameTimer(response: { data: { id: number; }; }) {
        if (response.data.id === TIMEOUT_TICK) {
            this._renderMask();
        }
    }

    /**
     * Represents the run post processing.
     *
     * @returns {void}
     */
    runPostProcessing() {
        const track = this._stream.getVideoTracks()[0];
        const { height, width } = track.getSettings() ?? track.getConstraints();
        const { backgroundType } = this._options.virtualBackground;

        if (!this._outputCanvasCtx || !this._compositionCtx) {
            return;
        }

        this._compositionCtx.globalCompositeOperation = 'copy';

        // Draw segmentation mask.
        // Smooth out the edges.
        this._compositionCtx.filter = backgroundType === VIRTUAL_BACKGROUND_TYPE.IMAGE ? 'blur(4px)' : 'blur(8px)';
        this._compositionCtx.drawImage(
            this._segmentationMaskCanvas,
            0,
            0,
            this._options.width,
            this._options.height,
            0,
            0,
            width,
            height
        );
        this._compositionCtx.globalCompositeOperation = 'source-in';
        this._compositionCtx.filter = 'none';

        // Draw the foreground video.
        this._compositionCtx.drawImage(this._inputVideoElement, 0, 0);

        // Draw the background.
        this._compositionCtx.globalCompositeOperation = 'destination-over';
        if (backgroundType === VIRTUAL_BACKGROUND_TYPE.IMAGE) {
            this._compositionCtx.drawImage(
                backgroundType === VIRTUAL_BACKGROUND_TYPE.IMAGE
                    ? this._virtualImage : this._virtualVideo,
                0,
                0,
                width,
                height
            );
        } else {
            this._compositionCtx.filter = `blur(${this._options.virtualBackground.blurValue}px)`;
            this._compositionCtx.drawImage(this._inputVideoElement, 0, 0);
        }

        // Final atomic draw to output canvas to avoid flickering.
        this._outputCanvasCtx.drawImage(this._compositionCanvas, 0, 0);
    }

    /**
     * Converts hex color to RGB.
     *
     * @param {string} hex - Hex color string.
     * @returns {Object} RGB object.
     */
    hexToRgb(hex: string) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);

        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 0, g: 255, b: 0 };
    }

    /**
     * Represents the run Tensorflow Interference.
     *
     * @returns {void}
     */
    runInference() {
        this._model._runInference();
        const outputMemoryOffset = this._model._getOutputMemoryOffset() / 4;

        for (let i = 0; i < this._segmentationPixelCount; i++) {
            const person = this._model.HEAPF32[outputMemoryOffset + i];

            // Sets only the alpha component of each pixel.
            this._segmentationMask.data[(i * 4) + 3] = 255 * person;

        }
        this._segmentationMaskCtx?.putImageData(this._segmentationMask, 0, 0);
    }

    /**
     * Loops function to render the background mask.
     *
     * @private
     * @returns {void}
     */
    async _renderMask() {
        const { backgroundType, virtualSource, chromaEnabled } = this._options.virtualBackground;
        const hasBackground = backgroundType === VIRTUAL_BACKGROUND_TYPE.IMAGE && virtualSource;

        this.resizeSource();
        this.runInference();

        if (chromaEnabled) {
            await this.runChromaKeyLoop(hasBackground);
        } else {
            this.runPostProcessing();
        }

        this._maskFrameTimerWorker.postMessage({
            id: SET_TIMEOUT,
            timeMs: 1000 / 30
        });
    }

    /**
     * Runs the chroma key loop in a worker.
     *
     * @param {string | false | undefined} hasBackground - Background image source.
     * @returns {Promise<void>}
     */
    runChromaKeyLoop(hasBackground: string | false | undefined) {
        if (this._chromaWorkerInProgress) {
            return Promise.resolve();
        }

        const track = this._stream.getVideoTracks()[0];
        const { height, width } = track.getSettings() ?? track.getConstraints();

        if (!this._outputCanvasCtx || !this._compositionCtx || !this._segmentationMaskCtx) {
            return Promise.resolve();
        }

        if (this._compositionCanvas.width !== width || this._compositionCanvas.height !== height) {
            this._compositionCanvas.width = width;
            this._compositionCanvas.height = height;
        }

        const chromaColor = this._options.virtualBackground.chromaColor || '#00ff00';
        const targetRgb = this.hexToRgb(chromaColor);

        const baseThreshold = 15;
        const softnessRange = 30;
        const personThreshold = 0.6;

        this._compositionCtx.globalCompositeOperation = 'copy';
        this._compositionCtx.filter = 'blur(3px)';
        this._compositionCtx.drawImage(
            this._segmentationMaskCanvas,
            0, 0, this._options.width, this._options.height,
            0, 0, width, height
        );
        this._compositionCtx.globalCompositeOperation = 'source-in';
        this._compositionCtx.filter = 'none';
        this._compositionCtx.drawImage(this._inputVideoElement, 0, 0);

        const segImageData = this._segmentationMaskCtx.getImageData(
            0,
            0,
            this._options.width,
            this._options.height
        );

        const imageData = this._compositionCtx.getImageData(
            0,
            0,
            width,
            height
        );

        this._chromaWorkerInProgress = true;

        return new Promise<void>(resolve => {
            this._onChromaWorkerDone = data => {
                const processedImageData = new ImageData(data, width, height);

                this._compositionCtx?.putImageData(processedImageData, 0, 0);

                this._compositionCtx!.globalCompositeOperation = 'destination-over';
                if (hasBackground) {
                    this._compositionCtx?.drawImage(
                        this._virtualImage,
                        0, 0,
                        width,
                        height
                    );
                } else {
                    this._compositionCtx?.clearRect(0, 0, width, height);
                }

                // Final atomic draw to output canvas to avoid flickering.
                this._outputCanvasCtx?.drawImage(this._compositionCanvas, 0, 0);

                this._onChromaWorkerDone = null;
                this._chromaWorkerInProgress = false;
                resolve();
            };

            this._chromaWorker.postMessage({
                data: imageData.data,
                segData: segImageData.data,
                targetRgb,
                baseThreshold,
                softnessRange,
                personThreshold,
                width,
                height,
                segWidth: this._options.width,
                segHeight: this._options.height
            }, [ imageData.data.buffer ]);
        });
    }

    /**
     * Represents the resize source process.
     *
     * @returns {void}
     */
    resizeSource() {
        this._segmentationMaskCtx?.drawImage( // @ts-ignore
            this._inputVideoElement,
            0,
            0,
            this._inputVideoElement.width,
            this._inputVideoElement.height,
            0,
            0,
            this._options.width,
            this._options.height
        );

        const imageData = this._segmentationMaskCtx?.getImageData(
            0,
            0,
            this._options.width,
            this._options.height
        );
        const inputMemoryOffset = this._model._getInputMemoryOffset() / 4;

        for (let i = 0; i < this._segmentationPixelCount; i++) {
            this._model.HEAPF32[inputMemoryOffset + (i * 3)] = Number(imageData?.data[i * 4]) / 255;
            this._model.HEAPF32[inputMemoryOffset + (i * 3) + 1] = Number(imageData?.data[(i * 4) + 1]) / 255;
            this._model.HEAPF32[inputMemoryOffset + (i * 3) + 2] = Number(imageData?.data[(i * 4) + 2]) / 255;
        }
    }

    /**
     * Checks if the local track supports this effect.
     *
     * @param {JitsiLocalTrack} jitsiLocalTrack - Track to apply effect.
     * @returns {boolean} - Returns true if this effect can run on the specified track
     * false otherwise.
     */
    isEnabled(jitsiLocalTrack: any) {
        return jitsiLocalTrack.isVideoTrack() && jitsiLocalTrack.videoType === 'camera';
    }

    /**
     * Starts loop to capture video frame and render the segmentation mask.
     *
     * @param {MediaStream} stream - Stream to be used for processing.
     * @returns {MediaStream} - The stream with the applied effect.
     */
    startEffect(stream: MediaStream) {
        this._stream = stream;

        const firstVideoTrack = this._stream.getVideoTracks()[0];
        const { height, frameRate, width }
            = firstVideoTrack.getSettings ? firstVideoTrack.getSettings() : firstVideoTrack.getConstraints();

        this._maskFrameTimerWorker = new Worker(timerWorkerScript, { name: 'Background effect worker' });
        this._maskFrameTimerWorker.onmessage = this._onMaskFrameTimer;
        this._chromaWorker = new Worker(chromaWorkerScript, { name: 'Chroma key worker' });

        this._chromaWorker.onmessage = event => {
            const { data } = event.data;

            if (this._onChromaWorkerDone) {
                this._onChromaWorkerDone(data);
            }
        };

        this._segmentationMask = new ImageData(this._options.width, this._options.height);
        this._segmentationMaskCanvas = document.createElement('canvas');
        this._segmentationMaskCanvas.width = this._options.width;
        this._segmentationMaskCanvas.height = this._options.height;
        this._segmentationMaskCtx = this._segmentationMaskCanvas.getContext('2d');

        this._outputCanvasElement.width = parseInt(width, 10);
        this._outputCanvasElement.height = parseInt(height, 10);
        this._outputCanvasCtx = this._outputCanvasElement.getContext('2d');

        this._compositionCanvas = document.createElement('canvas');
        this._compositionCanvas.width = this._outputCanvasElement.width;
        this._compositionCanvas.height = this._outputCanvasElement.height;
        this._compositionCtx = this._compositionCanvas.getContext('2d');

        this._inputVideoElement.width = parseInt(width, 10);
        this._inputVideoElement.height = parseInt(height, 10);
        this._inputVideoElement.autoplay = true;
        this._inputVideoElement.srcObject = this._stream;
        this._inputVideoElement.onloadeddata = () => {
            this._maskFrameTimerWorker.postMessage({
                id: SET_TIMEOUT,
                timeMs: 1000 / 30
            });
        };

        return this._outputCanvasElement.captureStream(parseInt(frameRate, 10));
    }

    /**
     * Stops the capture and render loop.
     *
     * @returns {void}
     */
    stopEffect() {
        this._maskFrameTimerWorker.postMessage({
            id: CLEAR_TIMEOUT
        });

        this._maskFrameTimerWorker.terminate();
        this._chromaWorker.terminate();
    }
}
