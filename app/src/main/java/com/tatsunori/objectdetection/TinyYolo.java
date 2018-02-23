package com.tatsunori.objectdetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

public class TinyYolo {
    private static final String TAG = "ObjectDetection";

    private static final int MAX_RESULTS = 5;

    private static final int NUM_CLASSES = 20;

    private static final int NUM_BOXES_PER_BLOCK = 5;

    private static final double[] ANCHORS = {
            1.08, 1.19,
            3.42, 4.41,
            6.63, 11.38,
            9.42, 5.11,
            16.62, 10.52
    };

    private static final String[] LABELS = {
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
    };
//    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final String YOLO_MODEL_FILE = "file:///android_asset/tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    // Config values.
    private String inputName;

    // Pre-allocated buffers.
    private int[] mIntValues;
    private float[] mFloatValues;

    private boolean logStats = false;

    private TensorFlowInferenceInterface mInferenceInterface;

    public TinyYolo(Context context) {
        mInferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), YOLO_MODEL_FILE);
        mIntValues = new int[YOLO_INPUT_SIZE * YOLO_INPUT_SIZE];
        mFloatValues = new float[YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3];
    }

    public List<DetectionInfo> detection(Bitmap bitmap) {
        Bitmap inputBitmap = Bitmap.createScaledBitmap(bitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, false);
        inputBitmap.getPixels(mIntValues, 0, inputBitmap.getWidth(), 0, 0, inputBitmap.getWidth(), inputBitmap.getHeight());

        for (int i = 0; i < mIntValues.length; ++i) {
            mFloatValues[i * 3 + 0] = ((mIntValues[i] >> 16) & 0xFF) / 255.0f;
            mFloatValues[i * 3 + 1] = ((mIntValues[i] >> 8) & 0xFF) / 255.0f;
            mFloatValues[i * 3 + 2] = (mIntValues[i] & 0xFF) / 255.0f;
        }

        mInferenceInterface.feed(YOLO_INPUT_NAME, mFloatValues, 1, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3);

        String[] outputNames = YOLO_OUTPUT_NAMES.split(",");
        mInferenceInterface.run(outputNames, logStats);

        Log.d(TAG, "logstats:" + logStats);

        final int gridWidth = inputBitmap.getWidth() / YOLO_BLOCK_SIZE;
        final int gridHeight = inputBitmap.getHeight() / YOLO_BLOCK_SIZE;
        final float[] output =
                new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK * 60];

        mInferenceInterface.fetch(outputNames[0], output);

        List<DetectionInfo> infoList = new ArrayList<>();

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    float scaleW = (float)bitmap.getWidth() / (float)YOLO_INPUT_SIZE;
                    float scaleH = (float)bitmap.getHeight() / (float)YOLO_INPUT_SIZE;
                    final float xPos = (x + expit(output[offset + 0])) * YOLO_BLOCK_SIZE * scaleW;
                    final float yPos = (y + expit(output[offset + 1])) * YOLO_BLOCK_SIZE * scaleH;

                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * YOLO_BLOCK_SIZE * scaleW;
                    final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * YOLO_BLOCK_SIZE * scaleH;

                    final RectF rect =
                            new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                    Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                    final float confidence = expit(output[offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = output[offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.2) {
                        DetectionInfo info = new DetectionInfo(LABELS[detectedClass], confidence, rect);
                        infoList.add(info);
                        Log.i(TAG,
                                String.format("%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect));
                    }
                }
            }
        }
        return infoList;
    }

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }
}
