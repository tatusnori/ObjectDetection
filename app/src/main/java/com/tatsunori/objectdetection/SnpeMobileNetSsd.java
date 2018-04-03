package com.tatsunori.objectdetection;

import android.app.Application;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.qualcomm.qti.snpe.SNPE.LOG_TAG;

public class SnpeMobileNetSsd {
    private static final int INPUT_SIZE = 300;
    private NeuralNetwork mNetwork;

    public SnpeMobileNetSsd(Application application, NeuralNetwork.Runtime targetRiuntime, File model) {
        mNetwork = null;
        try {
            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(application)
                    .setDebugEnabled(true)
                    .setRuntimeOrder(NeuralNetwork.Runtime.GPU)
                    .setOutputLayers("Postprocessor/BatchMultiClassNonMaxSuppression", "add_6")
                    .setCpuFallbackEnabled(true)
                    .setModel(model);
            mNetwork = builder.build();
        } catch (IllegalStateException | IOException e) {
            Log.e(LOG_TAG, e.getMessage(), e);
        }
    }

    public List<DetectionInfo> detection(Bitmap bitmap) {
        Bitmap inputBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

        final FloatTensor tensor = mNetwork.createFloatTensor(
                mNetwork.getInputTensorsShapes().get("Preprocessor/sub:0"));

        final int[] dimensions = tensor.getShape();

        writeRgbBitmapAsFloat(inputBitmap, tensor);

        final Map<String, FloatTensor> inputs = new HashMap<>();
        inputs.put("Preprocessor/sub:0", tensor);

        final Map<String, FloatTensor> outputs = mNetwork.execute(inputs);

        FloatTensor scoresTensor = outputs.get("Postprocessor/BatchMultiClassNonMaxSuppression_scores");
        float[] scores = new float[scoresTensor.getSize()];
        scoresTensor.read(scores, 0, scoresTensor.getSize());

        FloatTensor boxesTensor = outputs.get("Postprocessor/BatchMultiClassNonMaxSuppression_boxes");
        float[] boxes = new float[boxesTensor.getSize()];
        boxesTensor.read(boxes, 0, boxesTensor.getSize());
        FloatTensor classesTensor = outputs.get("detection_classes:0");
        float[] classesId = new float[classesTensor.getSize()];
        classesTensor.read(classesId, 0, classesTensor.getSize());

        List<DetectionInfo> infoList = new ArrayList<>();
        for (int i = 0;i < scores.length;i++) {
            if (scores[i] > 0.2) {
                final RectF detection =
                        new RectF(
                                boxes[4 * i + 1] * INPUT_SIZE,
                                boxes[4 * i] * INPUT_SIZE,
                                boxes[4 * i + 3] * INPUT_SIZE,
                                boxes[4 * i + 2] * INPUT_SIZE);
                DetectionInfo info = new DetectionInfo("", scores[i], detection);
                infoList.add(info);
                Log.d("@@@", "classes:" + classesId[i] + " sores" + scores[i]);
            }
        }


        return infoList;
    }

    private void writeRgbBitmapAsFloat(Bitmap image, FloatTensor tensor) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int rgb = pixels[y * image.getWidth() + x];
                float b = ((rgb)       & 0xFF);
                float g = ((rgb >>  8) & 0xFF);
                float r = ((rgb >> 16) & 0xFF);
                float[] pixelFloats = {b, g, r};
                tensor.write(pixelFloats, 0, pixelFloats.length, y, x);
            }
        }
    }
}
