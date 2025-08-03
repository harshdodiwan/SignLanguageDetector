package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class ObjectDetector {

    private Interpreter interpreter;
    private Interpreter classifier;
    private List<String> labelList;
    private int inputSize;
    private final int PIXEL_SIZE = 3;
    private final float IMAGE_STD = 255.0f;
    private GpuDelegate gpuDelegate;
    private int height = 0;
    private int width = 0;
    private int classificationInputSize;

    public ObjectDetector(AssetManager assetManager, String detectionModelPath, String labelPath,
                          int inputSize, String classificationModelPath, int classificationInputSize) throws IOException {

        this.inputSize = inputSize;
        this.classificationInputSize = classificationInputSize;

        Interpreter.Options detectionOptions = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        detectionOptions.addDelegate(gpuDelegate);
        detectionOptions.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(assetManager, detectionModelPath), detectionOptions);

        labelList = loadLabelList(assetManager, labelPath);

        Interpreter.Options classificationOptions = new Interpreter.Options();
        classificationOptions.setNumThreads(2);
        classifier = new Interpreter(loadModelFile(assetManager, classificationModelPath), classificationOptions);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        return labels;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();




        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Mat recognizeImage(Mat inputMat) {
        Mat rotatedMat = new Mat();
        Core.flip(inputMat.t(), rotatedMat, 1);

        Bitmap bitmap = Bitmap.createBitmap(rotatedMat.cols(), rotatedMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotatedMat, bitmap);

        height = bitmap.getHeight();
        width = bitmap.getWidth();

        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(scaledBitmap, inputSize);

        float[][][] boxes = new float[1][10][4];
        float[][] scores = new float[1][10];
        float[][] classes = new float[1][10];

        Object[] input = { inputBuffer };
        Map<Integer, Object> outputMap = new TreeMap<>();
        outputMap.put(0, boxes);
        outputMap.put(1, classes);
        outputMap.put(2, scores);

        interpreter.runForMultipleInputsOutputs(input, outputMap);

        for (int i = 0; i < 10; i++) {
            float score = scores[0][i];
            if (Float.isNaN(score) || score < 0.5f || score > 1.0f) {
                continue;
            }

            float[] box = boxes[0][i];

            float y1 = box[0] * height;
            float x1 = box[1] * width;
            float y2 = box[2] * height;
            float x2 = box[3] * width;

            x1 = Math.max(x1, 0);
            y1 = Math.max(y1, 0);
            x2 = Math.min(x2, width);
            y2 = Math.min(y2, height);

            int rectWidth = (int)(x2 - x1);
            int rectHeight = (int)(y2 - y1);

            if (rectWidth <= 0 || rectHeight <= 0) {
                continue;  // skip invalid box
            }

            Rect roi = new Rect((int) x1, (int) y1, rectWidth, rectHeight);
            Mat cropped = new Mat(rotatedMat, roi).clone();

            if (cropped.cols() == 0 || cropped.rows() == 0) {
                continue;
            }

            Bitmap croppedBitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped, croppedBitmap);

            Bitmap resized = Bitmap.createScaledBitmap(croppedBitmap, classificationInputSize, classificationInputSize, false);
            ByteBuffer classBuffer = convertBitmapToByteBuffer(resized, classificationInputSize);

            float[][] classOutput = new float[1][1];
            classifier.run(classBuffer, classOutput);
            int predictedIndex = Math.round(classOutput[0][0]);

            String label;
            if (predictedIndex >= 0 && predictedIndex < 26) {
                label = String.valueOf((char) ('A' + predictedIndex));
            } else {
                label = "?";
            }


            Imgproc.putText(rotatedMat, label, new Point(x1 + 10, y1 + 40), Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(255, 255, 255), 2);
            Imgproc.rectangle(rotatedMat, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2);

            // Optionally recycle bitmaps to free memory:
            croppedBitmap.recycle();
            resized.recycle();
        }

        Mat outputMat = new Mat();
        Core.flip(rotatedMat.t(), outputMat, 0);
        return outputMat;
    }



    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int inputSize) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)));
                byteBuffer.putFloat((((val >> 8) & 0xFF)));
                byteBuffer.putFloat(((val & 0xFF)));
            }
        }

        return byteBuffer;
    }

}
