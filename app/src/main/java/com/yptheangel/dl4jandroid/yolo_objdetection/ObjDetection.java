package com.yptheangel.dl4jandroid.yolo_objdetection;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import com.yptheangel.dl4jandroid.yolo_objdetection.utils.VOCLabelsAndroid;

import static android.os.Environment.getExternalStoragePublicDirectory;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

    public class ObjDetection extends Activity implements CvCameraPreview.CvCameraViewListener {

    private CvCameraPreview cameraView;
    String LOG_TAG="DEMO_ObjDetection";

    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.5;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;

    org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout=null;

    ComputationGraph model =null;
    VOCLabelsAndroid labels = null;

    //As mentioned in docs, input image should be converted to RGB and scaled to range 0 to 1.
    NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
    ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_objdetection);

        cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        LoadFiles dependencies_loader = new LoadFiles();
        dependencies_loader.execute();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(Mat rgbaMat)  {

        int w = rgbaMat.cols();
        int h = rgbaMat.rows();
        INDArray inputImage = null;
        Mat resizedImage = new Mat();

        if (model != null) {
            long start_time=System.nanoTime();
            putText(rgbaMat, "Model loaded", new Point(70, 40), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
            resize(rgbaMat, resizedImage, new Size(tinyyolowidth, tinyyoloheight));
            long puttextresize= System.nanoTime();
            long elapsed_time_puttextresize= puttextresize-start_time;
            try {
                inputImage = loader.asMatrix(resizedImage);
            } catch (IOException e) {
                e.printStackTrace();
            }

            scaler.transform(inputImage);
            long matrixload_scaler= System.nanoTime();
            long elapsed_time_matrixload_scaler= matrixload_scaler-puttextresize;

            INDArray outputs = model.outputSingle(inputImage);
            List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);

            //List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);

            long predict= System.nanoTime();
            long elapsed_time_predict= predict-matrixload_scaler;
            Log.i(LOG_TAG, "Before for loop: " + objs.toString());

            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.getLabel(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                rectangle(rgbaMat, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
                putText(rgbaMat, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
            }
            long draw = System.nanoTime();
            long elapsed_time_draw = draw - predict;
            long elasped_time_total = elapsed_time_puttextresize + elapsed_time_matrixload_scaler + elapsed_time_predict + elapsed_time_draw;

            long elapsed_time_puttextresize_ms = TimeUnit.MILLISECONDS.convert(elapsed_time_puttextresize, TimeUnit.NANOSECONDS);
            long elapsed_time_matrixload_scaler_ms = TimeUnit.MILLISECONDS.convert(elapsed_time_matrixload_scaler, TimeUnit.NANOSECONDS);
            long elapsed_time_matrixload_predict_ms = TimeUnit.MILLISECONDS.convert(elapsed_time_predict, TimeUnit.NANOSECONDS);
            long elapsed_time_matrixload_draw_ms = TimeUnit.MILLISECONDS.convert(elapsed_time_draw, TimeUnit.NANOSECONDS);
            long elasped_time_total_ms = TimeUnit.MILLISECONDS.convert(elasped_time_total, TimeUnit.NANOSECONDS);

            Log.i(LOG_TAG, "puttextresize: " + elapsed_time_puttextresize_ms + " ms");
            Log.i(LOG_TAG, "scaler: " + elapsed_time_matrixload_scaler_ms + " ms");
            Log.i(LOG_TAG, "predict: " + elapsed_time_matrixload_predict_ms + " ms");
            Log.i(LOG_TAG, "drawBB: " + elapsed_time_matrixload_draw_ms + " ms");
            Log.i(LOG_TAG, "total: " + elasped_time_total_ms + " ms");
        }
        return rgbaMat;
    }

        private class LoadFiles extends AsyncTask<Void, Void, Void> {
        @Override
        protected void onPreExecute() {
        }

        @Override
        protected Void doInBackground(Void... params) {

            try {
                model =ModelSerializer.restoreComputationGraph(getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/tiny-yolo-voc_dl4j_inference.v2.zip",false);
                Log.i(LOG_TAG,"Model is successfully loaded!");
                Log.i(LOG_TAG,model.summary());

                labels = new VOCLabelsAndroid();
                Log.i(LOG_TAG,"Labels are successfully loaded!");
                Log.i(LOG_TAG,labels.getLabels().toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
            yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);

            return null;
        }

        @Override
        public void onPostExecute(Void result) {
        }
    }
}



