package com.yptheangel.dl4jandroid.yolo_objdetection;

import android.app.Activity;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.Nullable;
import android.util.Log;
import android.widget.Toast;

import com.yptheangel.dl4jandroid.yolo_objdetection.utils.StorageHelper;

import com.yptheangel.dl4jandroid.yolo_objdetection.utils.VOCLabelsAndroid;

import static android.os.Environment.getExternalStoragePublicDirectory;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.IOException;
import java.util.ArrayList;

public class ObjDetection extends Activity implements CvCameraPreview.CvCameraViewListener {
//    private CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;
    private CvCameraPreview cameraView;

    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.5;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;

    String LOG_TAG="DEMO_ObjDetection";

    ComputationGraph initializedModel =null;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_objdetection);

        cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {

//            ZooModel model = TinyYOLO.builder().numClasses(0).build();

//            try {
////                initializedModel = (ComputationGraph) model.initPretrained();
//                initializedModel =ModelSerializer.restoreComputationGraph(getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/tiny-yolo-voc_dl4j_inference.v2.zip");
//                Log.i(LOG_TAG,initializedModel.summary());
//
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//            if (initializedModel != null) {
//                Log.i(LOG_TAG,initializedModel.summary());
//                Log.i(LOG_TAG,"Yeeha!");
//            }
//            NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
//            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

                try {
                    ArrayList<String> labels = new VOCLabelsAndroid().getLabels();

                    Log.i(LOG_TAG,"Labels loaded!");
                    Log.i(LOG_TAG,labels.toString());

                } catch (IOException e) {
                    e.printStackTrace();
                }

                return null;
            }
        }.execute();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        absoluteFaceSize = (int) (width * 0.32f);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(Mat rgbaMat) {

//        if (faceDetector != null) {
//            Mat grayMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
//
//
//            cvtColor(rgbaMat, grayMat, CV_BGR2GRAY);
//
//            RectVector faces = new RectVector();
//            faceDetector.detectMultiScale(grayMat, faces, 1.25f, 3, 1,
//                    new Size(absoluteFaceSize, absoluteFaceSize),
//                    new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));
//            if (faces.size() == 1) {
//                int x = faces.get(0).x();
//                int y = faces.get(0).y();
//                int w = faces.get(0).width();
//                int h = faces.get(0).height();
//                rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), Scalar.GREEN, 2, LINE_8, 0);
//            }
//            grayMat.release();
//        }

        return rgbaMat;
    }
}