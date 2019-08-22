package com.yptheangel.dl4jandroid.yolo_objdetection;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.Nullable;

import com.yptheangel.dl4jandroid.yolo_objdetection.utils.StorageHelper;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect;

import static org.bytedeco.javacpp.opencv_core.LINE_8;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

public class ObjDetection extends Activity implements CvCameraPreview.CvCameraViewListener {
    private opencv_objdetect.CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;
    private CvCameraPreview cameraView;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_objdetection);

        cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                faceDetector = StorageHelper.loadClassifierCascade(ObjDetection.this, R.raw.frontalface);
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
    public opencv_core.Mat onCameraFrame(opencv_core.Mat rgbaMat) {
        if (faceDetector != null) {
            opencv_core.Mat grayMat = new opencv_core.Mat(rgbaMat.rows(), rgbaMat.cols());

            cvtColor(rgbaMat, grayMat, CV_BGR2GRAY);

            opencv_core.RectVector faces = new opencv_core.RectVector();
            faceDetector.detectMultiScale(grayMat, faces, 1.25f, 3, 1,
                    new opencv_core.Size(absoluteFaceSize, absoluteFaceSize),
                    new opencv_core.Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));
            if (faces.size() == 1) {
                int x = faces.get(0).x();
                int y = faces.get(0).y();
                int w = faces.get(0).width();
                int h = faces.get(0).height();
                rectangle(rgbaMat, new opencv_core.Point(x, y), new opencv_core.Point(x + w, y + h), opencv_core.Scalar.GREEN, 2, LINE_8, 0);
            }

            grayMat.release();
        }

        return rgbaMat;
    }
}
