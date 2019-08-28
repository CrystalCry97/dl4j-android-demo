package com.yptheangel.dl4jandroid.yolo_objdetection;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;

import com.yptheangel.dl4jandroid.yolo_objdetection.utils.VOCLabelsAndroid;

import org.bytedeco.javacv.OpenCVFrameConverter;
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

import static android.os.Environment.getExternalStoragePublicDirectory;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import static org.opencv.android.Utils.matToBitmap;


public class PredictSingle extends Activity {

    String LOG_TAG = "DEMO_PredictSingle";
    // request code for permission requests to the os for image
    public static final int REQUEST_IMAGE = 100;
    private Uri imageUri;
    Mat myMat = new Mat();
    Mat resizedImage= new Mat();

    //Load the opencv java libs
    static {
    System.loadLibrary("opencv_java");
    }
    org.opencv.core.Mat cvmat = new org.opencv.core.Mat();


    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.5;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;


    org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout=null;
    ComputationGraph model =null;
    VOCLabelsAndroid labels = null;
    String imagepath = null;

    //As mentioned in docs, input image should be converted to RGB and scaled to range 0 to 1.
    NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
    ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_testsingle);

        openCameraIntent();

        Log.i(LOG_TAG, "Image URi is : " + imageUri);
         imagepath = getRealPathFromURI(getApplicationContext(),imageUri);
        Log.i(LOG_TAG, "Imagepath is : " + imagepath);
        if (model==null) {
            try {
                model = ModelSerializer.restoreComputationGraph(getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/tiny-yolo-voc_dl4j_inference.v2.zip", false);
                Log.i(LOG_TAG, "Model is successfully loaded!");
                Log.i(LOG_TAG, model.summary());

                labels = new VOCLabelsAndroid();
                Log.i(LOG_TAG, "Labels are successfully loaded!");
                Log.i(LOG_TAG, labels.getLabels().toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
            yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
            Toast.makeText(getApplicationContext(), "Model loaded successfully.", Toast.LENGTH_LONG).show();
        }
        if (model != null) {
            predictSingle();
        }
    }

    // opens camera for user
    private void openCameraIntent(){

        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From your Camera");
        // tell camera where to store the resulting picture
        imageUri = getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        // start camera, and wait for it to finish
        startActivityForResult(intent, REQUEST_IMAGE);
    }

    private void predictSingle(){
        myMat = imread(imagepath);

        INDArray inputImage = null;
        int w = myMat.cols();
        int h = myMat.rows();
        Log.i(LOG_TAG,"Original Frame Width: "+w);
        Log.i(LOG_TAG,"Original Frame Height: "+h);

        long start_time=System.nanoTime();
        resize(myMat, resizedImage, new Size(tinyyolowidth, tinyyoloheight));
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
//            rescale the coordinates to match the original input image and draw the bounding box and label
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
            rectangle(myMat, new Point(x1, y1), new Point(x2, y2), Scalar.GREEN, 3, 0, 0);
            putText(myMat, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 2, Scalar.BLUE);
//            draw the bounding box and label on the preview image
            int x1_preview = (int) Math.round(tinyyolowidth*xy1[0]/gridWidth);
            int y1_preview = (int) Math.round(tinyyoloheight*xy1[1]/gridHeight);
            int x2_preview = (int) Math.round(tinyyolowidth*xy2[0]/gridWidth);
            int y2_preview = (int) Math.round(tinyyoloheight*xy2[1]/gridHeight);
            rectangle(resizedImage, new Point(x1_preview, y1_preview), new Point(x2_preview, y2_preview), Scalar.GREEN, 2, 0, 0);
            putText(resizedImage, label, new Point(x1_preview + 2, y2_preview - 2), FONT_HERSHEY_DUPLEX, 0.8, Scalar.BLUE);

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

//        Display a low resolution image preview of the prediced output in the app
        ImageView myImageView =  findViewById(R.id.imageView1);
        Bitmap myBitmap = Bitmap.createBitmap(416,
                416, Bitmap.Config.ARGB_8888);
        OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
        OpenCVFrameConverter.ToOrgOpenCvCoreMat converter2 = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();
        cvmat = converter2.convert(converter1.convert(resizedImage));
        matToBitmap(cvmat,myBitmap);
        myImageView.setImageBitmap(myBitmap);

//        save the predicted output on original size frame
        String savedFile = getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)+"/predictedOutput.jpg";
        Log.i(LOG_TAG,"output image saved at: "+savedFile);
        imwrite(savedFile,myMat);
        Log.i(LOG_TAG, "Execution Done");
    }

//    Method to get the exact file path from the Uri
    public String getRealPathFromURI(Context context, Uri contentUri) {
        Cursor cursor = null;
        try {
            String[] proj = { MediaStore.Images.Media.DATA };
            cursor = context.getContentResolver().query(contentUri,  proj, null, null, null);
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            return cursor.getString(column_index);
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
    }
}
